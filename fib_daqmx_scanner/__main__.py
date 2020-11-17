import sys
import signal
from pathlib import Path
from enum import IntEnum
import threading
from queue import Queue, Empty
import ctypes

import appdirs

import toml
import numpy as np
from qtutils.qt import QtCore, QtGui, QtWidgets
from qtutils import inmain, inmain_later, inmain_decorator, UiLoader
import qtutils.icons
import pyqtgraph as pg

from PyDAQmx import Task
from PyDAQmx.DAQmxFunctions import (
    SamplesNotYetAvailableError,
    InvalidTaskError,
    DAQmxGetSampClkTerm,
    DAQmxGetReadCurrReadPos
)
from PyDAQmx.DAQmxConstants import (
    DAQmx_Val_RSE,
    DAQmx_Val_Volts,
    DAQmx_Val_Rising,
    DAQmx_Val_ContSamps,
    DAQmx_Val_GroupByScanNumber,
    DAQmx_Val_GroupByChannel,
    DAQmx_Val_FiniteSamps,
    DAQmx_Val_CountUp,
)

from PyDAQmx.DAQmxTypes import int32, uInt32, uInt64

SOURCE_DIR = Path(__file__).absolute().parent
CONFIG_FILE = Path(appdirs.user_config_dir(), 'fib-daqmx-scanner', 'config.toml')
DEFAULT_CONFIG_FILE = SOURCE_DIR / 'default_config.toml'


# TODO: use fill_value and initial_value = np.nan once this pyqtgraph bug fixed:
# https://github.com/pyqtgraph/pyqtgraph/issues/1057

class RollingData:
    """An append-only numpy array of fixed length, with old values discarded"""
    def __init__(self, initial_size, initial_value=0):
        self.data = np.full(initial_size, initial_value, dtype=float)

    def add_data(self, data):
        n = min(len(data), len(self.data))
        self.data[:-n] = self.data[n:]
        self.data[-n:] = data[-n:]

    def resize(self, newsize, fill_value=0):
        if newsize <= len(self.data):
            self.data = self.data[-newsize:]
        else:
            self.data = np.append(
                np.full(newsize - len(self.data), fill_value, dtype=float), self.data
            )


def queue_getall(queue):
    """Return a list of all items in a Queue"""
    items = []
    while True:
        try:
            items.append(queue.get_nowait())
        except Empty:
            return items


def get_sample_clock_term(task):
    """Return the string name of the sample clock terminal for a task"""
    BUFSIZE = 4096
    buff = ctypes.create_string_buffer(BUFSIZE)
    DAQmxGetSampClkTerm(task.taskHandle.value, buff, uInt32(BUFSIZE))
    return buff.value.decode('utf8')


def get_read_pos(task):
    """Return index of the next sample to be read in a finite acquisition task"""
    read_pos = uInt64()
    DAQmxGetReadCurrReadPos(task.taskHandle.value, read_pos)
    return read_pos.value


class State(IntEnum):
    STOPPED = 0
    RUNNING = 1
    SCANNING = 2

class ScanType(IntEnum):
    CEM_COUNTS = 0
    TARGET_CURRENT = 1

class App:
    MAX_READ_PTS = 10000
    MAX_READ_INTERVAL = 1 / 30

    def __init__(self):
        self.ui = loader = UiLoader()
        self.ui = loader.load(Path(SOURCE_DIR, 'main.ui'))
        self.image_view = pg.ImageView()
        self.image = np.zeros((1, 1), dtype=int)
        self.image_view.setSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding
        )
        self.ui.verticalLayoutImage.addWidget(self.image_view)
        self.ui.pushButtonStopScan.hide()

        self.fc_plotwidget = pg.GraphicsLayoutWidget()
        self.fc_plot = self.fc_plotwidget.addPlot()
        self.fc_plot.setDownsampling(mode='peak')
        self.fc_data = RollingData(self.ui.spinBoxPlotBuffer.value())
        self.fc_data_queue = Queue()
        self.fc_gain = 10 ** self.ui.spinBoxFaradayCupGain.value()
        self.fc_curve = self.fc_plot.plot(self.fc_data.data)
        self.fc_plot.setLabel('left', 'Faraday cup current', units='A')
        self.fc_plot.showGrid(True, True)
        self.ui.frameFaradayCup.layout().addWidget(self.fc_plotwidget)

        self.target_plotwidget = pg.GraphicsLayoutWidget()
        self.target_plot = self.target_plotwidget.addPlot()
        self.target_plot.setDownsampling(mode='peak')
        self.target_data = RollingData(self.ui.spinBoxPlotBuffer.value())
        self.target_data_queue = Queue()
        self.target_gain = 10 ** self.ui.spinBoxTargetGain.value()
        self.target_curve = self.target_plot.plot(self.target_data.data)
        self.target_plot.setLabel('left', 'target current', units='A')
        self.target_plot.showGrid(True, True)
        self.ui.frameTarget.layout().addWidget(self.target_plotwidget)

        self.cem_plotwidget = pg.GraphicsLayoutWidget()
        self.cem_plot = self.cem_plotwidget.addPlot()
        self.cem_plot.setDownsampling(mode='peak')
        self.cem_data = RollingData(self.ui.spinBoxPlotBuffer.value())
        self.cem_data_queue = Queue()
        self.cem_binwidth = 1 / self.ui.spinBoxSampleRate.value()
        self.cem_curve = self.cem_plot.plot(self.cem_data.data)
        self.cem_plot.setLabel('left', 'CEM counts', units='counts s⁻¹')
        self.cem_plot.showGrid(True, True)
        self.ui.frameCEM.layout().addWidget(self.cem_plotwidget)

        self.ui.toolButtonStart.clicked.connect(self.start)
        self.ui.toolButtonStop.clicked.connect(self.stop)
        self.ui.pushButtonStartScan.clicked.connect(self.start_scan)
        self.ui.pushButtonStopScan.clicked.connect(self.stop_scan)
        self.ui.spinBoxPlotBuffer.valueChanged.connect(self.on_plot_buffer_changed)
        self.ui.spinBoxSampleRate.valueChanged.connect(self.on_sample_rate_changed)
        self.ui.spinBoxFaradayCupGain.valueChanged.connect(self.on_fc_gain_changed)
        self.ui.spinBoxTargetGain.valueChanged.connect(self.on_target_gain_changed)

        self.render_plots_timer = QtCore.QTimer()
        self.render_plots_timer.timeout.connect(self.render_plots)

        self.state = State.STOPPED
        self.update_widget_state()

        self.AO_task = None
        self.AI_task = None
        self.CI_task = None
        self.acquisition_thread = None

        self.load_config()
        self.ui.show()

    def setup_AO_task(self, nx, ny, rate):
        """Write data to, but don't start the analog output task for a scan"""
        xmin = self.ui.doubleSpinBoxXmin.value()
        xmax = self.ui.doubleSpinBoxXmax.value()
        ymin = self.ui.doubleSpinBoxYmin.value()
        ymax = self.ui.doubleSpinBoxYmax.value()
        xcal = self.ui.doubleSpinBoxXcal.value()
        ycal = self.ui.doubleSpinBoxYcal.value()
        Vx = np.linspace(xmin / xcal, xmax / xcal, nx)
        Vy = np.linspace(ymin / ycal, ymax / ycal, ny)
        Vmin = min(Vx.min(), Vy.min())
        Vmax = max(Vx.max(), Vy.max())
        Vx, Vy = np.meshgrid(Vx, Vy)
        # We need one extra point than pixels, since we want to acquire 1 dwell time
        # after each sample is output. So we'll be throwing away the first acquired
        # point.
        Vx = np.append(Vx.flatten(), [0])
        Vy = np.append(Vy.flatten(), [0])
    
        self.AO_task = Task()

        AO_chans = [self.ui.lineEditScanX.text(), self.ui.lineEditScanY.text()]
        self.AO_task.CreateAOVoltageChan(
            ", ".join(AO_chans), "", Vmin, Vmax, DAQmx_Val_Volts, None
        )

        # Set up timing:
        self.AO_task.CfgSampClkTiming(
            "",
            rate,
            DAQmx_Val_Rising,
            DAQmx_Val_FiniteSamps,
            nx * ny + 1,
        )

        samples_written = int32()

        # Write data:
        self.AO_task.WriteAnalogF64(
            nx * ny + 1,
            False,
            10.0,
            DAQmx_Val_GroupByChannel,
            np.array([Vx, Vy]),
            samples_written,
            None,
        )

    def setup_AI_task(self, scanning, nx, ny, rate, npts_per_read):
        """Initialise but don't start the AI task. If scanning, nx, ny and rate will be
        used to configure the task in finite sample mode with the sample clock synced to
        the AO task, otherwise npts_per_read should be the number of points that will be
        read per read in continuous sample mode"""
        self.AI_task = Task()
        chans = [
            self.ui.lineEditFaradayCup.text(),
            self.ui.lineEditTarget.text(),
        ]
        self.AI_task.CreateAIVoltageChan(
            ', '.join(chans),
            "",
            DAQmx_Val_RSE,
            -10,
            10,
            DAQmx_Val_Volts,
            None,
        )
        if scanning:
            # Finite samples with sample clock synced to AO task:
            self.AI_task.CfgSampClkTiming(
                get_sample_clock_term(self.AO_task),
                rate,
                DAQmx_Val_Rising,
                DAQmx_Val_FiniteSamps,
                nx * ny + 1,
            )
        else:
            # Continuous samples being read at npts per read
            self.AI_task.CfgSampClkTiming(
                "", rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, npts_per_read
            )

    def setup_CI_task(self, scanning, nx, ny, rate, npts_per_read):
        self.CI_task = Task()
        self.CI_task.CreateCICountEdgesChan(
            self.ui.lineEditCEM.text(),
            "",
            DAQmx_Val_Rising,
            0,
            DAQmx_Val_CountUp,
        )

        if scanning:
            # Finite samples with sample clock synced to AO task:
            self.CI_task.CfgSampClkTiming(
                get_sample_clock_term(self.AO_task),
                rate,
                DAQmx_Val_Rising,
                DAQmx_Val_FiniteSamps,
                nx * ny + 1,
            )
        else:
            # Continuous samples with sample clock synced to AI task:
            self.CI_task.CfgSampClkTiming(
                get_sample_clock_term(self.AI_task),
                rate,
                DAQmx_Val_Rising,
                DAQmx_Val_ContSamps,
                npts_per_read,
            )

    def start_tasks(self, scanning=False, scan_type=None):
        nx = self.ui.spinBoxNx.value()
        ny = self.ui.spinBoxNy.value()

        assert self.AO_task is None
        assert self.AI_task is None
        assert self.CI_task is None
        if scanning:
            assert scan_type is not None

        if scanning:
            dwell_time = self.ui.spinBoxDwellTime.value() * 1e-6
            rate = 1 / dwell_time
            # Clear the image, if needed:
            if scan_type is ScanType.CEM_COUNTS:
                image_dtype = int
            elif scan_type is ScanType.TARGET_CURRENT:
                image_dtype = float
            else:
                raise ValueError(scan_type)
            # Blank the image if the dtype or shape has changed:
            if (self.image.dtype, self.image.shape) != (image_dtype, (nx, ny)):
                self.image = np.zeros((nx, ny), dtype=image_dtype)

            # Start the scanning AO task:
            self.setup_AO_task(nx, ny, rate)
        else:
            rate = float(self.ui.spinBoxSampleRate.value())

        # For acquisition, read data MAX_READ_PTS points at a time or once every
        # MAX_READ_INTERVAL seconds, whichever is faster, and a minimum of 2 pts:
        npts_per_read = max(
            2, min(self.MAX_READ_PTS, int(rate * self.MAX_READ_INTERVAL))
        )

        self.setup_AI_task(scanning, nx, ny, rate, npts_per_read)
        self.setup_CI_task(scanning, nx, ny, rate, npts_per_read)

        accumulate = self.ui.pushButtonAccumulate.isChecked()

        self.acquisition_thread = threading.Thread(
            target=self.acquisition_loop,
            args=(scanning, nx, ny, rate, npts_per_read, accumulate, scan_type),
            daemon=True,
        )

        # Ready
        self.AI_task.StartTask()
        self.CI_task.StartTask()

        # Set
        self.acquisition_thread.start()

        # Go!
        if self.AO_task is not None:
            self.AO_task.StartTask()

    def acquisition_loop(
        self, scanning, nx, ny, rate, npts_per_read, accumulate, scan_type
    ):
        """Acquire data in a loop, putting it to the data queues and inserting/adding to
        the image if we are scanning. If scanning, calls self.end_scan in the main
        thread once complete, otherwise runs indefinitely until one of the tasks is
        cleared"""

        AI_read_array = np.zeros((npts_per_read, 2), dtype=np.float64)
        CI_read_array = np.zeros(npts_per_read, dtype=np.uint32)

        total_samples= nx * ny + 1
        samples_read = int32()
        last_counter_value = None
        try:
            while True:
                read_pos = get_read_pos(self.AI_task)
                if scanning:
                    if read_pos == total_samples:
                        # End of scan. Repeat or stop, depending on settings. Queue this
                        # up in the main thread, passing in this thread to prevent a
                        # race.
                        inmain_later(self.end_scan, self.acquisition_thread)
                        return
                    samples_to_acquire = min(npts_per_read, total_samples - read_pos)
                else:
                    samples_to_acquire = npts_per_read
                
                self.CI_task.ReadCounterU32(
                        samples_to_acquire,
                        -1,
                        CI_read_array,
                        CI_read_array.size,
                        samples_read,
                        None,
                    )
                data = CI_read_array[: int(samples_read.value)]
                
                # FAKE DATA
                data = np.random.randint(
                    0, 1000, size=len(data), dtype=np.uint32
                ).cumsum() + (last_counter_value or 0)
                # END FAKE_DATA

                # Compute differences in counter values.
                if last_counter_value is None:
                    # First datapoint after starting the task is bogus, duplicate
                    # the second point instead. It will be ignored for the purposes
                    # of the image anyway, so this will only show up on the
                    # monitoring plot.
                    diffs = np.diff(data, prepend=0)
                    diffs[0] = diffs[1]
                else:
                    diffs = np.diff(data, prepend=last_counter_value)
                last_counter_value = data[-1]
                self.cem_data_queue.put(rate * diffs)
                if scanning and scan_type is ScanType.CEM_COUNTS:
                    image_data = diffs

                self.AI_task.ReadAnalogF64(
                    samples_to_acquire,
                    -1,
                    DAQmx_Val_GroupByScanNumber,
                    AI_read_array,
                    AI_read_array.size,
                    samples_read,
                    None,
                )
                data = AI_read_array[: int(samples_read.value), :]
                self.fc_data_queue.put(data[:, 0] / self.fc_gain)
                self.target_data_queue.put(data[:, 1] / self.target_gain)
                if scanning and scan_type is ScanType.TARGET_CURRENT:
                    image_data = data[:, 1] / self.target_gain

                if scanning:
                    # Ignore first point:
                    if read_pos == 0:
                        image_data = image_data[1:]
                    start_ix = max(read_pos - 1, 0)
                    end_ix = start_ix + len(image_data)
                    if accumulate:
                        self.image.ravel()[start_ix:end_ix] += image_data
                    else:
                        self.image.ravel()[start_ix:end_ix] = image_data

        except InvalidTaskError:
            # Task cleared by the main thread - we are being stopped.
            return

    def stop_tasks(self):
        self.AI_task.ClearTask()
        self.CI_task.ClearTask()
        if self.AO_task is not None:
            self.AO_task.ClearTask()
        self.acquisition_thread.join()
        self.CI_task = None
        self.AI_task = None
        self.AO_task = None

    def render_plots(self):
        target_data_chunks = queue_getall(self.target_data_queue)
        if target_data_chunks:
            self.target_data.add_data(np.concatenate(target_data_chunks))
            self.target_curve.setData(self.target_data.data)
            self.ui.labelTargetCurrent.setText(
                pg.functions.siFormat(
                    self.target_data.data[-1], precision=5, suffix='A'
                )
            )

        fc_data_chunks =  queue_getall(self.fc_data_queue)
        if fc_data_chunks:
            self.fc_data.add_data(np.concatenate(fc_data_chunks))
            self.fc_curve.setData(self.fc_data.data)
            self.ui.labelFaradayCupCurrent.setText(
                pg.functions.siFormat(self.fc_data.data[-1], precision=5, suffix='A')
            )

        cem_data_chunks = queue_getall(self.cem_data_queue)
        if cem_data_chunks:
            self.cem_data.add_data(np.concatenate(cem_data_chunks))
            self.cem_curve.setData(self.cem_data.data)
            self.ui.labelCEMCounts.setText(
                pg.functions.siFormat(
                    self.cem_data.data[-1], precision=5, suffix='counts s⁻¹'
                )
            )

        if self.state is State.SCANNING:
            autoscale = self.ui.toolButtonAutoScaleImage.isChecked()
            self.image_view.setImage(
                self.image.swapaxes(-1, -2),
                autoRange=autoscale,
                autoLevels=autoscale,
                autoHistogramRange=autoscale,
            )

    def start(self):
        # Transition from STOPPED to RUNNING
        assert self.state is State.STOPPED
        self.start_tasks(scanning=False)
        # TODO: create and start Beam blanker AO task. Possible...?
        self.render_plots_timer.start(int(1000 * self.MAX_READ_INTERVAL))
        self.state = State.RUNNING
        self.update_widget_state()

    def stop(self):
        # Transition from RUNNING or SCANNING to STOPPED
        assert self.state in [State.RUNNING, State.SCANNING]
        if self.state is State.SCANNING:
            self.stop_scan()
        self.stop_tasks()
        # Stop the rendering timer and call the render function one last time to render
        # remaining data:
        self.render_plots_timer.stop()
        self.render_plots()

        # TODO:
        # - Disable beam blanker (ensure button is disabled) and then stop beam blanker
        #   task.

        self.state = State.STOPPED
        self.update_widget_state()

    def start_scan(self):
        # Transition from RUNNING to SCANNING:
        assert self.state is State.RUNNING
        # Stop monitoring tasks and restart without the channel we're measuring as part
        # of the scan:
        self.stop_tasks()
        scan_type = ScanType(self.ui.comboBoxAcquire.currentIndex())
        self.start_tasks(scanning=True, scan_type=scan_type)
        self.state = State.SCANNING
        self.update_widget_state()

    def end_scan(self, acquisition_thread):
        """Run when a scan completes. Either stop or repeat scanning, depending on the
        repeat button. Ensures the """
        if self.acquisition_thread is not acquisition_thread:
            # Acquisition was stopped by the user before this function ran:
            return
        if self.ui.pushButtonRepeat.isChecked():
            self.stop_tasks()
            scan_type = ScanType(self.ui.comboBoxAcquire.currentIndex())
            self.start_tasks(scanning=True, scan_type=scan_type)
        else:
            self.stop_scan()

    def stop_scan(self):
        # Transition from SCANNING to RUNNING
        assert self.state is State.SCANNING
        self.stop_tasks()
        self.start_tasks(scanning=False)
        self.state = State.RUNNING
        self.update_widget_state()

    def update_widget_state(self):
        # Set widget enabled and and visibility state according to the current state.
        # This is a bit neater and less repetitive than putting it in the individual
        # transition methods:

        stopped = self.state is State.STOPPED
        running = self.state is State.RUNNING
        scanning = self.state is State.SCANNING

        # Start and stop button visibility:
        self.ui.toolButtonStart.setVisible(stopped)
        self.ui.toolButtonStop.setVisible(not stopped)

        # Connections and monitoring rate only editable when stopped:
        self.ui.lineEditFaradayCup.setEnabled(stopped)
        self.ui.lineEditTarget.setEnabled(stopped)
        self.ui.lineEditCEM.setEnabled(stopped)
        self.ui.lineEditScanX.setEnabled(stopped)
        self.ui.lineEditScanY.setEnabled(stopped)
        self.ui.lineEditBeamBlanker.setEnabled(stopped)
        self.ui.spinBoxSampleRate.setEnabled(stopped)

        # Beam blanker can only be enabled when not stopped:
        self.ui.pushButtonBeamBlankerEnable.setEnabled(not stopped)

        # Disable scanning when stopped, swap the start scan for the stop scan button
        # when scanning:
        self.ui.pushButtonStartScan.setEnabled(running)
        self.ui.pushButtonStartScan.setVisible(not scanning)
        self.ui.pushButtonStopScan.setVisible(scanning)

        # Progress bar only enabled when scanning:
        self.ui.progressBar.setEnabled(scanning)

        # Can't change these scan parameters when scanning:
        self.ui.doubleSpinBoxXmin.setEnabled(not scanning)
        self.ui.doubleSpinBoxXmax.setEnabled(not scanning)
        self.ui.doubleSpinBoxYmin.setEnabled(not scanning)
        self.ui.doubleSpinBoxYmax.setEnabled(not scanning)
        self.ui.spinBoxDwellTime.setEnabled(not scanning)
        self.ui.spinBoxNx.setEnabled(not scanning)
        self.ui.spinBoxNy.setEnabled(not scanning)
        self.ui.doubleSpinBoxXcal.setEnabled(not scanning)
        self.ui.doubleSpinBoxYcal.setEnabled(not scanning)
        self.ui.comboBoxAcquire.setEnabled(not scanning)

    def load_config(self):
        # Create with default if doesn't exist:
        if not CONFIG_FILE.exists():
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_FILE.write_text(DEFAULT_CONFIG_FILE.read_text())

        config = toml.load(CONFIG_FILE)

        self.ui.toolButtonAutoScaleImage.setChecked((config['misc']['autoscale_image']))

        self.ui.doubleSpinBoxXmin.setValue(config['scanning']['xmin'])
        self.ui.doubleSpinBoxXmax.setValue(config['scanning']['xmax'])
        self.ui.doubleSpinBoxYmin.setValue(config['scanning']['ymin'])
        self.ui.doubleSpinBoxYmax.setValue(config['scanning']['ymax'])
        self.ui.spinBoxDwellTime.setValue(config['scanning']['dwell_time'])
        self.ui.spinBoxNx.setValue(config['scanning']['nx'])
        self.ui.spinBoxNy.setValue(config['scanning']['ny'])
        self.ui.doubleSpinBoxXcal.setValue(config['scanning']['xcal'])
        self.ui.doubleSpinBoxYcal.setValue(config['scanning']['ycal'])
        self.ui.comboBoxAcquire.setCurrentText(config['scanning']['acquire'])

        self.ui.doubleSpinBoxBeamBlankerVoltage.setValue(
            config['beam_blanker']['voltage']
        )

        self.ui.lineEditCEM.setText(config['connections']['CEM'])
        self.ui.lineEditFaradayCup.setText(config['connections']['faraday_cup'])
        self.ui.lineEditTarget.setText(config['connections']['target'])
        self.ui.lineEditScanX.setText(config['connections']['scan_x'])
        self.ui.lineEditScanY.setText(config['connections']['scan_y'])
        self.ui.lineEditBeamBlanker.setText(config['connections']['beam_blanker'])

        self.ui.spinBoxSampleRate.setValue(config['monitoring']['sample_rate'])
        self.ui.spinBoxPlotBuffer.setValue(config['monitoring']['plot_buffer'])
        self.ui.spinBoxFaradayCupGain.setValue(config['monitoring']['faraday_cup_gain'])
        self.ui.spinBoxTargetGain.setValue(config['monitoring']['target_gain'])

    def save_config(self):
        config = {
            'misc': {'autoscale_image': self.ui.toolButtonAutoScaleImage.isChecked()},
            'scanning': {
                'xmin': self.ui.doubleSpinBoxXmin.value(),
                'xmax': self.ui.doubleSpinBoxXmax.value(),
                'ymin': self.ui.doubleSpinBoxYmin.value(),
                'ymax': self.ui.doubleSpinBoxYmax.value(),
                'dwell_time': self.ui.spinBoxDwellTime.value(),
                'nx': self.ui.spinBoxNx.value(),
                'ny': self.ui.spinBoxNy.value(),
                'xcal': self.ui.doubleSpinBoxXcal.value(),
                'ycal': self.ui.doubleSpinBoxYcal.value(),
                'acquire': self.ui.comboBoxAcquire.currentText(),
            },
            'beam_blanker': {
                'voltage': self.ui.doubleSpinBoxBeamBlankerVoltage.value()
            },
            'connections': {
                'CEM': self.ui.lineEditCEM.text(),
                'faraday_cup': self.ui.lineEditFaradayCup.text(),
                'target': self.ui.lineEditTarget.text(),
                'scan_x': self.ui.lineEditScanX.text(),
                'scan_y': self.ui.lineEditScanY.text(),
                'beam_blanker': self.ui.lineEditBeamBlanker.text(),
            },
            'monitoring': {
                'sample_rate': self.ui.spinBoxSampleRate.value(),
                'plot_buffer': self.ui.spinBoxPlotBuffer.value(),
                'faraday_cup_gain': self.ui.spinBoxFaradayCupGain.value(),
                'target_gain': self.ui.spinBoxTargetGain.value(),
            },
        }
        if not CONFIG_FILE.exists():
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(toml.dumps(config))

    def on_plot_buffer_changed(self, new_buffer_size):
        self.fc_data.resize(new_buffer_size)
        self.fc_curve.setData(self.fc_data.data)

        self.target_data.resize(new_buffer_size)
        self.target_curve.setData(self.target_data.data)

        self.cem_data.resize(new_buffer_size)
        self.cem_curve.setData(self.cem_data.data)

    def on_sample_rate_changed(self, value):
        self.cem_binwidth = 1 / value

    def on_fc_gain_changed(self, value):
        self.fc_gain = 10 ** value

    def on_target_gain_changed(self, value):
        self.target_gain = 10 ** value


def main():
    qapplication = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app = App()
    # Let the interpreter run every 500ms so it sees Ctrl-C interrupts:
    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)
    # Upon seeing a ctrl-c interrupt, quit the event loop
    signal.signal(signal.SIGINT, lambda *args: qapplication.exit())
    qapplication.exec()
    app.save_config()


if __name__ == '__main__':
    main()

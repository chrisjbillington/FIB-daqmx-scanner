import sys
import signal
from pathlib import Path
from enum import IntEnum
import threading

import appdirs

import toml
import numpy as np
from qtutils.qt import QtCore, QtGui, QtWidgets
from qtutils import inmain, inmain_decorator, UiLoader
import qtutils.icons
import pyqtgraph as pg

from PyDAQmx import Task
from PyDAQmx.DAQmxFunctions import DAQmxGetBufInputBufSize
from PyDAQmx.DAQmxConstants import (
    DAQmx_Val_RSE,
    DAQmx_Val_Volts,
    DAQmx_Val_Rising,
    DAQmx_Val_ContSamps,
    DAQmx_Val_Acquired_Into_Buffer,
    DAQmx_Val_GroupByScanNumber,
)

from PyDAQmx.DAQmxTypes import DAQmxEveryNSamplesEventCallbackPtr, int32, uInt32
# from PyDAQmx.DAQmxCallBack import 

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


class state(IntEnum):
    STOPPED = 0
    RUNNING = 1
    SCANNING = 2


class App:
    MAX_READ_PTS = 10000
    MAX_READ_INTERVAL = 1 / 30

    def __init__(self):
        self.ui = loader = UiLoader()
        self.ui = loader.load(Path(SOURCE_DIR, 'main.ui'))
        self.image = pg.ImageView()
        self.image.setSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding
        )
        self.ui.verticalLayoutImage.addWidget(self.image)
        self.ui.pushButtonStopScan.hide()

        self.fc_plotwidget = pg.GraphicsLayoutWidget()
        self.fc_plot = self.fc_plotwidget.addPlot()
        self.fc_plot.setDownsampling(mode='peak')
        self.fc_data = RollingData(self.ui.spinBoxPlotBuffer.value())
        self.fc_gain = 10 ** self.ui.spinBoxFaradayCupGain.value()
        self.fc_curve = self.fc_plot.plot(self.fc_data.data)
        self.fc_plot.setLabel('left', 'Faraday cup current', units='A')
        self.fc_plot.showGrid(True, True)
        self.ui.frameFaradayCup.layout().addWidget(self.fc_plotwidget)

        self.target_plotwidget = pg.GraphicsLayoutWidget()
        self.target_plot = self.target_plotwidget.addPlot()
        self.target_plot.setDownsampling(mode='peak')
        self.target_data = RollingData(self.ui.spinBoxPlotBuffer.value())
        self.target_gain = 10 ** self.ui.spinBoxTargetGain.value()
        self.target_curve = self.target_plot.plot(self.target_data.data)
        self.target_plot.setLabel('left', 'target current', units='A')
        self.target_plot.showGrid(True, True)
        self.ui.frameTarget.layout().addWidget(self.target_plotwidget)

        self.cem_plotwidget = pg.GraphicsLayoutWidget()
        self.cem_plot = self.cem_plotwidget.addPlot()
        self.cem_plot.setDownsampling(mode='peak')
        self.cem_data = RollingData(self.ui.spinBoxPlotBuffer.value())
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


        self.state = state.STOPPED
        self.update_widget_state()

        self.AI_task = None
        self.AI_read_array = None
        self.AI_tasklock = threading.RLock()

        self.AO_beam_blanker_task = None
        self.CI_task = None
        self.AO_scanning_task = None

        self.load_config()
        self.ui.show()

    def start_AI_task(self, target_current=True):
        """Start tha analog input task for monitoring faraday cup and target current. If
        target_current is False, do not include it in the task - this is the case when
        we are instead measuring the target current in a scan"""

        # Set up a task that acquires data with a callback every self.MAX_READ_PTS
        # points or self.MAX_READ_INTERVAL seconds, whichever is faster. NI DAQmx calls
        # callbacks in a separate thread, so this method returns, but data acquisition
        # continues in the thread.

        assert self.AI_task is None
        self.AI_task = Task()

        # Acquisition rate in samples per second:
        rate = self.ui.spinBoxSampleRate.value()

        chans = [self.ui.lineEditFaradayCup.text()]
        if target_current:
            chans.append(self.ui.lineEditTarget.text())


        for chan in chans:
            self.AI_task.CreateAIVoltageChan(
                chan,
                "",
                DAQmx_Val_RSE,
                -10,
                10,
                DAQmx_Val_Volts,
                None,
            )

        self.AI_task.CfgSampClkTiming(
            "", rate, DAQmx_Val_Rising, DAQmx_Val_ContSamps, int(rate)
        )

        bufsize = uInt32()
        DAQmxGetBufInputBufSize(self.AI_task.taskHandle, bufsize)
        bufsize = bufsize.value

        # This must not be garbage collected until the task is:
        self.AI_task.callback_ptr = DAQmxEveryNSamplesEventCallbackPtr(self.read_AI)

        # Get data MAX_READ_PTS points at a time or once every MAX_READ_INTERVAL
        # seconds, whichever is faster:
        num_samples = min(self.MAX_READ_PTS, int(rate * self.MAX_READ_INTERVAL))
        # Round to a multiple of two:
        num_samples = 2 * (num_samples // 2)
        self.AI_read_array = np.zeros((num_samples, len(chans)), dtype=np.float64)

        self.AI_task.RegisterEveryNSamplesEvent(
            DAQmx_Val_Acquired_Into_Buffer,
            num_samples,
            0,
            self.AI_task.callback_ptr,
            100,
        )
        self.AI_task.StartTask()

    def read_AI(self, task_handle, event_type, num_samples, callback_data=None):
        """Called as a callback by DAQmx while task is running. Also called by us to get
        remaining data just prior to stopping the task. Since the callback runs
        in a separate thread, we need to serialise access to instance variables"""
        samples_read = int32()
        with self.AI_tasklock:
            if self.AI_task is None or task_handle != self.AI_task.taskHandle.value:
                # Task stopped already.
                return 0
            self.AI_task.ReadAnalogF64(
                num_samples,
                -1,
                DAQmx_Val_GroupByScanNumber,
                self.AI_read_array,
                self.AI_read_array.size,
                samples_read,
                None,
            )
            # Select only the data read:
            data = self.AI_read_array[: int(samples_read.value), :]
            self.fc_data.add_data(data[:, 0] / self.fc_gain)
            if data.shape[1] > 1:
                self.target_data.add_data(data[:, 1] / self.target_gain)
            self.update_plots()
        return 0

    @inmain_decorator(wait_for_return=False)
    def update_plots(self):
        self.target_curve.setData(self.target_data.data)
        self.ui.labelTargetCurrent.setText(
            pg.functions.siFormat(self.target_data.data[-1], precision=5, suffix='A')
        )

        self.fc_curve.setData(self.fc_data.data)
        self.ui.labelFaradayCupCurrent.setText(
            pg.functions.siFormat(self.fc_data.data[-1], precision=4, suffix='A')
        )

        self.cem_curve.setData(self.cem_data.data)
        self.ui.labelCEMCounts.setText(
            pg.functions.siFormat(self.cem_data.data[-1], precision=5, suffix='counts s⁻¹')
        )

    def start(self):
        # Transition from STOPPED to RUNNING
        assert self.state is state.STOPPED
        
        self.start_AI_task()
        # create and start CI monitoring task and associated thread
        # Create and start AI monitoring task and associated thread
        # Create and start Beam blanker AO task


        self.state = state.RUNNING
        self.update_widget_state()
        

    def stop(self):
        # Transition from RUNNING or SCANNING to STOPPED
        assert self.state in [state.RUNNING, state.SCANNING]
        if self.state is state.SCANNING:
            self.stop_scan()

        with self.AI_tasklock:
            self.AI_task.StopTask()
            self.AI_task.ClearTask()
            self.AI_task = None
            self.AI_read_array = None
        # TODO:
        # - Stop CI, AI monitoring tasks and threads, set labels to read "–".
        # - Disable beam blanker (ensure button is disabled) and then stop beam blanker
        #   task.

        self.state = state.STOPPED
        self.update_widget_state()

    def start_scan(self):
        # Transition from RUNNING to SCANNING:
        assert self.state is state.RUNNING

        # TODO:
        # - if CEM counts:
        #   - Stop CI monitoring task
        #   - Setup AO and CI tasks and threads, start them
        # - if Target current:
        #   - Stop AI monitoring task and restart with only FC
        #   - Setup AO and AI tasks and threads, start them 

        self.state = state.SCANNING
        self.update_widget_state()

    def stop_scan(self):
        # Transition from SCANNING to RUNNING
        assert self.state is state.SCANNING

        # TODO:
        # - Stop tasks if they're not already stopped - figure this out when writing
        #   those threads.
        # - If CEM counts:
        #   - restart CEM monitoring thread
        # - If Target current:
        #   - Stop AI monitoring task and restart with both AI channels

        self.state = state.RUNNING
        self.update_widget_state()

    def update_widget_state(self):
        # Set widget enabled and and visibility state according to the current state.
        # This is a bit neater and less repetitive than putting it in the individual
        # transition methods:

        stopped = self.state is state.STOPPED
        running = self.state is state.RUNNING
        scanning = self.state is state.SCANNING

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
        self.ui.doubleSpinBoxXmin.setValue(config['scanning']['xmin'])
        self.ui.doubleSpinBoxXmax.setValue(config['scanning']['xmax'])
        self.ui.doubleSpinBoxYmin.setValue(config['scanning']['ymin'])
        self.ui.doubleSpinBoxYmax.setValue(config['scanning']['ymax'])
        self.ui.spinBoxDwellTime.setValue(config['scanning']['dwell_time'])
        self.ui.spinBoxNx.setValue(config['scanning']['nx'])
        self.ui.spinBoxNy.setValue(config['scanning']['ny'])
        self.ui.doubleSpinBoxXcal.setValue(config['scanning']['xcal'])
        self.ui.doubleSpinBoxYcal.setValue(config['scanning']['ycal'])
        self.ui.comboBoxAcquire.setCurrentIndex(config['scanning']['acquire'])

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
                'acquire': self.ui.comboBoxAcquire.currentIndex(),
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

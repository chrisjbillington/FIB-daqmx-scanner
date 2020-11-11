import sys
import signal
from pathlib import Path
from enum import IntEnum
import appdirs

import toml
import numpy as np
from qtutils.qt import QtCore, QtGui, QtWidgets
from qtutils import inmain, inmain_decorator, UiLoader
import qtutils.icons
import pyqtgraph as pg

SOURCE_DIR = Path(__file__).absolute().parent
CONFIG_FILE = Path(appdirs.user_config_dir(), 'fib-daqmx-scanner', 'config.toml')
DEFAULT_CONFIG_FILE = SOURCE_DIR / 'default_config.toml'


class state(IntEnum):
    STOPPED = 0
    RUNNING = 1
    SCANNING = 2


class App:
    def __init__(self):
        self.ui = loader = UiLoader()
        self.ui = loader.load(Path(SOURCE_DIR, 'main.ui'))
        self.image = pg.ImageView()
        self.image.setSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding
        )
        self.ui.verticalLayoutImage.addWidget(self.image)
        self.ui.pushButtonStopScan.hide()

        self.fc_plot = pg.GraphicsLayoutWidget(show=True)
        self.ui.frameFaradayCup.layout().addWidget(self.fc_plot)

        self.target_plot = pg.GraphicsLayoutWidget(show=True)
        self.ui.frameTarget.layout().addWidget(self.target_plot)

        self.cem_plot = pg.GraphicsLayoutWidget(show=True)
        self.ui.frameCEM.layout().addWidget(self.cem_plot)
        self.ui.toolButtonStart.clicked.connect(self.start)
        self.ui.toolButtonStop.clicked.connect(self.stop)
        self.ui.pushButtonStartScan.clicked.connect(self.start_scan)
        self.ui.pushButtonStopScan.clicked.connect(self.stop_scan)

        self.state = state.STOPPED
        self.update_widget_state()

        self.AO_beam_blanker_task = None
        self.CI_task = None
        self.AO_scanning_task = None
        self.AI_task = None

        self.load_config()
        self.ui.show()

    def start(self):
        # Transition from STOPPED to RUNNING
        assert self.state is state.STOPPED

        # TODO:
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

import logging

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams

import pyqtgraph as pg
import sys
from PyQt6 import QtWidgets, QtCore

class RealTime(QtWidgets.QMainWindow):
    def __init__(self, board_shim, *args, **kwargs):
        super(RealTime, self).__init__(*args, **kwargs)

        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_eeg_channels(self.board_id)  # [1, 2, 3, 4]
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 10
        self.num_points = self.window_size * self.sampling_rate

        self.win = pg.GraphicsLayoutWidget(title='BrainFlow Plot', size=(800, 600))
        self.setCentralWidget(self.win)
        self.win.setBackground('w')

        self._init_timeseries()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.update_speed_ms)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 3.0, 45.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 58.0, 62.0, 2,
                                        FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
            self.curves[count].setData(data[channel].tolist())

class Mindfulness(QtWidgets.QMainWindow):
    def __init__(self, board_shim, *args, **kwargs):
        super(Mindfulness, self).__init__(*args, **kwargs)

        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_eeg_channels(self.board_id)  # [1, 2, 3, 4]
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 25
        self.num_points = self.window_size * self.sampling_rate

        self.win = pg.GraphicsLayoutWidget(title='BrainFlow Plot', size=(800, 600))
        self.setCentralWidget(self.win)
        self.win.setBackground('w')

        self._init_timeseries()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.update_speed_ms)
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        for i in range(1):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('Mindfulness Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)
        
        mindfulness_params = BrainFlowModelParams(BrainFlowMetrics.MINDFULNESS.value,
                                              BrainFlowClassifiers.DEFAULT_CLASSIFIER.value)
        self.mindfulness = MLModel(mindfulness_params)
        self.mindfulness.prepare()
        self.output = []

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        bands = DataFilter.get_avg_band_powers(data, self.exg_channels, self.sampling_rate, True)
        feature_vector = bands[0] # this is avgs of band power across channels
        result = self.mindfulness.predict(feature_vector)
        self.output = self.output + result.tolist()

        print(self.window_size / (self.update_speed_ms / 1000))
        print(len(self.output))
        if len(self.output) > self.window_size / (self.update_speed_ms / 1000):
            self.output.pop(0)
        self.curves[0].setData(range(len(self.output)), self.output)

    def __del__(self):
        self.mindfulness.release()

def main():
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbmodem11"

    board_shim = BoardShim(BoardIds.GANGLION_BOARD, params)
    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000)
        app = QtWidgets.QApplication(sys.argv)
        w = Mindfulness(board_shim)
        w.show()
        sys.exit(app.exec())
    except BaseException:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()
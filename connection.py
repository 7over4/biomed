from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowOperations, DetrendOperations

import serial.tools.list_ports
import time
import matplotlib.pyplot as plt
import sched

def get_alpha_beta(board, board_id):
    data = board.get_board_data()

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    eeg_channel = eeg_channels[1]
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
    psd = DataFilter.get_psd_welch(data[eeg_channel], nfft, nfft // 2, sampling_rate, WindowOperations.HANNING.value)

    # calc band power
    alpha = DataFilter.get_band_power(psd, 7.0, 13.0)
    beta = DataFilter.get_band_power(psd, 14.0, 30.0)

    return alpha / beta

def main():
    # setup
    print("running")
    board_id = BoardIds.GANGLION_BOARD
    params = BrainFlowInputParams()
    
    ports = serial.tools.list_ports.comports()
    print("---PORTS---")
    for port, desc, hwid in sorted(ports):
        print(f"{port}: {desc} [{hwid}]")
    params.serial_port = "COM3"
    print("---ENDPORTS---")

    # create a board
    board = BoardShim(board_id, params)

    board.prepare_session()
    board.start_stream()

    # retrive data
    time.sleep(10)
    print(get_alpha_beta(board, board_id))
    if board.is_prepared():
        board.release_session()

if __name__ == "__main__":
    main()
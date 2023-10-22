from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import serial.tools.list_ports
import time

def main():
    boardID = BoardIds.GANGLION_BOARD
    params = BrainFlowInputParams()
    
    ports = serial.tools.list_ports.comports()
    print("---PORTS---")
    for port, desc, hwid in sorted(ports):
        print(f"{port}: {desc} [{hwid}]")
    params.serial_port = ""

    board = BoardShim(boardID, params)

    board.prepare_session()
    board.start_stream()

    time.sleep(10)
    data = board.get_board_data(30)

    board.stop_stream()
    board.release_session()

    print(data)

if __name__ == "__main__":
    main()
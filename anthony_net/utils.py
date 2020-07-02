import numpy as np
from minihex import HexGame, player


def convert_state(state):
    board = state.board
    board_size = board.shape[1]
    converted_state = np.zeros((6, board_size+4, board_size+4))

    converted_state[[0, 2], :2, :] = 1
    converted_state[[0, 3], -2:, :] = 1
    converted_state[[1, 4], :, :2] = 1
    converted_state[[1, 5], :, -2:] = 1

    converted_state[player.BLACK, 2:-2, 2:-2] = board == player.BLACK
    converted_state[player.WHITE, 2:-2, 2:-2] = board == player.WHITE

    region_black = np.pad(state.regions[player.BLACK], 1)
    region_white = np.pad(state.regions[player.WHITE], 1)

    positions = np.where(region_black == region_black[1, 1])
    converted_state[2][positions] = 1

    positions = np.where(region_black == region_black[-2, -2])
    converted_state[3][positions] = 1

    positions = np.where(region_white == region_white[1, 1])
    converted_state[4][positions] = 1

    positions = np.where(region_white == region_white[-2, -2])
    converted_state[5][positions] = 1

    return np.moveaxis(converted_state, 0, -1)

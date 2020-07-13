import numpy as np
from minihex import HexGame, player
import random


def convert_state(sim):
    board = sim.board
    board_size = board.shape[1]
    converted_state = np.zeros((6, board_size+4, board_size+4))

    converted_state[[0, 2], :2, :] = 1
    converted_state[[0, 3], -2:, :] = 1
    converted_state[[1, 4], :, :2] = 1
    converted_state[[1, 5], :, -2:] = 1

    converted_state[player.BLACK, 2:-2, 2:-2] = board == player.BLACK
    converted_state[player.WHITE, 2:-2, 2:-2] = board == player.WHITE

    region_black = np.ones((board_size + 4, board_size + 4))
    region_black[1:-1, 1:-1] = sim.regions[player.BLACK]
    region_white = np.ones((board_size + 4, board_size + 4))
    region_white[1:-1, 1:-1] = sim.regions[player.WHITE]

    positions = np.where(region_black == region_black[1, 1])
    converted_state[2][positions] = 1

    positions = np.where(region_black == region_black[-2, -2])
    converted_state[3][positions] = 1

    positions = np.where(region_white == region_white[1, 1])
    converted_state[4][positions] = 1

    positions = np.where(region_white == region_white[-2, -2])
    converted_state[5][positions] = 1

    return np.moveaxis(converted_state, 0, -1)


def generate_sample(board_size=9):
    num_white_stones = np.random.randint(board_size ** 2 // 2)
    if random.random() > 0.5:
        num_black_stones = num_white_stones + 1
        active_player = player.WHITE
    else:
        num_black_stones = num_white_stones
        active_player = player.BLACK
    positions = np.random.rand(board_size, board_size)
    board_shape = (board_size, board_size)
    ny, nx = np.unravel_index(np.argsort(positions.flatten()), board_shape)
    white_y = ny[:num_white_stones]
    white_x = nx[:num_white_stones]
    black_y = ny[num_white_stones:num_white_stones+num_black_stones]
    black_x = nx[num_white_stones:num_white_stones+num_black_stones]
    board = player.EMPTY * np.ones((board_size, board_size))
    board[white_y, white_x] = player.WHITE
    board[black_y, black_x] = player.BLACK

    sim = HexGame(active_player, board, active_player)
    while sim.done:
        sim, active_player = generate_sample(board_size)

    return sim, active_player

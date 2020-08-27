import numpy as np
from minihex import HexGame, player
import random
from enum import IntEnum


class Direction(IntEnum):
    NORTH = 2
    EAST = 5
    SOUTH = 3
    WEST = 4


def convert_state_batch(sim_batch):
    board_size = sim_batch[0].board_size
    batch_size = len(sim_batch)
    board_batch = np.stack([sim.board for sim in sim_batch])
    black_regions = np.stack([sim.regions[player.BLACK] for sim in sim_batch])
    white_regions = np.stack([sim.regions[player.WHITE] for sim in sim_batch])
    regions = np.stack([sim.regions for sim in sim_batch])

    # order: batch, player, [north/west, south/east]
    border_regions = regions[:, :, [0, -1], [0, -1]]
    border_regions = border_regions.reshape((batch_size, 2, 1, 1, 2))
    border_stones = regions[..., np.newaxis] == border_regions

    converted_state = np.ones((batch_size, board_size+4, board_size+4, 6))
    converted_state[:, 2:-2, :, player.BLACK] = 0
    converted_state[:, :, 2:-2, player.WHITE] = 0
    converted_state[:, 2:, :, Direction.NORTH] = 0
    converted_state[:, :-2, :, Direction.SOUTH] = 0
    converted_state[:, :, 2:, Direction.WEST] = 0
    converted_state[:, :, :-2, Direction.EAST] = 0

    converted_state[:, 2:-2, 2:-2, player.BLACK] = board_batch == player.BLACK
    converted_state[:, 2:-2, 2:-2, player.WHITE] = board_batch == player.WHITE
    converted_state[:, 1:-1, 1:-1,
                    Direction.NORTH] = border_stones[:, player.BLACK, :, :, 0]
    converted_state[:, 1:-1, 1:-1,
                    Direction.SOUTH] = border_stones[:, player.BLACK, :, :, 1]
    converted_state[:, 1:-1, 1:-1,
                    Direction.WEST] = border_stones[:, player.WHITE, :, :, 0]
    converted_state[:, 1:-1, 1:-1,
                    Direction.EAST] = border_stones[:, player.WHITE, :, :, 1]

    return converted_state


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


def generate_board(board_size=9):
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
        sim, active_player = generate_board(board_size)

    return sim, active_player

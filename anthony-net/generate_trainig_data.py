import numpy as np
from mcts.MCTSAgent import MCTSAgent
import gym
from minihex import player
from minihex.HexGame import HexGame
import minihex
import tqdm
from enum import IntEnum


class Side(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


def flood_fill(board, position, side, connected_stones=None):
    connections = np.array([[-1, -1,  0,  0,  1,  1],
                            [0,   1, -1,  1, -1,  0]])

    if connected_stones is None:
        connected_stones = np.zeros_like(board)

    positions_to_test = [position]
    while len(positions_to_test) > 0:
        current_position = positions_to_test.pop()

        if board[current_position] == 0:
            continue

        if connected_stones[current_position] == 1:
            continue

        neighbours = list()
        check_neighbours = False
        for direction in connections.T:
            neighbour_position = tuple(current_position + direction)

            if np.all(neighbour_position) == board.shape[1]:
                if side == Side.SOUTH or side == Side.EAST:
                    check_neighbours = True
                    connected_stones[current_position] = 1
                continue
            elif np.all(neighbour_position) < 0:
                if side == side.NORTH or side == Side.WEST:
                    check_neighbours = True
                    connected_stones[current_position] = 1
                continue
            elif neighbour_position[0] < 0:
                if side == Side.NORTH:
                    check_neighbours = True
                    connected_stones[current_position] = 1
                continue
            elif neighbour_position[1] >= board.shape[1]:
                if side == Side.EAST:
                    check_neighbours = True
                    connected_stones[current_position] = 1
                continue
            elif neighbour_position[0] >= board.shape[1]:
                if side == Side.SOUTH:
                    check_neighbours = True
                    connected_stones[current_position] = 1
                continue
            elif neighbour_position[1] < 0:
                if side == Side.WEST:
                    check_neighbours = True
                    connected_stones[current_position] = 1
                continue

            if board[neighbour_position] == 1:
                neighbours.append(neighbour_position)

            if connected_stones[neighbour_position] == 1:
                connected_stones[current_position] = 1
                check_neighbours = True

        if check_neighbours:
            positions_to_test += neighbours

    return connected_stones


def convert_state(state):
    game_state, active_player = state
    board_size = game_state.shape[1]

    black_stones = game_state[player.BLACK, ...]
    white_stones = game_state[player.WHITE, ...]

    black_north = np.zeros_like(black_stones)
    black_south = np.zeros_like(black_stones)
    white_west = np.zeros_like(white_stones)
    white_east = np.zeros_like(white_stones)

    for idx in range(board_size):
        black_north = flood_fill(black_stones, (0, idx), Side.NORTH, black_north)
        black_south = flood_fill(black_stones, (board_size - 1, idx), Side.SOUTH, black_south)
        white_west = flood_fill(white_stones, (idx, 0), Side.WEST, white_west)
        white_east = flood_fill(white_stones, (idx, board_size - 1), Side.EAST, white_east)

    converted_state = np.stack([
        black_stones,
        white_stones,
        black_north,
        black_south,
        white_west,
        white_east
    ], axis=0)

    return converted_state


def play_match(agent1, agent2):
    env = gym.make("hex-v0",
                   opponent_policy=agent2.policy,
                   player_color=np.random.randint(2))
    state = env.reset()

    history = list()
    done = False
    while not done:
        action = agent1.policy(*state)
        history.append((state, action))

        state, reward, done, _ = env.step(action)

    random_element = np.random.randint(len(history))
    state, action = history[random_element]
    return convert_state(state), action


class Agent(object):
    def __init__(self, depth=1000):
        self.depth = depth

    def policy(self, state, active_player):
        env = HexGame(active_player, state, active_player)
        agent = MCTSAgent(env)
        return agent.plan(self.depth)

    def __str__(self):
        return f"MCTS({self.depth})"


def generate_dataset(num_examples, prefix=None):
    if prefix is None:
        prefix = ""
    else:
        prefix += "_"

    agent = Agent(depth=1000)

    dataset = list()
    labels = list()
    for sample in tqdm.tqdm(range(num_examples)):
        position, action = play_match(agent, agent)
        board = position[1, ...]
        pos = np.unravel_index(action, board.shape)
        assert np.all(position[:2, pos[0], pos[1]]) == 0
        dataset.append(position)
        labels.append(action)

    with open(f"{prefix}data.npy", "wb") as out_file:
        np.save(out_file, np.stack(dataset, axis=0))
    with open(f"{prefix}labels.npy", "wb") as out_file:
        np.save(out_file, np.stack(labels, axis=0))


generate_dataset(100, prefix="training")
generate_dataset(10, prefix="validation")
generate_dataset(10, prefix="test")

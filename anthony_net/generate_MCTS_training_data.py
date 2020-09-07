import numpy as np
from mcts.MCTSAgent import MCTSAgent
from minihex import player, HexGame
import minihex
import tqdm
from multiprocessing import Pool
from .utils import convert_state
from .utils import generate_board
import random


def generate_sample(args):
    board_size, config = args
    sim, active_player = generate_board(board_size)

    depth = int(config["MCTSAgent"]["search_depth"])
    agent = MCTSAgent(sim, depth=depth)
    action = agent.act(sim.board, active_player)

    if active_player == player.BLACK:
        return sim.board, active_player, (action, -1)
    else:
        return sim.board, active_player, (-1, action)


def generate_dataset(config):
    dataset = list()
    players = list()
    labels = list()

    num_samples = int(config["GLOBAL"]["dataset_size"])
    num_threads = int(config["GLOBAL"]["num_threads"])
    board_size = int(config["GLOBAL"]["board_size"])
    chunksize = int(config["GLOBAL"]["chunksize"])

    with Pool(num_threads) as workers:
        return_val = list(tqdm.tqdm(workers.imap(
            generate_sample,
            [(board_size, config)] * num_samples,
            chunksize=1),
            total=num_samples,
            desc="Generating Samples"))
    for example, active_player, label in return_val:
        dataset.append(example)
        players.append(active_player)
        labels.append(label)

    data_file = config["nnEval"]["training_file"]
    data_file = data_file.format(board_size=board_size)
    np.savez(data_file, np.stack(dataset, axis=0), np.stack(players, axis=0))

    label_file = config["nnEval"]["label_file"]
    label_file = label_file.format(board_size=board_size)
    np.savez(label_file, np.stack(labels, axis=0))


if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read('ExIt.ini')

    generate_dataset(config)

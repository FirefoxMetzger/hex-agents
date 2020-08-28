import numpy as np
from mcts.MCTSAgent import MCTSAgent
from minihex import player, HexGame
import minihex
import tqdm
from multiprocessing import Pool
from utils import convert_state
from utils import generate_board
import random


def generate_sample(board_size=5):
    sim, active_player = generate_board(board_size)

    agent = MCTSAgent(sim, depth=1000)
    action = agent.act(sim.board, active_player)

    if active_player == player.BLACK:
        return convert_state(sim), (action, -1)
    else:
        return convert_state(sim), (-1, action)


def generate_dataset(config):
    dataset = list()
    labels = list()

    num_samples = int(config["nnEval"]["dataset_size"])
    num_threads = int(config["GLOBAL"]["num_threads"])
    board_size = int(config["GLOBAL"]["board_size"])

    with Pool(num_threads) as workers:
        return_val = list(tqdm.tqdm(workers.imap(
            generate_sample,
            [board_size for idx in range(num_samples)],
            chunksize=1),
            total=num_samples))
    for example, label in return_val:
        dataset.append(example)
        labels.append(label)

    data_file = config["nnEval"]["training_file"]
    np.save(data_file, np.stack(dataset, axis=0))

    label_file = config["nnEval"]["label_file"]
    np.save(label_file, np.stack(labels, axis=0))


if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read('ExIt.ini')

    generate_dataset(config)

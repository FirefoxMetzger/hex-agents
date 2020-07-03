import numpy as np
from mcts.MCTSAgent import MCTSAgent
import gym
from minihex import player, HexGame
import minihex
import tqdm
from anthony_net.utils import convert_state
from anthony_net.NNAgent import NNAgent
import random
from utils import generate_sample


def generate_dataset(num_examples, prefix=None, batch_size=32):
    if prefix is None:
        prefix = ""
    else:
        prefix += "_"

    num_batches = num_examples // batch_size + 1
    agent = NNAgent('best_model.h5')

    dataset = list()
    active_player = list()
    labels = list()
    for batch in tqdm.tqdm(range(num_batches)):
        for sample_idx in range(batch_size):
            sample, player = generate_sample()
            dataset.append(sample)
            active_player.append(player)
    dataset = np.stack(dataset, axis=0)
    active_player = np.stack(active_player, axis=0)

    labels = agent.predict(dataset, active_player)

    with open(f"{prefix}data.npy", "wb") as out_file:
        np.save(out_file, dataset)
    with open(f"{prefix}labels.npy", "wb") as out_file:
        np.save(out_file, labels)


if __name__ == "__main__":
    generate_dataset(100000, prefix="training")
    # generate_dataset(3000, prefix="validation")
    # generate_dataset(3000, prefix="test")

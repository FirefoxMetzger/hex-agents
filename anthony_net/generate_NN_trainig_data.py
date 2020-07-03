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


def generate_dataset(num_examples, batch_size=32):
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

    return dataset, labels


if __name__ == "__main__":
    dataset, labels = generate_dataset(100000)
    with open(f"training_data.npy", "wb") as out_file:
        np.save(out_file, dataset)
    with open(f"training_labels.npy", "wb") as out_file:
        np.save(out_file, labels)

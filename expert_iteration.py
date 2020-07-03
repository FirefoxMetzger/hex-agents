import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from anthony_net.network import gen_model, selective_loss
import tqdm
from minihex import player
import numpy as np
from anthony_net.utils import convert_state, generate_sample
from mcts.MCTSAgent import MCTSAgent
from Agent import RandomAgent
from nmcts.NMCTSAgent import NMCTSAgent
from anthony_net.NNAgent import NNAgent
import tensorflow as tf


def build_expert(apprentice_agent, board_size, depth):
    return NMCTSAgent(board_size=board_size,
                      depth=depth,
                      agent=apprentice_agent)


def build_apprentice(training_dataset, validation_dataset, board_size, iteration):
    training_data, training_labels = training_dataset
    validation_data, validation_labels = validation_dataset

    training_labels = tf.one_hot(training_labels, board_size ** 2)
    training_labels = (training_labels[:, 0, :], training_labels[:, 1, :])
    validation_labels = tf.one_hot(validation_labels, board_size ** 2)
    validation_labels = (
        validation_labels[:, 0, :], validation_labels[:, 1, :])

    network = gen_model(board_size)
    network.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=[selective_loss, selective_loss],
        metrics=["CategoricalAccuracy"]
    )
    network.fit(training_data, training_labels,
                epochs=100,
                batch_size=256,
                validation_data=(validation_data, validation_labels),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=3,
                    ),
                    tf.keras.callbacks.ModelCheckpoint(
                        f'best_model_iteration_{iteration}.h5',
                        monitor='val_loss',
                        save_only_best=True)

                ])
    return NNAgent(network)


def generate_samples(num_samples, board_size):
    # the paper generates a rollout via selfplay and then samples a single
    # position from the entire trajectory to avoid correlations in the data
    # Here we improve this, by directly sampling a game position randomly
    # (0 correlation with other states in the dataset)
    board_positions = list()
    active_players = list()
    for idx in range(num_samples):
        board, active_player = generate_sample(board_size)
        board_positions.append(convert_state(board))
        active_players.append(active_player)
    return np.stack(board_positions, axis=0), np.stack(active_players, axis=0)


def compute_labels(board_positions, active_players, expert):
    labels = list()
    for board, active_player in zip(board_positions, active_players):
        action = expert.act(board, active_player, None)
        if active_player == player.WHITE:
            labels.append((-1, action))
        else:
            labels.append((action, -1))

    return np.stack(labels, axis=0)


board_size = 5
search_depth = 10
dataset_size = 10
iterations = 2

apprentice = RandomAgent(board_size=board_size)
expert = MCTSAgent(board_size=board_size, depth=search_depth)
for idx in tqdm.tqdm(range(iterations)):
    # apprenticeship learning
    # -----
    training_data, active_players = generate_samples(
        dataset_size, board_size)
    training_labels = compute_labels(training_data, active_players, expert)

    validation_data, active_players = generate_samples(
        dataset_size, board_size)
    validation_labels = compute_labels(validation_data, active_players, expert)

    apprentice = build_apprentice(
        (training_data, training_labels),
        (validation_data, validation_labels),
        board_size, idx)

    # expert iteration
    # -----
    expert = build_expert(apprentice, board_size, search_depth)

import silence_tensorflow
from collections import deque
import tensorflow as tf
from anthony_net.NNAgent import NNAgent
from nmcts.NMCTSAgent import NMCTSAgent
from Agent import RandomAgent
from mcts.MCTSAgent import MCTSAgent
from anthony_net.utils import convert_state, generate_sample
import numpy as np
from minihex import player, HexGame
import tqdm
from tqdm.keras import TqdmCallback
from anthony_net.network import gen_model, selective_loss
import itertools
from nmcts.NeuralSearchNode import NeuralSearchNode
from multiprocessing import Pool, cpu_count
from utils import simulate, nmcts_builder, dataset_converter


class KerasBar(tqdm.tqdm):
    pass


def task_iter(queue):
    while queue:
        yield queue.popleft()


def build_expert(apprentice_agent, board_size, depth):
    return NMCTSAgent(board_size=board_size,
                      depth=depth,
                      agent=apprentice_agent)


def build_apprentice(train_dataset, val_dataset, board_size, iteration):
    training_data, training_labels = train_dataset
    validation_data, validation_labels = val_dataset

    training_data = np.stack([example for example in workers.imap(dataset_converter, training_data, chunksize=10)])
    validation_data = np.stack([example for example in workers.imap(dataset_converter, validation_data, chunksize=10)])
    
    training_labels = tf.one_hot(training_labels, board_size ** 2)
    training_labels = (training_labels[:, 0, :], training_labels[:, 1, :])
    validation_labels = tf.one_hot(validation_labels, board_size ** 2)
    validation_labels = (
        validation_labels[:, 0, :], validation_labels[:, 1, :])

    tf.keras.backend.clear_session()

    network = gen_model(board_size)
    network.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=[selective_loss, selective_loss],
        metrics=["CategoricalAccuracy"]
    )
    network.fit(
        training_data, training_labels,
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
                save_only_best=True),
            TqdmCallback(data_size=len(training_data), batch_size=256)
        ],
        verbose=0)
    return NNAgent(network)


def generate_samples(num_samples, board_size, workers):
    # the paper generates a rollout via selfplay and then samples a single
    # position from the entire trajectory to avoid correlations in the data
    # Here we improve this, by directly sampling a game position randomly
    # (0 correlation with other states in the dataset)
    board_positions = list()
    active_players = list()

    chunksize = 10
    result = tqdm.tqdm(workers.imap(generate_sample, [board_size] * num_samples, chunksize=chunksize),
                       desc="Generating Samples", position=1, leave=False, total=num_samples)
    for sim, active_player in result:
        board_positions.append(sim.board)
        active_players.append(active_player)
    return np.stack(board_positions, axis=0), np.stack(active_players, axis=0)


def compute_labels(board_positions, active_players, expert, workers):
    # setup
    # ---
    labels = list()
    batch_size = active_players.shape[0]
    chunksize = 20
    nn_agent = expert.agent
    depth = expert.depth
    pbar = tqdm.tqdm(desc="Generating Labels",
                     total=batch_size, position=1, leave=False)

    sim_args = zip(active_players, board_positions, active_players)
    initial_sims = [sim for sim in  tqdm.tqdm(workers.starmap(HexGame, sim_args, chunksize=chunksize), desc="Building Environments", position=2, leave=False, total=batch_size)]
    converted_boards = np.stack([example for example in workers.imap(dataset_converter, board_positions, chunksize=chunksize)])
    policies = nn_agent.get_scores(converted_boards, active_players)

    nmcts_args = zip([depth] * batch_size, initial_sims, policies)
    agents = [agent for agent in tqdm.tqdm(
        workers.imap(nmcts_builder, nmcts_args, chunksize=chunksize), desc="Initializing Agents", position=2, leave=False, total=batch_size)]

    tasks = deque()
    for idx, agent in enumerate(agents):
        gen = agent.deferred_plan()
        tasks.append((idx, "init", gen, None))

    while tasks:
        node_creation_batch = list()
        simulation_batch = list()
        for task in task_iter(tasks):
            job = task[1]
            if job == "expand":
                node_creation_batch.append(task)
            elif job == "simulate":
                simulation_batch.append(task)
            elif job == "init":
                idx, job, gen, args = task
                job, args = gen.send(None)
                tasks.append((idx, job, gen, args))
            elif job == "done":
                idx, _, _, _ = task
                action = agents[idx].act_greedy(None, None, None)
                if initial_sims[idx].active_player == player.WHITE:
                    labels.append((-1, action))
                else:
                    labels.append((action, -1))
                pbar.update(1)

        # handle expansions
        if node_creation_batch:
            board_batch = list()
            player_batch = list()
            for task in node_creation_batch:
                _, _, _, sim = task
                board_batch.append(convert_state(sim))
                player_batch.append(sim.active_player)
            board_batch = np.stack(board_batch)
            player_batch = np.stack(player_batch)

            policies = nn_agent.get_scores(board_batch, player_batch)

            for task, policy in zip(node_creation_batch, policies):
                idx, job, gen, sim = task
                node = NeuralSearchNode(sim, agent=nn_agent,
                                        network_policy=policy)
                try:
                    job, args = gen.send(node)
                    tasks.append((idx, job, gen, args))
                except StopIteration:
                    tasks.append((idx, "done", gen, None))

        # handle simulation
        if simulation_batch:
            sims = [task[3] for task in simulation_batch]
            board_sizes = [sim.board_size for sim in sims]
            winners = [winner for winner in workers.starmap(
                simulate, zip(sims, board_sizes))]
            for task, winner in zip(simulation_batch, winners):
                idx, job, gen, _ = task
                try:
                    job, args = gen.send(winner)
                    tasks.append((idx, job, gen, args))
                except StopIteration:
                    tasks.append((idx, "done", gen, None))

    return np.stack(labels, axis=0)


if __name__ == "__main__":
    board_size = 9
    search_depth = 5
    dataset_size = 100000
    validation_size = 1000
    iterations = 3

    expert = NMCTSAgent(board_size=board_size,
                        depth=search_depth, model_file="best_model.h5")
    with Pool(cpu_count() - 2) as workers:
        for idx in tqdm.tqdm(range(iterations), desc="Training Experts", position=0):
            # apprenticeship learning
            # -----
            training_data, active_players = generate_samples(
                dataset_size, board_size, workers)
            training_labels = compute_labels(
                training_data, active_players, expert, workers)

            validation_data, active_players = generate_samples(
                validation_size, board_size, workers)
            validation_labels = compute_labels(
                validation_data, active_players, expert, workers)

            apprentice = build_apprentice(
                (training_data, training_labels),
                (validation_data, validation_labels),
                board_size, idx)

            # expert iteration
            # -----
            expert = build_expert(apprentice, board_size, search_depth)

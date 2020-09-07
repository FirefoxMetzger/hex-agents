import silence_tensorflow
import tensorflow as tf
from anthony_net.NNAgent import NNAgent
from nmcts.NMCTSAgent import NMCTSAgent
from mcts.MCTSAgent import MCTSAgent
from anthony_net.utils import generate_board, convert_state_batch
import numpy as np
from minihex import player, HexGame
import tqdm
from multiprocessing import Pool
from anthony_net.train_network import train_network
import configparser
import os
from scheduler.scheduler import Scheduler
from scheduler.handlers import (
    HandleInit,
    HandleDone,
    HandleNNPolicy,
    HandleRollout
)
from scheduler.tasks import InitExit


def build_expert(apprentice_agent, config):
    board_size = int(config["GLOBAL"]["board_size"])
    depth = int(config["NMCTSAgent"]["search_depth"])
    return NMCTSAgent(board_size=board_size,
                      depth=depth,
                      agent=apprentice_agent)


def build_apprentice(samples, labels, config, workers):
    board_size = int(config["GLOBAL"]["board_size"])
    chunksize = int(config["GLOBAL"]["chunksize"])

    boards, players = samples
    sim_args = zip(players, boards, players)
    if workers:
        data = [sim for sim in workers.starmap(
            HexGame,
            sim_args,
            chunksize=chunksize)]
    else:
        data = [sim for sim in map(
            HexGame,
            sim_args)]
    data = np.stack(convert_state_batch(data))

    labels = tf.one_hot(labels, board_size ** 2)
    labels = (labels[:, 0, :], labels[:, 1, :])

    tf.keras.backend.clear_session()
    network = train_network(data, labels, config)
    return NNAgent(network)


def generate_samples(config, workers):
    # the paper generates a rollout via selfplay and then samples a single
    # position from the entire trajectory to avoid correlations in the data
    # Here we "improve" this, by directly sampling a game position randomly
    # (0 correlation with other states in the dataset)

    num_samples = int(config["GLOBAL"]["dataset_size"])
    board_size = int(config["GLOBAL"]["board_size"])

    board_positions = list()
    active_players = list()

    chunksize = int(config["GLOBAL"]["chunksize"])
    result = tqdm.tqdm(workers.imap(
        generate_board,
        [board_size] * num_samples,
        chunksize=chunksize),
        desc="Generating Samples",
        position=1,
        leave=False,
        total=num_samples)
    for sim, active_player in result:
        board_positions.append(sim.board)
        active_players.append(active_player)

    return np.stack(board_positions, axis=0), np.stack(active_players, axis=0)


def compute_labels(samples, expert, config, workers):
    dataset_size = int(config["GLOBAL"]["dataset_size"])
    handlers = [
        HandleNNPolicy(nn_agent=expert.agent),
        HandleRollout(
            workers=workers,
            config=config
        ),
        HandleInit(
            nn_agent=expert.agent,
            config=config,
            workers=workers
        ),
        HandleDone(dataset_size)]
    sched = Scheduler(handlers)

    queue = [InitExit(sample, idx) for idx, sample in enumerate(zip(*samples))]
    max_active = int(config["GLOBAL"]["active_simulations"])
    active_tasks = queue[:max_active]
    queue = queue[max_active:]

    pbar = tqdm.tqdm(
        desc="Generating Labels",
        total=dataset_size,
        position=1,
        leave=False)
    while queue or active_tasks:
        if len(active_tasks) < max_active:
            num_new = max_active - len(active_tasks)
            num_new = min(num_new, len(queue))
            new_tasks = queue[:num_new]
            active_tasks += new_tasks
            queue = queue[num_new:]

        old_count = len(active_tasks)
        active_tasks = sched.process(active_tasks)
        completed = old_count - len(active_tasks)
        pbar.update(completed)

    labels = handlers[-1].labels
    return np.stack(labels, axis=0)


def exIt(workers, config):
    board_size = int(config["GLOBAL"]["board_size"])
    iterations = int(config["ExpertIteration"]["iterations"])

    log_dir = config["GLOBAL"]["log_dir"]
    out_dir = config["ExpertIteration"]["dir"]
    base_dir = "/".join([log_dir, out_dir])
    base_dir = base_dir.format(board_size=board_size)
    os.makedirs(base_dir, exist_ok=True)

    step_location = config["ExpertIteration"]["step_location"]

    depth = int(config["NMCTSAgent"]["search_depth"])
    expert = NMCTSAgent(
        board_size=board_size,
        depth=depth,
        model_file=config["Training"]["model_file"])

    pbar = tqdm.tqdm(range(iterations),
                     desc="Training Experts",
                     position=0)
    for idx in pbar:
        directory = "/".join([base_dir, step_location])
        directory = directory.format(idx=idx)
        os.makedirs(directory, exist_ok=True)
        config["Training"]["model_file"] = "/".join([directory, "model.h5"])
        config["Training"]["history_file"] = (
            "/".join([directory, "history.pickle"]))

        samples = generate_samples(config, workers)
        labels = compute_labels(samples, expert, config, workers)

        data_file = "/".join([directory, "data.npz"])
        np.savez(data_file, *samples)

        label_file = "/".join([directory, "labels.npz"])
        np.savez(label_file, labels)

        apprentice = build_apprentice(samples, labels, config, workers)

        expert = build_expert(apprentice, config)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('ExIt.ini')

    num_threads = int(config["GLOBAL"]["num_threads"])
    with Pool(num_threads) as workers:
        exIt(workers, config)

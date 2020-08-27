import silence_tensorflow
from collections import deque
import tensorflow as tf
from anthony_net.NNAgent import NNAgent
from nmcts.NMCTSAgent import NMCTSAgent
from Agent import RandomAgent
from mcts.MCTSAgent import MCTSAgent
from anthony_net.utils import generate_board, convert_state_batch
import numpy as np
from minihex import player, HexGame
import tqdm
from tqdm.keras import TqdmCallback
from anthony_net.network import gen_model, selective_loss
import itertools
from nmcts.NeuralSearchNode import NeuralSearchNode
from multiprocessing import Pool, cpu_count
from utils import simulate, nmcts_builder, step_and_rollout
from anthony_net.train_network import train_network, load_data
import configparser
from utils import save_array
import os


def task_iter(queue):
    while queue:
        yield queue.popleft()


def build_expert(apprentice_agent, board_size, depth):
    return NMCTSAgent(board_size=board_size,
                      depth=depth,
                      agent=apprentice_agent)


def build_apprentice(samples, labels, config, workers):
    board_size = int(config["GLOBAL"]["board_size"])
    chunksize = int(config["BuildApprentice"]["chunksize"])

    boards, players = samples
    sim_args = zip(players, boards, players)
    data = [sim for sim in workers.starmap(
        HexGame,
        sim_args,
        chunksize=chunksize)]
    data = np.stack(convert_state_batch(data))

    labels = tf.one_hot(labels, board_size ** 2)
    labels = (labels[:, 0, :], labels[:, 1, :])

    tf.keras.backend.clear_session()
    network = train_network(data, labels, config)
    return NNAgent(network)


@save_array("logs/iteration_{idx}/samples")
def generate_samples(config, workers):
    # the paper generates a rollout via selfplay and then samples a single
    # position from the entire trajectory to avoid correlations in the data
    # Here we improve this, by directly sampling a game position randomly
    # (0 correlation with other states in the dataset)

    num_samples = int(config["GLOBAL"]["dataset_size"])
    board_size = int(config["GLOBAL"]["board_size"])

    board_positions = list()
    active_players = list()

    chunksize = int(config["GenerateSamples"]["chunksize"])
    result = tqdm.tqdm(workers.imap(generate_board,
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


@save_array("logs/iteration_{idx}/labels")
def compute_labels(samples, expert, config, workers):
    board_positions, active_players = samples
    labels = list()
    batch_size = int(config["GLOBAL"]["dataset_size"])
    chunksize = int(config["ComputeLabels"]["chunksize"])
    nn_agent = expert.agent
    depth = expert.depth
    pbar = tqdm.tqdm(
        desc="Generating Labels",
        total=batch_size,
        position=1,
        leave=False)

    sim_args = zip(active_players, board_positions, active_players)
    initial_sims = [sim for sim in tqdm.tqdm(
        workers.starmap(HexGame,
                        sim_args,
                        chunksize=chunksize),
        desc="Building Environments",
        position=2,
        leave=False,
        total=batch_size)]
    converted_boards = convert_state_batch(initial_sims)
    policies = nn_agent.get_scores(converted_boards, active_players)

    nmcts_args = zip([depth] * batch_size, initial_sims, policies)
    agents = [agent for agent in tqdm.tqdm(
        workers.imap(nmcts_builder,
                     nmcts_args,
                     chunksize=chunksize),
        desc="Initializing Agents",
        position=2,
        leave=False,
        total=batch_size)]

    tasks = deque()
    for idx, agent in enumerate(agents):
        gen = agent.deferred_plan()
        tasks.append((idx, "init", gen, None))

    active_queue_size = int(512 * 3)
    active_tasks = deque()
    for idx in range(min(active_queue_size, len(tasks))):
        active_tasks.append(tasks.popleft())

    while active_tasks:
        expand_and_sim_batch = list()
        for task in task_iter(active_tasks):
            job = task[1]
            if job == "init":
                idx, job, gen, args = task
                job, args = gen.send(None)
                active_tasks.append((idx, job, gen, args))
            elif job == "expand_and_simulate":
                expand_and_sim_batch.append(task)
            elif job == "done":
                idx, _, _, _ = task
                action = agents[idx].act_greedy(None, None, None)
                if initial_sims[idx].active_player == player.WHITE:
                    labels.append((-1, action))
                else:
                    labels.append((action, -1))
                pbar.update(1)

                # don't keep the tree in memory after it has finished
                agents[idx] = None

                if tasks:
                    active_tasks.append(tasks.popleft())

        # handle expand and simulate
        if expand_and_sim_batch:
            sim_batch = list()
            history_batch = list()
            for task in expand_and_sim_batch:
                idx, _, _, action_history = task
                sim_batch.append(initial_sims[idx])
                history_batch.append(action_history)
            results = workers.starmap(
                step_and_rollout, zip(sim_batch, history_batch))
            envs = list()
            players = list()
            winners = list()
            for env, winner in results:
                envs.append(env)
                players.append(env.active_player)
                winners.append(winner)
            board_batch = np.stack(convert_state_batch(envs))
            players = np.stack(players)
            policies = nn_agent.get_scores(board_batch, players)

            result_iter = zip(expand_and_sim_batch, policies, winners, envs)
            for task, policy, winner, env in result_iter:
                idx, _, gen, action_history = task
                node = NeuralSearchNode(env, network_policy=policy)
                try:
                    job, args = gen.send((node, winner))
                    active_tasks.append((idx, job, gen, args))
                except StopIteration:
                    active_tasks.append((idx, "done", gen, None))

    return np.stack(labels, axis=0)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('ExIt.ini')

    board_size = int(config["GLOBAL"]["board_size"])
    search_depth = int(config["GLOBAL"]["search_depth"])
    iterations = int(config["GLOBAL"]["iterations"])

    expert = NMCTSAgent(board_size=board_size,
                        depth=search_depth, model_file="best_model.h5")
    with Pool(cpu_count() - 2) as workers:
        pbar = tqdm.tqdm(range(iterations),
                         desc="Training Experts",
                         position=0)
        for idx in pbar:
            os.makedirs(f"logs/iteration_{idx}", exist_ok=True)
            config["Training"]["model_file"] = f"logs/iteration_{idx}/model.h5"
            config["Training"]["history_file"] = (
                f"logs/iteration_{idx}/history.pickle")

            samples = generate_samples(config, workers)
            labels = compute_labels(samples, expert, config, workers)

            apprentice = build_apprentice(samples, labels, config, workers)

            expert = build_expert(apprentice, board_size, search_depth)

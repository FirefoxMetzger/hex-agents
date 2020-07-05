import silence_tensorflow
from anthony_net.NNAgent import NNAgent
from nmcts.NMCTSAgent import NMCTSAgent
from nmcts.NeuralSearchNode import NeuralSearchNode
from minihex import HexGame, player
import gym
import numpy as np
import itertools
from anthony_net.utils import convert_state
from collections import deque
from multiprocessing import Pool, freeze_support
from rollout import simulate


def task_iter(queue):
    while queue:
        yield queue.popleft()


def initialize_tasks(nn_agent):
    board_size = 5
    board = player.EMPTY * np.ones((board_size, board_size))
    env = HexGame(player.BLACK, board, player.BLACK)
    initial_policy = nn_agent.predict_env(env)
    agents = [NMCTSAgent(depth=1000, board_size=board_size, env=env,
                         agent=nn_agent, network_policy=initial_policy)
              for _ in range(120)]

    tasks = deque()
    for idx, agent in enumerate(agents):
        gen = agent.deferred_plan()
        tasks.append((idx, "init", gen, None))
    return agents, tasks


def main_loop(agents, tasks, nn_agent, workers):
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
                    pass

        # handle simulation
        if simulation_batch:
            sims = [task[3] for task in simulation_batch]
            winners = workers.map(simulate, sims)
            for task, winner in zip(simulation_batch, winners):
                idx, job, gen, _ = task
                try:
                    job, args = gen.send(winner)
                    tasks.append((idx, job, gen, args))
                except StopIteration:
                    pass


if __name__ == "__main__":
    nn_agent = NNAgent("best_model.h5")
    agents, tasks = initialize_tasks(nn_agent)
    with Pool(4) as workers:
        main_loop(agents, tasks, nn_agent, workers)

    print([agent.root_node.total_simulations for agent in agents])

    # for agent in agents:
    #     agent.act_greedy(env.board, player.BLACK, None)

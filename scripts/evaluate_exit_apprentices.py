import silence_tensorflow
import gym
import minihex
from minihex import player
from minihex.HexGame import HexEnv, HexGame
import numpy as np
import tqdm
import random
import trueskill
import json
from anthony_net.utils import convert_state_batch
from multiprocessing import Pool
import configparser
from copy import deepcopy
import os


from scheduler.scheduler import Scheduler, Task, DoneTask, FinalHandler
from scheduler.handlers import (
    HandleRollout,
    HandleNNEval,
    Handler
)
from scheduler.tasks import NNEval
from utils import InitGame, HandleDone, HandleInit, play_match
from plotting import plot_iteration
from anthony_net.NNAgent import NNAgent
from mcts.MCTSAgent import MCTSAgent
from Agent import Agent


class ApprenticeHandleDone(HandleDone):
    def __init__(self, rating_agents, config):
        super(ApprenticeHandleDone, self).__init__(
            rating_agents,
            config
        )

        board_size = int(config["GLOBAL"]["board_size"])
        log_dir = config["GLOBAL"]["log_dir"]
        mcts_dir = config["mctsEval"]["dir"]
        mcts_dir = mcts_dir.format(board_size=board_size)
        mcts_rating_file = config["mctsEval"]["eval_file"]
        mcts_rating_file = "/".join([log_dir, mcts_dir, mcts_rating_file])
        with open(mcts_rating_file, "r") as ratings_f:
            mcts_ratings = json.load(ratings_f)

        self.mcts_ratings = {
            player.WHITE: dict(),
            player.BLACK: dict()
        }

        for key in self.mcts_ratings:
            mus = mcts_ratings[str(int(key))]["mu"]
            sigmas = mcts_ratings[str(int(key))]["sigma"]
            depths = mcts_ratings[str(int(key))]["depth"]
            for mu, sigma, depth in zip(mus, sigmas, depths):
                self.mcts_ratings[key][depth] = trueskill.Rating(
                    mu=mu, sigma=sigma)

    def get_ratings(self, task):
        opponent_color = task.metadata["opponent_color"]
        opponent_key = task.metadata["opponent"].depth
        opponent_rating = self.mcts_ratings[opponent_color][opponent_key]

        agent_color = task.metadata["agent_color"]
        agent_key = task.metadata["iteration"]
        agent = self.rating_agents[agent_color][agent_key]

        return agent.rating, opponent_rating

    def get_result_storage(self, task):
        agent_color = task.metadata["agent_color"]
        agent_key = task.metadata["iteration"]
        agent = self.rating_agents[agent_color][agent_key]
        return agent, Agent()


class HandleMultiNNEval(Handler):
    allowed_task = NNEval

    def __init__(self, nn_agents):
        self.nn_agents = nn_agents

    def handle_batch(self, batch):
        sub_batches = dict()

        players = np.stack([task.sim.active_player for task in batch])
        sims = [task.sim for task in batch]
        boards = np.stack(convert_state_batch(sims))
        possible_actions = [task.sim.get_possible_actions() for task in batch]
        scores = np.empty((len(batch), batch[0].sim.board.size))

        for idx, task in enumerate(batch):
            iteration = task.metadata["iteration"]
            if iteration not in sub_batches:
                sub_batches[iteration] = list()

            sub_batches[iteration].append(idx)

        for iteration, batch in sub_batches.items():
            nn_agent = self.nn_agents[iteration]
            indices = np.asarray(batch)

            batch_players = players[indices]
            batch_boards = boards[indices]
            scores[indices, ...] = nn_agent.get_scores(
                batch_boards,
                batch_players
            )

        actions = list()
        for score, possible in zip(scores, possible_actions):
            action_idx = np.argmax(score[possible])
            actions.append(possible[action_idx])

        return actions


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('ExIt.ini')

    board_size = int(config["GLOBAL"]["board_size"])
    log_dir = config["GLOBAL"]["log_dir"]
    out_dir = config["apprenticeEval"]["dir"]
    out_dir = out_dir.format(board_size=board_size)
    base_dir = "/".join([log_dir, out_dir])
    os.makedirs(base_dir, exist_ok=True)

    mcts_location = config["mctsEval"]["dir"]
    mcts_location = mcts_location.format(board_size=board_size)
    mcts_rating_file = config["mctsEval"]["eval_file"]
    mcts_rating_file = "/".join([log_dir, mcts_location, mcts_rating_file])

    exit_dir = config["ExpertIteration"]["dir"]
    exit_dir = exit_dir.format(board_size=board_size)
    step_location = config["ExpertIteration"]["step_location"]
    agent_template = "/".join([log_dir, exit_dir, step_location, "model.h5"])

    trueskill.setup(mu=float(config["TrueSkill"]["initial_mu"]),
                    sigma=float(config["TrueSkill"]["initial_sigma"]),
                    beta=float(config["TrueSkill"]["beta"]),
                    tau=float(config["TrueSkill"]["tau"]),
                    draw_probability=float(
                        config["TrueSkill"]["draw_probability"]),
                    backend="scipy")

    num_threads = int(config["GLOBAL"]["num_threads"])
    iterations = int(config["ExpertIteration"]["iterations"])

    model_files = [
        agent_template.format(idx=idx) for idx in range(iterations)]
    agents = [NNAgent(model_file) for model_file in model_files]
    rating_agents = {
        player.WHITE: {iteration: Agent() for iteration in range(iterations)},
        player.BLACK: {iteration: Agent() for iteration in range(iterations)}
    }
    depths = [0, 50, 100, 500, 1000, 1500, 2000,
              2500, 3000, 3500, 4000, 4500, 5000]

    num_matches = int(config["GLOBAL"]["num_matches"])
    queue = list()
    for iteration in range(iterations):
        nn_agent = agents[iteration]
        add_to_queue = list()
        add_to_queue += [InitGame(idx, nn_agent, player.WHITE)
                         for idx in range(num_matches)]
        add_to_queue += [InitGame(idx, nn_agent, player.BLACK)
                         for idx in range(num_matches)]
        for task in add_to_queue:
            task.metadata["iteration"] = iteration

        queue += add_to_queue

    with Pool(num_threads) as workers:
        handlers = [
            HandleInit(depths, config),
            HandleRollout(workers, config),
            ApprenticeHandleDone(rating_agents, config),
            HandleMultiNNEval(agents)
        ]
        sched = Scheduler(handlers)

        queue_bar = tqdm.tqdm(
            total=len(queue),
            desc="Games Played")
        max_active = int(config["GLOBAL"]["active_simulations"])
        active_tasks = queue[:max_active]
        queue = queue[max_active:]
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
            queue_bar.update(completed)

    ratings = {
        player.WHITE: {"mu": [], "sigma": [], "iterations": []},
        player.BLACK: {"mu": [], "sigma": [], "iterations": []}
    }

    for key in ratings:
        for it in range(iterations):
            ratings[key]["mu"].append(rating_agents[key][it].rating.mu)
            ratings[key]["sigma"].append(rating_agents[key][it].rating.sigma)
        ratings[key]["iterations"] = [it for it in range(iterations)]

    eval_file = config["apprenticeEval"]["eval_file"]
    eval_file = "/".join([base_dir, eval_file])
    with open(eval_file, "w") as json_file:
        json.dump(ratings, json_file)

    plot_file = config["apprenticeEval"]["plot_file"]
    plot_file = "/".join([base_dir, plot_file])
    plot_iteration(ratings, plot_file, config)

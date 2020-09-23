import silence_tensorflow
from Agent import Agent
from anthony_net.NNAgent import NNAgent
from mcts.MCTSAgent import MCTSAgent
from nmcts.NMCTSAgent import NMCTSAgent
import gym
import minihex
from minihex import player
import numpy as np
import tqdm
import random
import trueskill
import json
import configparser
from multiprocessing import Pool
import os

from scheduler.scheduler import Scheduler, Task, DoneTask, FinalHandler
from scheduler.handlers import (
    HandleNNPolicy,
    HandleRollout,
    Handler
)
from utils import play_match, InitGame, HandleInit, HandleDone
from plotting import plot_expansions


class NMCTSDone(HandleDone):
    def __init__(self, rating_agents, config):
        super(NMCTSDone, self).__init__(
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
        agent_key = task.metadata["agent"].depth
        agent = self.rating_agents[agent_color][agent_key]

        return agent.rating, opponent_rating

    def get_result_storage(self, task):
        agent_color = task.metadata["agent_color"]
        agent_key = task.metadata["agent"].depth
        agent = self.rating_agents[agent_color][agent_key]

        return agent, Agent()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('ExIt.ini')

    board_size = int(config["GLOBAL"]["board_size"])
    log_dir = config["GLOBAL"]["log_dir"]
    out_dir = config["expertEval"]["dir"]
    out_dir = out_dir.format(board_size=board_size)
    base_dir = "/".join([log_dir, out_dir])
    os.makedirs(base_dir, exist_ok=True)

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
    board_size = int(config["GLOBAL"]["board_size"])

    model_file = config["expertEval"]["model_file"]
    nn_agent = NNAgent(model_file)
    depths = [0, 50, 100, 500, 1000, 1500, 2000,
              2500, 3000, 3500, 4000, 4500, 5000]
    num_matches = int(config["GLOBAL"]["num_matches"])

    rating_agents = {
        player.BLACK: {depth: Agent() for depth in depths},
        player.WHITE: {depth: Agent() for depth in depths}
    }

    mcts_depths = [0, 50, 100, 500, 1000, 1500, 2000,
                   2500, 3000, 3500, 4000, 4500, 5000]
    num_threads = int(config["GLOBAL"]["num_threads"])
    with Pool(num_threads) as workers:
        handlers = [
            HandleInit(
                depths=mcts_depths,
                config=config
            ),
            HandleNNPolicy(nn_agent=nn_agent),
            HandleRollout(
                workers=workers,
                config=config
            ),
            NMCTSDone(
                rating_agents=rating_agents,
                config=config
            )
        ]
        sched = Scheduler(handlers)

        queue = list()
        for depth in depths:
            add_to_queue = [
                InitGame(
                    idx,
                    NMCTSAgent(
                        depth=depth,
                        board_size=board_size,
                        agent=nn_agent
                    ),
                    color)
                for idx in range(num_matches)
                for color in [player.BLACK, player.WHITE]
            ]
            for task in add_to_queue:
                task.metadata["iteration"] = depth

            queue += add_to_queue

        queue_bar = tqdm.tqdm(
            total=len(queue),
            desc="Games Played")
        max_active = int(config["GLOBAL"]["active_simulations"])
        active_tasks = queue[: max_active]
        queue = queue[max_active:]
        while queue or active_tasks:
            if len(active_tasks) < max_active:
                num_new = max_active - len(active_tasks)
                num_new = min(num_new, len(queue))
                new_tasks = queue[: num_new]
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
        for it in depths:
            ratings[key]["mu"].append(rating_agents[key][it].rating.mu)
            ratings[key]["sigma"].append(rating_agents[key][it].rating.sigma)
        ratings[key]["depth"] = depths

    eval_file = config["expertEval"]["depth_eval_file"]
    eval_file = "/".join([base_dir, eval_file])
    with open(eval_file, "w") as json_file:
        json.dump(ratings, json_file)

    plot_file = config["expertEval"]["depth_plot_file"]
    plot_file = "/".join([base_dir, plot_file])
    plot_expansions(ratings, plot_file, config)

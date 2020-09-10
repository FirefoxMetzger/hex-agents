import silence_tensorflow
from Agent import Agent
from mcts.MCTSAgent import MCTSAgent
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
from utils import InitGame, HandleDone, play_match, HandleInit
from plotting import plot_expansions
from scheduler.handlers import (
    HandleRollout,
    Handler
)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('ExIt.ini')

    board_size = int(config["GLOBAL"]["board_size"])
    num_threads = int(config["GLOBAL"]["num_threads"])
    num_matches = int(config["GLOBAL"]["num_matches"])
    max_active = int(config["GLOBAL"]["active_simulations"])

    log_dir = config["GLOBAL"]["log_dir"]
    out_dir = config["mctsEval"]["dir"]
    base_dir = "/".join([log_dir, out_dir])
    base_dir = base_dir.format(board_size=board_size)
    os.makedirs(base_dir, exist_ok=True)

    trueskill.setup(mu=float(config["TrueSkill"]["initial_mu"]),
                    sigma=float(config["TrueSkill"]["initial_sigma"]),
                    beta=float(config["TrueSkill"]["beta"]),
                    tau=float(config["TrueSkill"]["tau"]),
                    draw_probability=float(
                        config["TrueSkill"]["draw_probability"]),
                    backend="scipy")

    depths = [0, 50, 100, 500, 1000, 1500, 2000,
              2500, 3000, 3500, 4000, 4500, 5000]
    rating_agents = {
        player.WHITE: {depth: Agent() for depth in depths},
        player.BLACK: {depth: Agent() for depth in depths}
    }

    num_threads = int(config["GLOBAL"]["num_threads"])
    with Pool(num_threads) as workers:
        handlers = [
            HandleInit(depths, config),
            HandleRollout(workers, config),
            HandleDone(rating_agents, config)
        ]
        sched = Scheduler(handlers)

        queue = list()
        for depth in depths:
            queue += [
                InitGame(
                    idx,
                    MCTSAgent(
                        depth=depth,
                        board_size=board_size
                    ),
                    player.WHITE
                )
                for idx in range(num_matches)
            ]
            queue += [
                InitGame(
                    idx,
                    MCTSAgent(
                        depth=depth,
                        board_size=board_size
                    ),
                    player.BLACK
                )
                for idx in range(num_matches)
            ]

        queue_bar = tqdm.tqdm(
            total=len(queue),
            desc="Games Played")
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
        color: {
            "mu": list(),
            "sigma": list(),
            "depth": list()
        } for color in [player.BLACK, player.WHITE]
    }

    for color in [player.BLACK, player.WHITE]:
        for depth in depths:
            ratings[color]["mu"].append(rating_agents[color][depth].rating.mu)
            ratings[color]["sigma"].append(
                rating_agents[color][depth].rating.sigma
            )
        ratings[color]["depth"] = depths

    eval_file = config["mctsEval"]["eval_file"]
    eval_file = "/".join([base_dir, eval_file])
    with open(eval_file, "w") as json_file:
        json.dump(ratings, json_file)

    plot_file = config["mctsEval"]["plot_file"]
    plot_file = "/".join([base_dir, plot_file])
    plot_expansions(ratings, plot_file, config)

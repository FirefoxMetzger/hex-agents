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
from scheduler.handlers import (
    HandleRollout,
    Handler
)


def play_match(env, agent, opponent):
    state, info = env.reset()

    info_opponent = {
        'last_move_opponent': None,
        'last_move_player': env.previous_opponent_move
    }
    yield from opponent.update_root_state_deferred(info_opponent)

    info = {
        'last_move_opponent': env.previous_opponent_move,
        'last_move_player': None
    }
    yield from agent.update_root_state_deferred(info)

    done = False
    while not done:
        yield from agent.deferred_plan()
        action = agent.act_greedy(state[0], None, None)

        info_opponent = {
            'last_move_opponent': action,
            'last_move_player': None
        }
        yield from opponent.update_root_state_deferred(info_opponent)
        info = {
            'last_move_opponent': None,
            'last_move_player': action
        }
        yield from agent.update_root_state_deferred(info)

        yield from opponent.deferred_plan()
        state, reward, done, info = env.step(action)

        if not done:
            info_opponent = {
                'last_move_opponent': None,
                'last_move_player': env.previous_opponent_move
            }
            yield from opponent.update_root_state_deferred(info_opponent)
            info = {
                'last_move_opponent': env.previous_opponent_move,
                'last_move_player': None
            }
            yield from agent.update_root_state_deferred(info)

    task = DoneTask()
    task.metadata["result"] = reward
    yield task


class InitGame(Task):
    def __init__(self, idx, agent):
        super(InitGame, self).__init__()
        self.metadata = {
            "idx": idx,
            "agent": agent
        }


class HandleInit(Handler):
    allowed_task = InitGame

    def __init__(self, config):
        self.board_size = int(config["GLOBAL"]["board_size"])
        self.depths = [0, 50, 100, 500, 1000, 1500, 2000,
                       2500, 3000, 3500, 4000, 4500, 5000]

    def handle_batch(self, batch):
        for task in batch:
            if random.random() > 0.5:
                player_color = player.BLACK
            else:
                player_color = player.WHITE

            depth = self.depths[random.randint(0, len(self.depths) - 1)]
            opponent = MCTSAgent(depth=depth, board_size=self.board_size)

            env = gym.make(
                "hex-v0",
                player_color=player_color,
                opponent_policy=opponent.act_greedy,
                board_size=self.board_size)

            task.metadata["opponent"] = opponent

            task.gen = play_match(env, task.metadata["agent"], opponent)

        return [None] * len(batch)


class HandleDone(FinalHandler):
    allowed_task = DoneTask

    def __init__(self, rating_agents, config):
        self.rating_agents = rating_agents

    def handle_batch(self, batch):
        for task in batch:
            result = task.metadata["result"]
            opponent_key = task.metadata["opponent"].depth
            opponent = self.rating_agents[opponent_key]
            agent_key = task.metadata["agent"].depth
            agent = self.rating_agents[agent_key]

            if result == -1:
                opponent, agent.rating = trueskill.rate_1vs1(
                    opponent.rating, agent.rating)
            elif result == 1:
                agent.rating, opponent = trueskill.rate_1vs1(
                    agent.rating, opponent.rating)
            else:
                agent.rating, opponent = trueskill.rate_1vs1(
                    agent.rating, opponent.rating, drawn=True)


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
    rating_agents = {depth: Agent() for depth in depths}

    num_threads = int(config["GLOBAL"]["num_threads"])
    with Pool(num_threads) as workers:
        handlers = [
            HandleInit(config),
            HandleRollout(workers, config),
            HandleDone(rating_agents, config)
        ]
        sched = Scheduler(handlers)

        agent_bar = tqdm.tqdm(
            iter(depths),
            desc="Agents",
            total=len(depths),
            leave=False)
        queue = list()
        for depth in agent_bar:
            queue += [
                InitGame(
                    idx,
                    MCTSAgent(
                        depth=depth,
                        board_size=board_size
                    )
                )
                for idx in range(num_matches)
            ]

        active_tasks = queue[:max_active]
        queue = queue[max_active:]
        queue_bar = tqdm.tqdm(
            total=num_matches*len(depths),
            desc="Games Played")
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
        "mu": [rating_agents[key].rating.mu for key in rating_agents],
        "sigma": [rating_agents[key].rating.sigma for key in rating_agents],
        "depth": [key for key in rating_agents]
    }

    eval_file = config["mctsEval"]["eval_file"]
    eval_file = "/".join([base_dir, eval_file])
    with open(eval_file, "w") as json_file:
        json.dump(ratings, json_file)

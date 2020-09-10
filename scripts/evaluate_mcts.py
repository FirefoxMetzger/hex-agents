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
from plotting import plot_expansions
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
    def __init__(self, idx, agent, agent_color):
        super(InitGame, self).__init__()

        if agent_color == player.BLACK:
            opponent_color = player.WHITE
        else:
            opponent_color = player.BLACK

        self.metadata = {
            "idx": idx,
            "agent": agent,
            "agent_color": agent_color,
            "opponent_color": opponent_color
        }


class HandleInit(Handler):
    allowed_task = InitGame

    def __init__(self, depths, config):
        self.board_size = int(config["GLOBAL"]["board_size"])
        self.depths = depths

    def handle_batch(self, batch):
        for task in batch:
            player_color = task.metadata["agent_color"]

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
            agent_color = task.metadata["agent_color"]
            opponent_color = task.metadata["opponent_color"]
            opponent_key = task.metadata["opponent"].depth
            opponent = self.rating_agents[opponent_color][opponent_key]
            agent_key = task.metadata["agent"].depth
            agent = self.rating_agents[agent_color][agent_key]

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

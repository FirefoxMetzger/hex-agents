import silence_tensorflow
from anthony_net.NNAgent import NNAgent
from mcts.MCTSAgent import MCTSAgent
import gym
import minihex
from minihex import player
from minihex.HexGame import HexEnv, HexGame
import numpy as np
import tqdm
import random
from nmcts.NMCTSAgent import NMCTSAgent
import trueskill
import json
from utils import step_and_rollout
from anthony_net.utils import convert_state_batch
from nmcts.NeuralSearchNode import NeuralSearchNode
from mcts.SearchNode import SearchNode
from multiprocessing import Pool
import configparser
from copy import deepcopy
import os


from scheduler.scheduler import Scheduler, Task, DoneTask, FinalHandler
from scheduler.handlers import (
    HandleRollout,
    HandleNNEval,
    Handler,
)
from scheduler.tasks import NNEval


def play_match(env, agent, opponent):
    state, info = env.reset()

    info_opponent = {
        'last_move_opponent': None,
        'last_move_player': env.previous_opponent_move
    }
    yield from opponent.update_root_state_deferred(info_opponent)

    done = False
    while not done:
        task = NNEval(env.simulator)
        action = yield task

        info_opponent = {
            'last_move_opponent': action,
            'last_move_player': None
        }
        yield from opponent.update_root_state_deferred(info_opponent)
        yield from opponent.deferred_plan()
        state, reward, done, info = env.step(action)

        if not done:
            info_opponent = {
                'last_move_opponent': None,
                'last_move_player': env.previous_opponent_move
            }
            yield from opponent.update_root_state_deferred(info_opponent)

    task = DoneTask()
    task.metadata["result"] = reward
    yield task


class InitGame(Task):
    def __init__(self, idx, nn_agent):
        super(InitGame, self).__init__()
        self.metadata = {
            "idx": idx,
            "agent": nn_agent
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

    def __init__(self, ratings, config):
        board_size = int(config["GLOBAL"]["board_size"])
        self.mcts_ratings = ratings

    def handle_batch(self, batch):
        for task in batch:
            result = task.metadata["result"]
            opponent = task.metadata["opponent"]
            agent = task.metadata["agent"]

            idx = self.mcts_ratings["depth"].index(opponent.depth)
            mu = self.mcts_ratings["mu"][idx]
            sigma = self.mcts_ratings["sigma"][idx]
            op_rating = trueskill.Rating(mu, sigma)
            if result == -1:
                _, agent.rating = trueskill.rate_1vs1(
                    op_rating, agent.rating)
            elif result == 1:
                agent.rating, _ = trueskill.rate_1vs1(
                    agent.rating, op_rating)
            else:
                agent.rating, _ = trueskill.rate_1vs1(
                    agent.rating, op_rating, drawn=True)


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

    with open(mcts_rating_file, "r") as ratings_f:
        mcts_ratings = json.load(ratings_f)

    num_threads = int(config["GLOBAL"]["num_threads"])
    iterations = int(config["ExpertIteration"]["iterations"])

    model_files = [
        agent_template.format(idx=idx) for idx in range(iterations)]
    agents = [NNAgent(model_file) for model_file in model_files]

    num_matches = int(config["GLOBAL"]["num_matches"])

    with Pool(num_threads) as workers:
        handlers = [
            HandleInit(config),
            HandleRollout(workers, config),
            HandleDone(mcts_ratings, config),
            HandleNNEval(None)
        ]
        sched = Scheduler(handlers)

        agent_bar = tqdm.tqdm(
            iter(agents),
            desc="Agents",
            total=len(agents))
        for nn_agent in agent_bar:
            handlers[-1] = HandleNNEval(nn_agent)
            queue = [InitGame(idx, nn_agent)
                     for idx in range(num_matches)]

            max_active = int(config["GLOBAL"]["active_simulations"])
            active_tasks = queue[:max_active]
            queue = queue[max_active:]
            queue_bar = tqdm.tqdm(
                total=num_matches,
                desc="Games Played",
                leave=False,
                position=1)
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
        "mu": [agent.rating.mu for agent in agents],
        "sigma": [agent.rating.sigma for agent in agents],
        "model": [model for model in model_files]
    }

    eval_file = config["apprenticeEval"]["eval_file"]
    eval_file = "/".join([base_dir, eval_file])
    with open(eval_file, "w") as json_file:
        json.dump(ratings, json_file)

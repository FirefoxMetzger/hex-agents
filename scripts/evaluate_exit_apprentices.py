import silence_tensorflow
from Agent import RandomAgent
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


from scheduler.scheduler import Scheduler, Task, DoneTask, FinalHandler
from scheduler.handlers import (
    HandleMCTSExpandAndSimulate,
    HandleExpandAndSimulate,
    HandleNNEval,
    Handler,
    HandleMetadataUpdate
)
from scheduler.tasks import (
    MCTSExpandAndSimulate,
    ExpandAndSimulate,
    NNEval,
    UpdateMetadata
)


def play_match(env, agent, opponent):
    state, info = env.reset()
    # task = UpdateMetadata()
    # task.metadata["sim"] = env.simulator
    # yield task

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
            if depth == 0:
                opponent = RandomAgent(self.board_size)
            else:
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

    def __init__(self, config):
        board_size = int(config["GLOBAL"]["board_size"])
        rating_file = config["mctsEval"]["eval_file"]
        rating_file = rating_file.format(board_size=board_size)
        with open(rating_file, "r") as rating_f:
            mcts_ratings = json.load(rating_f)
        self.mcts_ratings = mcts_ratings

    def handle_batch(self, batch):
        for task in batch:
            result = task.metadata["result"]
            opponent = task.metadata["opponent"]
            agent = task.metadata["agent"]

            rating = self.mcts_ratings[str(opponent)]
            op_rating = trueskill.Rating(rating["mu"], rating["sigma"])
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

    trueskill.setup(mu=1200,
                    sigma=200.0,
                    beta=100.0,
                    tau=2.0,
                    draw_probability=0.01,
                    backend="scipy")

    num_threads = int(config["GLOBAL"]["num_threads"])
    iterations = int(config["ExpertIteration"]["iterations"])

    model_files = ["best_model.h5"]
    model_files += [
        f"logs/iteration_{idx}/model.h5" for idx in range(iterations)]
    agents = [NNAgent(model_file) for model_file in model_files]

    num_matches = int(config["exitEval"]["num_matches"])

    handlers = [
        HandleInit(config),
        HandleMCTSExpandAndSimulate(),
        HandleDone(config),
        HandleMetadataUpdate(),
        HandleNNEval(None)
    ]
    sched = Scheduler(handlers)

    agent_bar = tqdm.tqdm(
        iter(agents),
        desc="Agents",
        total=len(agents))
    for nn_agent in agents:
        handlers[-1] = HandleNNEval(nn_agent)
        queue = [InitGame(idx, nn_agent)
                 for idx in range(num_matches)]

        max_active = int(config["apprenticeEval"]["active_simulations"])
        active_tasks = queue[:max_active]
        queue = queue[max_active:]
        queue_bar = tqdm.tqdm(
            total=num_matches,
            desc="Games Played",
            leave=False)
        while queue or active_tasks:
            if len(active_tasks) < max_active:
                num_new = max_active - len(active_tasks)
                num_new = min(num_new, len(queue))
                new_tasks = queue[:num_new]
                active_tasks += new_tasks
                queue = queue[num_new:]
                queue_bar.update(num_new)

            active_tasks = sched.process(active_tasks)

    ratings = {
        model: {"mu": agent.rating.mu,
                "sigma": agent.rating.sigma}
        for model, agent in zip(model_files, agents)
    }

    board_size = int(config["GLOBAL"]["board_size"])
    eval_file = config["apprenticeEval"]["eval_file"]
    with open(eval_file.format(board_size=board_size), "w") as json_file:
        json.dump(ratings, json_file)

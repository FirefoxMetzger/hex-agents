from minihex import player, HexGame
import numpy as np

from utils import step_and_rollout
from nmcts.NeuralSearchNode import NeuralSearchNode
from scheduler.scheduler import Task, Handler, DoneTask, FinalHandler
from .tasks import *
from nmcts.NMCTSAgent import NMCTSAgent
from anthony_net.utils import convert_state_batch


class HandleInit(Handler):
    allowed_task = InitExit

    def __init__(self, nn_agent, config, workers):
        self.nn_agent = nn_agent
        self.config = config
        self.workers = workers

    def handle_batch(self, batch):
        batch_size = len(batch)
        chunksize = int(self.config["GLOBAL"]["chunksize"])
        depth = int(self.config["NMCTSAgent"]["search_depth"])

        sims = list()
        active_players = list()
        for task in batch:
            board_position, active_player = task.metadata["sample"]
            sim = HexGame(active_player, board_position, active_player)
            active_players.append(active_player)
            sims.append(sim)
        converted_boards = convert_state_batch(sims)
        policies = self.nn_agent.get_scores(
            converted_boards,
            np.asarray(active_players))

        for idx, task in enumerate(batch):
            agent = NMCTSAgent(
                depth=depth,
                env=sims[idx],
                network_policy=policies[idx])

            task.metadata.update({
                "sim": sims[idx],
                "agent": agent
            })

            task.gen = agent.deferred_plan()

        return [None] * len(batch)


class HandleExpandAndSimulate(Handler):
    allowed_task = ExpandAndSimulate

    def __init__(self, nn_agent, workers):
        self.workers = workers
        self.nn_agent = nn_agent

    def handle_batch(self, batch):
        envs = list()
        rollout_results = list()
        for task in batch:
            sim = task.metadata["sim"]
            hist = task.metadata["action_history"]
            env, result = step_and_rollout(sim, hist)
            envs.append(env)
            rollout_results.append(result)

        players = np.stack([env.active_player for env in envs], axis=0)
        boards = np.stack(convert_state_batch(envs))
        policies = self.nn_agent.get_scores(boards, players)

        results = list()
        for idx, task in enumerate(batch):
            winner = rollout_results[idx]
            policy = policies[idx]
            env = envs[idx]
            node = NeuralSearchNode(env, network_policy=policy)
            results.append((node, winner))

        return results


class HandleDone(FinalHandler):
    allowed_task = DoneTask

    def __init__(self, dataset_size):
        self.labels = np.empty((dataset_size, 2))

    def handle_batch(self, batch):
        for task in batch:
            agent = task.metadata["agent"]
            sim = task.metadata["sim"]
            idx = task.metadata["idx"]

            action = agent.act_greedy(None, None, None)
            if sim.active_player == player.WHITE:
                self.labels[idx] = (-1, action)
            else:
                self.labels[idx] = (action, -1)


class HandleMCTSExpandAndSimulate(Handler):
    allowed_task = MCTSExpandAndSimulate

    def handle_batch(self, batch):
        results = list()
        for task in batch:
            sim = task.metadata["sim"]
            hist = task.metadata["action_history"]
            env, winner = step_and_rollout(sim, hist)
            node = SearchNode(env)
            results.append((node, winner))

        return results


class HandleUpdateEnv(Handler):
    allowed_task = UpdateEnv

    def handle_batch(self, batch):
        return [None] * len(batch)


class HandleNNEval(Handler):
    allowed_task = NNEval

    def handle_batch(self, batch):
        sims = list()
        players = list()
        for task in batch:
            sim = task.metadata["sim"]
            sims.append(sim)
            players.append(sim.active_player)
        players = np.stack(players)
        boards = np.stack(convert_state_batch(sims))

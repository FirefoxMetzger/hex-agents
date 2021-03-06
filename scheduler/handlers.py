from minihex import player, HexGame
import numpy as np

from utils import step_and_rollout
from nmcts.NeuralSearchNode import NeuralSearchNode
from scheduler.scheduler import Task, Handler, DoneTask, FinalHandler
from .tasks import *
from nmcts.NMCTSAgent import NMCTSAgent
from anthony_net.utils import convert_state_batch
from mcts.SearchNode import SearchNode


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


class HandleRollout(Handler):
    allowed_task = Rollout

    def __init__(self, workers, config):
        self.workers = workers
        self.chunksize = int(config["GLOBAL"]["chunksize"])

    def handle_batch(self, batch):
        sims = [task.sim for task in batch]
        histories = [task.action_history for task in batch]
        results = self.workers.starmap(
            step_and_rollout,
            zip(sims, histories),
            chunksize=self.chunksize)
        results = [(env, winner) for env, winner in results]
        return results


class HandleNNPolicy(Handler):
    allowed_task = NNEval

    def __init__(self, nn_agent):
        self.nn_agent = nn_agent

    def handle_batch(self, batch):
        players = np.stack([task.sim.active_player for task in batch])
        sims = [task.sim for task in batch]
        boards = np.stack(convert_state_batch(sims))
        scores = self.nn_agent.get_scores(boards, players)
        return scores


class HandleNNEval(Handler):
    allowed_task = NNEval

    def __init__(self, nn_agent):
        self.nn_agent = nn_agent

    def handle_batch(self, batch):
        players = np.stack([task.sim.active_player for task in batch])
        sims = [task.sim for task in batch]
        boards = np.stack(convert_state_batch(sims))
        possible_actions = [task.sim.get_possible_actions() for task in batch]
        scores = self.nn_agent.get_scores(boards, players)

        actions = list()
        for score, possible in zip(scores, possible_actions):
            action_idx = np.argmax(score[possible])
            actions.append(possible[action_idx])
        return actions

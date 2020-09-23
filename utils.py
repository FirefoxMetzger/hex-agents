from copy import deepcopy
from minihex.HexGame import HexEnv, player
import numpy as np
import trueskill
import random
import gym

from Agent import RandomAgent
from mcts.MCTSAgent import MCTSAgent
from scheduler.scheduler import FinalHandler, Task, DoneTask, Handler
from scheduler.tasks import NNEval
from anthony_net.utils import convert_state_batch


def simulate(env, board_size=5):
    agent = RandomAgent(env.board_size)
    env = HexEnv(
        opponent_policy=agent.act,
        player_color=env.active_player,
        active_player=env.active_player,
        board=env.board.copy(),
        regions=env.regions.copy())

    state, info = env.reset()
    done = False
    while not done:
        action = agent.act(state[0], state[1], info)
        state, reward, done, info = env.step(action)

    return env.simulator.winner


def step_and_rollout(env, action_history):
    new_env = deepcopy(env)
    for action in action_history:
        new_env.make_move(action)

    winner = new_env.winner
    if winner is None:
        winner = simulate(new_env)

    return new_env, winner


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


class HandleDone(FinalHandler):
    allowed_task = DoneTask

    def __init__(self, rating_agents, config):
        self.rating_agents = rating_agents

    def get_ratings(self, task):
        opponent_color = task.metadata["opponent_color"]
        opponent_key = task.metadata["opponent"].depth
        opponent = self.rating_agents[opponent_color][opponent_key]

        agent_color = task.metadata["agent_color"]
        agent_key = task.metadata["agent"].depth
        agent = self.rating_agents[agent_color][agent_key]

        return agent.rating, opponent.rating

    def get_result_storage(self, task):
        opponent_color = task.metadata["opponent_color"]
        opponent_key = task.metadata["opponent"].depth
        opponent = self.rating_agents[opponent_color][opponent_key]

        agent_color = task.metadata["agent_color"]
        agent_key = task.metadata["agent"].depth
        agent = self.rating_agents[agent_color][agent_key]

        return agent, opponent

    def handle_batch(self, batch):
        for task in batch:
            result = task.metadata["result"]

            agent, opponent = self.get_result_storage(task)
            agent_rating, opponent_rating = self.get_ratings(task)

            if result == -1:
                opponent.rating, agent.rating = trueskill.rate_1vs1(
                    opponent_rating, agent_rating)
            elif result == 1:
                agent.rating, opponent.rating = trueskill.rate_1vs1(
                    agent_rating, opponent_rating)
            else:
                agent.rating, opponent.rating = trueskill.rate_1vs1(
                    agent_rating, opponent_rating, drawn=True)


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


def play_match(env, agent, opponent):
    state, info = env.reset()

    info_opponent = {
        'sim': env.simulator,
        'last_move_opponent': None,
        'last_move_player': env.previous_opponent_move
    }
    yield from opponent.update_root_state_deferred(info_opponent)

    info = {
        'sim': env.simulator,
        'last_move_opponent': env.previous_opponent_move,
        'last_move_player': None
    }
    yield from agent.update_root_state_deferred(info)

    done = False
    while not done:
        yield from agent.deferred_plan()
        action = agent.act_greedy(state[0], state[1], None)

        info_opponent = {
            'sim': env.simulator,
            'last_move_opponent': action,
            'last_move_player': None
        }
        yield from opponent.update_root_state_deferred(info_opponent)
        info = {
            'sim': env.simulator,
            'last_move_opponent': None,
            'last_move_player': action
        }
        yield from agent.update_root_state_deferred(info)

        yield from opponent.deferred_plan()
        state, reward, done, info = env.step(action)

        if not done:
            info_opponent = {
                'sim': env.simulator,
                'last_move_opponent': None,
                'last_move_player': env.previous_opponent_move
            }
            yield from opponent.update_root_state_deferred(info_opponent)
            info = {
                'sim': env.simulator,
                'last_move_opponent': env.previous_opponent_move,
                'last_move_player': None
            }
            yield from agent.update_root_state_deferred(info)

    task = DoneTask()
    task.metadata["result"] = reward
    yield task


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

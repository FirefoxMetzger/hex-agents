import gym
import minihex
import numpy as np
from mcts.MCTSAgent import MCTSAgent
from minihex.HexGame import HexGame, player
import trueskill
import json


class Agent(object):
    def __init__(self, depth=1000):
        self.depth = depth
        self.rating = trueskill.Rating()

    def policy(self, state, active_player):
        env = HexGame(active_player, state, active_player)
        agent = MCTSAgent(env)
        return agent.plan(self.depth)

    def __str__(self):
        return f"MCTS({self.depth})"


class RandomAgent(object):
    def __init__(self):
        self.rating = trueskill.Rating()

    def policy(self, state, active_player):
        idx = minihex.empty_tiles(state)
        choice = np.random.randint(len(idx))
        return idx[choice]

    def __str__(self):
        return "Random"


def play_match(agent1, agent2):
    env = gym.make("hex-v0", opponent_policy=agent2.policy)
    state = env.reset()

    done = False
    while not done:
        action = agent1.policy(*state)
        state, reward, done, _ = env.step(action)

    if reward == -1:
        agent2.rating, agent1.rating = trueskill.rate_1vs1(agent2.rating,
                                                           agent1.rating)
    elif reward == 1:
        agent1.rating, agent2.rating = trueskill.rate_1vs1(agent1.rating,
                                                           agent2.rating)
    else:
        agent2.rating, agent1.rating = trueskill.rate_1vs1(agent2.rating,
                                                           agent1.rating,
                                                           draw=True)


def find_match(player1, agent_pool):
    match_quality = list()
    eligable_opponent = list()
    for agent in agent_pool:
        if agent == player1:
            continue
        else:
            match_quality.append(trueskill.quality_1vs1(player1.rating,
                                                        agent.rating))
            eligable_opponent.append(agent)
    match_quality = np.exp(match_quality)/sum(np.exp(match_quality))
    return np.random.choice(eligible_opponent, p=match_quality)


trueskill.setup(mu=1200,
                sigma=200.0,
                beta=100.0,
                tau=2.0,
                draw_probability=0.01,
                backend="scipy")
agent_pool = [Agent(depth=d) for d in [100, 500, 1000, 1500]]
agent_pool += [RandomAgent()]
for game_idx in range(100):
    # pick the agent we are most uncertain about
    sigmas = [agent.rating.sigma for agent in agent_pool]
    player1_idx = np.argmax(sigmas)
    player1 = agent_pool[player1_idx]
    player2 = find_match(player1, agent_pool)
    play_match(player1, player2)

    print("---")
    print(f"MATCH {game_idx}: {player1} vs. {player2}")
    leaderboard = sorted(agent_pool,
                         key=lambda x: trueskill.expose(x.rating),
                         reverse=True)
    for agent in leaderboard:
        print(f"    {agent} mu:{agent.rating.mu:.2f} sigma:{agent.rating.sigma:.2f}")

ratings = {str(agent): {"mu": agent.rating.mu, "sigma": agent.rating.sigma}
           for agent in agent_pool}

with open("mcts_eval.json", "w") as json_file:
    json.dump(ratings, json_file)

import gym
import minihex
import numpy as np
from mcts.MCTSAgent import MCTSAgent
from minihex import HexGame, player
import trueskill
import json
from Agent import Agent, RandomAgent


def play_match(agent1, agent2, sim=None):
    env = gym.make("hex-v0", opponent_policy=agent2.act, board_size=9)
    state, info = env.reset()
    agent1.reset(sim)
    agent2.reset(sim)

    done = False
    while not done:
        action = agent1.act(*state, info)
        state, reward, done, info = env.step(action)

    if reward == -1:
        agent2.rating, agent1.rating = trueskill.rate_1vs1(agent2.rating,
                                                           agent1.rating)
    elif reward == 1:
        agent1.rating, agent2.rating = trueskill.rate_1vs1(agent1.rating,
                                                           agent2.rating)
    else:
        agent2.rating, agent1.rating = trueskill.rate_1vs1(agent2.rating,
                                                           agent1.rating,
                                                           drawn=True)


def find_match(player1, agent_pool):
    match_quality = list()
    eligible_opponent = list()
    for agent in agent_pool:
        if agent == player1:
            continue
        else:
            match_quality.append(trueskill.quality_1vs1(player1.rating,
                                                        agent.rating))
            eligible_opponent.append(agent)
    match_quality = np.exp(match_quality)/sum(np.exp(match_quality))
    return np.random.choice(eligible_opponent, p=match_quality)


trueskill.setup(mu=1200,
                sigma=200.0,
                beta=100.0,
                tau=2.0,
                draw_probability=0.01,
                backend="scipy")
env = gym.make("hex-v0", opponent_policy=None, board_size=9)
env.reset()
sim = env.simulator
agent_pool = [MCTSAgent(sim, depth=d) for d in [10, 50, 100, 500, 1000, 1500]]
agent_pool += [RandomAgent()]

for game_idx in range(1000):
    # pick the agent we are most uncertain about
    sigmas = [agent.rating.sigma for agent in agent_pool]
    player1_idx = np.argmax(sigmas)
    player1 = agent_pool[player1_idx]
    player2 = find_match(player1, agent_pool)
    print("---")
    print(f"MATCH {game_idx}: {player1} vs. {player2}")
    play_match(player1, player2, sim=sim)
    leaderboard = sorted(agent_pool,
                         key=lambda x: trueskill.expose(x.rating),
                         reverse=True)
    for agent in leaderboard:
        print(f"    {agent} mu:{agent.rating.mu:.2f} sigma:{agent.rating.sigma:.2f}")

ratings = {str(agent): {"mu": agent.rating.mu, "sigma": agent.rating.sigma}
           for agent in agent_pool}

with open("mcts_eval.json", "w") as json_file:
    json.dump(ratings, json_file)

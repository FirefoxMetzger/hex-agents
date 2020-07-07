import gym
import minihex
import numpy as np
from mcts.MCTSAgent import MCTSAgent
from minihex import HexGame, player
import trueskill
import json
from Agent import Agent, RandomAgent
from multiprocessing import Pool, cpu_count
import tqdm


def play_match(arg):
    agent1, agent2, sim, board_size = arg
    env = gym.make("hex-v0", opponent_policy=agent2.act, board_size=board_size)
    state, info = env.reset()
    agent1.reset(sim)
    agent2.reset(sim)

    done = False
    while not done:
        action = agent1.act(*state, info)
        state, reward, done, info = env.step(action)

    return reward


trueskill.setup(mu=1200,
                sigma=200.0,
                beta=100.0,
                tau=2.0,
                draw_probability=0.01,
                backend="scipy")

board_size = 11
env = gym.make("hex-v0", opponent_policy=None, board_size=board_size)
env.reset()
sim = env.simulator
agent_pool = [MCTSAgent(sim, depth=d) for d in [50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]]
agent_pool += [RandomAgent(board_size)]

num_games = 5000
agent1, agent2 = np.random.choice(agent_pool, size=(2, num_games * 2))
keepers = agent1 != agent2
agent1 = (agent1[keepers])[:num_games]
agent2 = (agent2[keepers])[:num_games]
startup_environments = [sim] * num_games
board_sizes = [board_size] * num_games

with Pool(cpu_count() - 2) as workers:
    results = list(tqdm.tqdm(workers.imap(play_match, zip(agent1, agent2, startup_environments, board_sizes), chunksize=1),
                            total=num_games))
for agentA, agentB, reward in zip(agent1, agent2, results):
    if reward == -1:
        agentB.rating, agentA.rating = trueskill.rate_1vs1(agentB.rating,
                                                           agentA.rating)
    elif reward == 1:
        agentA.rating, agentB.rating = trueskill.rate_1vs1(agentA.rating,
                                                           agentB.rating)
    else:
        agentB.rating, agentA.rating = trueskill.rate_1vs1(agentB.rating,
                                                           agentA.rating,
                                                           drawn=True)

leaderboard = sorted(agent_pool,
                        key=lambda x: trueskill.expose(x.rating),
                        reverse=True)
print("Leaderboard")
print("===================")
for agent in leaderboard:
        print(f"    {agent} mu:{agent.rating.mu:.2f} sigma:{agent.rating.sigma:.2f}")

ratings = {str(agent): {"mu": agent.rating.mu, "sigma": agent.rating.sigma}
           for agent in agent_pool}

with open("mcts_eval.json", "w") as json_file:
    json.dump(ratings, json_file)

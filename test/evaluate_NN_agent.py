from anthony_net.NNAgent import NNAgent
from mcts.MCTSAgent import MCTSAgent
import gym
import minihex
from minihex import player
import numpy as np
import tqdm
import random


agent_side = player.BLACK if random.random() < 0.5 else player.WHITE
opponent_side = player.WHITE if agent_side == player.BLACK else player.BLACK

env = gym.make("hex-v0", opponent_policy=None, board_size=5, player_color=opponent_side)
state = env.reset()
hexgame = env.simulator
opponent = MCTSAgent(hexgame, depth=1000)

env = gym.make("hex-v0", opponent_policy=opponent.act, board_size=5, player_color=agent_side)
agent = NNAgent("best_model.h5")

total_wins = 0
for game in tqdm.tqdm(range(100)):
    state, info = env.reset()
    opponent.reset(hexgame)
    done = False
    while not done:
        action = agent.act(state[0], state[1], info)
        state, _, done, info = env.step(action)
    if env.simulator.winner == agent_side:
        total_wins += 1
print(f"NN Agent won {total_wins} games. Winrate: {total_wins/100:.2f}")

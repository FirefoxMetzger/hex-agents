import gym
import minihex
import numpy as np
from mcts.MCTSAgent import MCTSAgent
from minihex.HexGame import HexGame, player


def mcts_policy(state, active_player):
    # never surrender :)
    env = HexGame(active_player, state, active_player)
    agent = MCTSAgent(env)
    return agent.plan(1000)


env = gym.make("hex-v0", opponent_policy=mcts_policy, board_size=9)
state = env.reset()

done = False
while not done:
    action = mcts_policy(*state)
    state, reward, done, _ = env.step(action)

env.render()

if reward == -1:
    print("Player (Black) Lost")
elif reward == 1:
    print("Player (Black) Won")
else:
    print("Draw")

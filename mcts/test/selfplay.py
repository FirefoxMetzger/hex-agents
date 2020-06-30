import gym
import minihex
import numpy as np
from mcts.MCTSAgent import MCTSAgent


env = gym.make("hex-v0", opponent_policy=None, board_size=9)
state = env.reset()
hexgame = env.simulator

opponent = MCTSAgent(env.simulator, depth=1000)
agent = MCTSAgent(env.simulator, depth=1000)
env = gym.make("hex-v0", opponent_policy=opponent.act, board_size=9)
state = env.reset()
info = {
    'state': env.simulator.board,
    'last_move_opponent': None,
    'last_move_player': None
}

done = False
while not done:
    action = agent.act(state[0], state[1], info)
    state, reward, done, info = env.step(action)

env.render()

if reward == -1:
    print("Player (Black) Lost")
elif reward == 1:
    print("Player (Black) Won")
else:
    print("Draw")

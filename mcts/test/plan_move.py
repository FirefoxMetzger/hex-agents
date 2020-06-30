import gym
import minihex
from MCTSAgent import MCTSAgent

board_size = 5
board = np.zeros((3, board_size+4, board_size+4))
board[player.BLACK, :2, :] = 1
board[player.BLACK, -2:, :] = 1
board[player.WHITE, :, :2] = 1
board[player.WHITE, :, -2:] = 1
board[2, 2:-2, 2:-2] = 1

env = HexGame(player.BLACK, board, player.BLACK)
agent = MCTSAgent(env)
print(agent.plan())
print(agent.quality())

from Agent import Agent
import tensorflow as tf
from minihex import player, HexGame
from network import HexagonalInitializer, HexagonalConstraint, selective_loss
import gym
import numpy as np


def convert_state(state):
    board = state.board
    board_size = board.shape[1]
    converted_state = np.zeros((6, board_size+4, board_size+4))

    converted_state[[0, 2], :2, :] = 1
    converted_state[[0, 3], -2:, :] = 1
    converted_state[[1, 4], :, :2] = 1
    converted_state[[1, 5], :, -2:] = 1

    converted_state[:2, 2:-2, 2:-2] = board[:2, ...]

    region_black = np.pad(state.regions[player.BLACK], 1)
    region_white = np.pad(state.regions[player.WHITE], 1)

    positions = np.where(region_black == region_black[1, 1])
    converted_state[2][positions] = 1

    positions = np.where(region_black == region_black[-2, -2])
    converted_state[3][positions] = 1

    positions = np.where(region_white == region_white[1, 1])
    converted_state[4][positions] = 1

    positions = np.where(region_white == region_white[-2, -2])
    converted_state[5][positions] = 1

    return np.moveaxis(converted_state, 0, -1)


class NNAgent(Agent):
    def __init__(self, model_file):
        with tf.keras.utils.custom_object_scope({
                    "HexagonalInitializer": HexagonalInitializer,
                    "HexagonalConstraint": HexagonalConstraint,
                    "selective_loss": selective_loss
                }):
            self.model = tf.keras.models.load_model(model_file)

    def act(self, state, active_player, info):
        game = HexGame(active_player, state, active_player)
        state = convert_state(game)
        action_black, action_white = self.model.predict(state[np.newaxis, ...])
        if active_player == player.BLACK:
            action_values = action_black.squeeze()
        else:
            action_values = action_white.squeeze()

        available_actions = game.get_possible_actions()
        action_values = action_values[available_actions]
        print(action_values)
        greedy_action = np.argmax(action_values)
        return available_actions[greedy_action]


if __name__ == "__main__":
    agent = NNAgent('best_model.h5')
    env = gym.make("hex-v0",
                   opponent_policy=agent.act,
                   board_size=5)
    state, info = env.reset()

    done = False
    while not done:
        action = agent.act(state[0], state[1], info)
        state, reward, done, info = env.step(action)
        env.render()

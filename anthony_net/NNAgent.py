from Agent import Agent
import tensorflow as tf
from minihex import player, HexGame
from anthony_net.network import HexagonalInitializer, HexagonalConstraint
from anthony_net.network import selective_loss
import gym
import numpy as np
from anthony_net.utils import convert_state


class NNAgent(Agent):
    def __init__(self, model):
        if isinstance(model, str):
            with tf.keras.utils.custom_object_scope({
                "HexagonalInitializer": HexagonalInitializer,
                "HexagonalConstraint": HexagonalConstraint,
                "selective_loss": selective_loss
            }):
                self.model = tf.keras.models.load_model(model)
        else:
            self.model = model

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
        greedy_action = np.argmax(action_values)
        return available_actions[greedy_action]

    def predict(self, state_batch, active_player_batch):
        policy_values = self.model.predict(state_batch)
        actions = np.argmax(policy_values, axis=2)
        actions[player.WHITE, active_player_batch == player.BLACK] = -1
        actions[player.BLACK, active_player_batch == player.WHITE] = -1
        actions = actions.T
        return actions

    def predict_env(self, env):
        state = convert_state(env)
        active_player = env.active_player
        action_black, action_white = self.model.predict(state[np.newaxis, ...])
        if active_player == player.BLACK:
            return action_black.squeeze()
        else:
            return action_white.squeeze()


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

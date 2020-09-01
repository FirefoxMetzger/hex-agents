import silence_tensorflow
import tensorflow as tf
import configparser

from anthony_net.train_network import load_data
from anthony_net.NNAgent import NNAgent


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('ExIt.ini')

    data_path = config["apprenticeEval"]["prediction_data"]
    label_path = config["apprenticeEval"]["prediction_labels"]
    data, labels = load_data(data_path, label_path, config)

    iterations = int(config["ExpertIteration"]["iterations"])
    for idx in range(iterations):
        agent = NNAgent(f"logs/iteration_{idx}/model.h5")
        model = agent.model
        _, _, _, black_accuracy, white_accuracy = model.evaluate(
            data,
            labels,
            verbose=0
        )
        print(f"Iteration {idx} Accuracy -- "
              f"B: {black_accuracy:.2f}    "
              f"W: {white_accuracy:.2f}")

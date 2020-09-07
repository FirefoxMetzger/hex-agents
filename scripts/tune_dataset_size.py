import configparser
from copy import deepcopy
import json
import numpy as np
import os

from anthony_net.generate_MCTS_training_data import generate_dataset
from anthony_net.train_network import train_network, load_data


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('ExIt.ini')

    board_size = int(config["GLOBAL"]["board_size"])
    dataset_size = int(config["GLOBAL"]["dataset_size"])
    num_splits = int(config["DatasetTuning"]["num_splits"])

    log_dir = config["GLOBAL"]["log_dir"]
    out_dir = config["DatasetTuning"]["dir"]
    base_dir = "/".join([log_dir, out_dir])
    base_dir = base_dir.format(board_size=board_size)
    os.makedirs(base_dir, exist_ok=True)

    testset_size = int(config["DatasetTuning"]["testset_size"])
    test_data_file = config["DatasetTuning"]["test_data_file"]
    test_data_file = "/".join([base_dir, test_data_file])
    test_label_file = config["DatasetTuning"]["test_label_file"]
    test_label_file = "/".join([base_dir, test_label_file])

    generic_model_location = config["DatasetTuning"]["model_dir"]
    generic_model_location = "/".join([base_dir, generic_model_location])

    data_file = config["nnEval"]["training_file"]
    # data_file = "/".join([base_dir, data_file])
    data_file = data_file.format(board_size=board_size)
    label_file = config["nnEval"]["label_file"]
    # label_file = "/".join([base_dir, label_file])
    label_file = label_file.format(board_size=board_size)
    try:
        data, labels = load_data(data_file, label_file, config)
    except FileNotFoundError:
        generate_dataset(config)
        data, labels = load_data(data_file, label_file, config)

    try:
        val_data, val_labels = load_data(
            test_data_file,
            test_label_file,
            config
        )
    except FileNotFoundError:
        tmp_config = deepcopy(config)
        tmp_config["nnEval"]["training_file"] = test_data_file
        tmp_config["nnEval"]["label_file"] = test_label_file
        tmp_config["GLOBAL"]["dataset_size"] = str(testset_size)
        generate_dataset(tmp_config)
        val_data, val_labels = load_data(
            test_data_file,
            test_label_file,
            config
        )

    accuracies = list()
    test_sizes = np.linspace(dataset_size // num_splits,
                             dataset_size, num_splits, dtype=np.intp)
    train_config = deepcopy(config)
    for split_size in test_sizes:
        model_location = generic_model_location.format(
            split_size=split_size
        )
        os.makedirs(model_location, exist_ok=True)

        train_config["Training"]["model_file"] = "/".join([
            model_location,
            "model.h5"
        ])
        train_config["Training"]["history_file"] = "/".join([
            model_location,
            "history.pickle"
        ])

        data_split = data[:split_size]
        label_split = (labels[0][:split_size], labels[1][:split_size])

        model = train_network(data_split, label_split, train_config)

        _, _, _, black_acc, white_acc = model.evaluate(val_data, val_labels)
        accuracies.append((black_acc + white_acc) / 2)

    result = {
        "acc": accuracies,
        "dataset_size": test_sizes.tolist()
    }

    out_file = config["DatasetTuning"]["result_file"]
    out_file = "/".join([base_dir, out_file])
    with open(out_file.format(board_size=board_size), "w") as json_file:
        json.dump(result, json_file)

import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from minihex import HexGame

from anthony_net.network import gen_model
from anthony_net.network import selective_loss, selective_CategoricalAccuracy
from anthony_net.utils import convert_state_batch


def load_data(data_file, label_file, config, workers=None, max_size=None):
    boards = np.load(data_file)['arr_0']
    players = np.load(data_file)['arr_1']
    labels = np.load(label_file)['arr_0']

    chunksize = int(config["GLOBAL"]["chunksize"])
    if workers:
        sim_args = zip(players, boards, players)
        data = [sim for sim in workers.starmap(
            HexGame,
            sim_args,
            chunksize=chunksize)]
    else:
        data = [sim for sim in map(
            HexGame,
            players,
            boards,
            players)]
    data = np.stack(convert_state_batch(data))

    board_size = int(config["GLOBAL"]["board_size"])
    labels = tf.one_hot(labels, board_size ** 2)

    if max_size is None:
        return data, (labels[:, 0, :], labels[:, 1, :])
    else:
        return (data[:max_size, ...],
                (labels[:max_size, 0, :], labels[:max_size, 1, :]))


def train_network(data, labels, config):
    board_size = int(config["GLOBAL"]["board_size"])
    network = gen_model(board_size)
    network.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=[selective_loss, selective_loss],
        metrics=[
            [selective_CategoricalAccuracy],
            [selective_CategoricalAccuracy]
        ]
    )

    dataset_size = data.shape[0]
    train_idx, val_idx, _, _ = train_test_split(
        np.arange(dataset_size, dtype=np.intp),
        np.ones(dataset_size, dtype=np.intp),
        test_size=float(config["Training"]["validation_split"]))

    train_data = data[train_idx, ...]
    train_lables = (tf.gather(labels[0], train_idx),
                    tf.gather(labels[1], train_idx))

    val_data = data[val_idx, ...]
    val_labels = (tf.gather(labels[0], val_idx),
                  tf.gather(labels[1], val_idx))

    history = network.fit(
        train_data, train_lables,
        validation_data=(val_data, val_labels),
        epochs=int(config["Training"]["max_epochs"]),
        batch_size=int(config["Training"]["batch_size"]),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=int(config["Training"]["patience"]),
            ),
            tf.keras.callbacks.ModelCheckpoint(
                config["Training"]["model_file"],
                monitor='val_loss',
                save_only_best=True
            )
        ])

    filename = config["Training"]["history_file"]
    with open(filename, "wb") as out_file:
        pickle.dump(history.history, out_file)

    return network


if __name__ == "__main__":
    import configparser
    config = configparser.ConfigParser()
    config.read('ExIt.ini')

    data_file = config["nnEval"]["training_file"]
    label_file = config["nnEval"]["label_file"]
    data, labels = load_data(data_file, label_file, config)

    train_network(data, labels, config)

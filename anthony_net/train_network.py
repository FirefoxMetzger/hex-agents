import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .network import gen_model
from .network import selective_loss, selective_CategoricalAccuracy


def load_data(prefix=None, max_size=None, postfix=None):
    if prefix is None:
        prefix = ""
    else:
        prefix += "_"

    if postfix is None:
        postfix = ""
    else:
        postfix = "_" + postfix

    with open(f"{prefix}data{postfix}.npy", "rb") as in_file:
        data = np.load(in_file)
    with open(f"{prefix}labels{postfix}.npy", "rb") as in_file:
        labels = np.load(in_file)

    board_size = data.shape[2] - 4
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

    data, labels = load_data("training")

    train_network(data, labels, config)

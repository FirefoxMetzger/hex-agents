import numpy as np
import tensorflow as tf
from network import gen_model, selective_loss, selective_CategoricalAccuracy


def load_data(prefix=None):
    if prefix is None:
        prefix = ""
    else:
        prefix += "_"

    with open(f"{prefix}data.npy", "rb") as in_file:
        data = np.load(in_file)
    with open(f"{prefix}labels.npy", "rb") as in_file:
        labels = np.load(in_file)

    board_size = data.shape[2] - 4
    labels = tf.one_hot(labels, board_size ** 2)

    return data, (labels[:, 0, :], labels[:, 1, :])


training_data, training_labels = load_data("training")
validation_data, validation_labels = load_data("validation")
test_data, test_labels = load_data("test")

network = gen_model(5)
network.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=[selective_loss, selective_loss],
    metrics=[[selective_CategoricalAccuracy], [selective_CategoricalAccuracy]]
)
network.fit(training_data, training_labels,
            epochs=100,
            batch_size=256,
            validation_data=(validation_data, validation_labels),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    monitor='val_loss',
                    save_only_best=True)

            ])

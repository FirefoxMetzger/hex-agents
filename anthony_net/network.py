import tensorflow as tf
import numpy as np


class HexagonalConstraint(tf.keras.constraints.Constraint):
    """Constrain convolution to hexagonal connections"""

    def __init__(self, *args, **kwargs):
        super(HexagonalConstraint, self).__init__(*args, **kwargs)
        mask = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]])
        mask = mask[..., np.newaxis, np.newaxis]
        self.tf_mask = tf.constant(mask, dtype=tf.float32)

    def __call__(self, weights):
        return self.tf_mask * weights


class HexagonalInitializer(tf.keras.initializers.GlorotUniform):
    def __call__(self, shape, dtype=tf.dtypes.float32):
        weights = super(HexagonalInitializer, self).__call__(shape, dtype)
        weights = weights.numpy()
        weights[0, 0, ...] = 0
        weights[-1, -1, ...] = 0
        return weights

    def get_config(self):
        return super(HexagonalInitializer, self).get_config()


def selective_loss(y_true, y_pred):
    factor = tf.reduce_sum(y_true, axis=-1)
    return factor * tf.keras.losses.KLD(y_true, y_pred)


def selective_CategoricalAccuracy(y_true, y_pred):
    factor = tf.reduce_sum(y_true, axis=-1, keepdims=True)
    return tf.keras.metrics.categorical_accuracy(y_true, factor * y_pred)


def gen_model(board_size, policy_head=True, value_head=False):
    feature_decomposition = tf.keras.Sequential()
    feature_decomposition.add(tf.keras.layers.Input(shape=(board_size+4,
                                                           board_size+4,
                                                           6)))
    for layer in range(8):
        feature_decomposition.add(tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            activation=tf.keras.activations.elu,
            kernel_constraint=HexagonalConstraint(),
            kernel_initializer=HexagonalInitializer()))

    feature_decomposition.add(tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation=tf.keras.activations.elu,
        kernel_constraint=HexagonalConstraint(),
        kernel_initializer=HexagonalInitializer()))
    feature_decomposition.add(tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation=tf.keras.activations.elu,
        kernel_constraint=HexagonalConstraint(),
        kernel_initializer=HexagonalInitializer()))
    feature_decomposition.add(tf.keras.layers.Conv2D(
        64,
        (1, 1),
        activation=tf.keras.activations.elu))
    feature_decomposition.add(tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation=tf.keras.activations.elu,
        kernel_constraint=HexagonalConstraint(),
        kernel_initializer=HexagonalInitializer()))
    feature_decomposition.add(tf.keras.layers.Conv2D(
        64,
        (1, 1),
        activation=tf.keras.activations.elu))

    head_input = feature_decomposition.output
    head_input = tf.keras.layers.Flatten()(head_input)
    if policy_head:
        policy_head_black = tf.keras.layers.Dense(
            board_size ** 2,
            activation=tf.keras.activations.softmax,
            use_bias=False)(head_input)
        policy_head_white = tf.keras.layers.Dense(
            board_size ** 2,
            activation=tf.keras.activations.softmax,
            use_bias=False)(head_input)

    if value_head:
        value_head_black = tf.keras.layers.Dense(
            1,
            activation=tf.keras.activations.sigmoid,
            use_bias=False)(head_input)
        value_head_white = tf.keras.layers.Dense(
            1,
            activation=tf.keras.activations.sigmoid,
            use_bias=False)(head_input)

    if policy_head and value_head:
        output = [policy_head_black,
                  policy_head_white,
                  value_head_black,
                  value_head_white]
    elif policy_head:
        output = [policy_head_black,
                  policy_head_white]
    elif value_head:
        output = [value_head_black,
                  value_head_white]
    else:
        raise Exception("Must specify either policy or value head")

    network = tf.keras.Model(inputs=feature_decomposition.input,
                             outputs=output)

    return network

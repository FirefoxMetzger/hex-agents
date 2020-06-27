import numpy as np


def load_data(prefix=None):
    if prefix is None:
        prefix = ""
    else:
        prefix += "_"

    with open(f"{prefix}data.npy", "rb") as in_file:
        data = np.load(in_file)
    with open(f"{prefix}labels.npy", "rb") as in_file:
        labels = np.load(in_file)

    return data, labels


for kind in ["training", "validation", "test"]:
    data, labels = load_data(kind)

    for (example, label) in zip(data, labels):
        board = example[1, ...]
        pos = np.unravel_index(label, board.shape)
        assert np.all(example[:2, pos[0], pos[1]]) == 0

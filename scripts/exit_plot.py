import matplotlib.pyplot as plt
import json
import numpy as np


def get_mcts_scores(file_name):
    with open(file_name, "r") as result_file:
        result = json.load(result_file)

    x = list()
    y = list()
    ci = list()
    for key in result.keys():
        if key == "Random":
            x.append(0)
        else:
            x.append(int(key[5:-1]))
        y.append(result[key]['mu'])
        ci.append(result[key]['sigma'])

    idx = np.argsort(x)
    x = np.array(x)[idx]
    y = np.array(y)[idx]
    ci = np.array(ci)[idx]

    return x, y, ci


def get_exit_scores(file_name):
    with open(file_name, "r") as result_file:
        result = json.load(result_file)

    x = list()
    y = list()
    ci = list()
    for key in result.keys():
        if key == "0":
            pass
        else:
            x.append(int(key) - 1)
        y.append(result[key]['mu'])
        ci.append(result[key]['sigma'])

    idx = np.argsort(x)
    x = np.array(x)[idx]
    y = np.array(y)[idx]
    ci = np.array(ci)[idx]

    return x, y, ci


fig, ax = plt.subplots()
mcts_x, mcts_y, mcts_ci = get_mcts_scores("mcts_eval(5x5)big.json")
x, y, ci = get_exit_scores("exit_eval(5x5).json")

for depth, rating in zip(mcts_x, mcts_y):
    ax.plot(x, rating * np.ones_like(x), linestyle="--", color="grey")
    ax.text(x[-1], rating-25, f"MCTS({depth})", verticalalignment="center", wrap=True, ha="right", color="grey")

ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.15)
ax.plot(x, y, marker='.')


# ax.legend(["5x5 board", "9x9 board", "11x11 board"])
ax.set_xlabel("ExIt Iteration")
ax.set_ylabel("TrueSkill Rating")
plt.show()

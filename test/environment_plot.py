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

    # RandomAgent has no skill
    y = y - y[0]

    return x, y, ci


fig, ax = plt.subplots()

x, y, ci = get_mcts_scores("mcts_eval(5x5).json")
x2, y2, ci2 = get_mcts_scores("mcts_eval(9x9).json")
x3, y3, ci3 = get_mcts_scores("mcts_eval(11x11)big.json")

ax.plot(x, y, marker='.')
ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)

ax.plot(x2, y2, marker='.')
ax.fill_between(x2, (y2-ci2), (y2+ci2), color='r', alpha=.1)

ax.plot(x3, y3, marker='.')
ax.fill_between(x3, (y3-ci3), (y3+ci3), color='g', alpha=.1)

ax.legend(["5x5 board", "9x9 board", "11x11 board"])
ax.set_xlabel("MCTS Depth (Number of Simulations)")
ax.set_ylabel("TrueSkill Rating")
plt.show()

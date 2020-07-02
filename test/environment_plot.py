import matplotlib.pyplot as plt
import json
import numpy as np


def get_mcts_scores(file_name):
    with open(file_name, "r") as result_file:
        result_5x5 = json.load(result_file)

    x = list()
    y = list()
    ci = list()
    for key in result_5x5.keys():
        if key == "Random":
            x.append(0)
        else:
            x.append(int(key[5:-1]))
        y.append(result_5x5[key]['mu'])
        ci.append(result_5x5[key]['sigma'])

    idx = np.argsort(x)
    x = np.array(x)[idx]
    y = np.array(y)[idx]
    ci = np.array(ci)[idx]

    return x, y, ci


fig, ax = plt.subplots()

x, y, ci = get_mcts_scores("mcts_eval(5x5).json")
x2, y2, ci2 = get_mcts_scores("mcts_eval(9x9).json")

ax.plot(x, y, marker='.')
ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)

ax.plot(x2, y2, marker='.')
ax.fill_between(x2, (y2-ci2), (y2+ci2), color='r', alpha=.1)

plt.show()

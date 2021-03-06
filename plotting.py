import matplotlib.pyplot as plt
import numpy as np

from minihex import player


def plot_expansions(data, out_file, config):
    board_size = int(config["GLOBAL"]["board_size"])

    fig, ax = plt.subplots()

    for color in [player.BLACK, player.WHITE]:
        mu = np.array(data[color]["mu"])
        sigma = np.array(data[color]["sigma"])
        depth = np.array(data[color]["depth"])

        ax.fill_between(
            depth, (mu - sigma), (mu + sigma),
            alpha=.15
        )
        ax.plot(depth, mu, marker='.')

    ax.set_title(f"{board_size}x{board_size} Board")
    ax.set_xlabel("Number of Expansions")
    ax.set_ylabel("Trueskill Score")

    ax.legend(["Black", "White"], title="Agent Color")

    plt.savefig(out_file)


def plot_iteration(data, out_file, config):
    board_size = int(config["GLOBAL"]["board_size"])
    width = float(config["Plotting"]["bar_width"])

    fig, ax = plt.subplots()

    for color in [player.BLACK, player.WHITE]:
        bar_width = -width if color == player.BLACK else width
        mu = np.array(data[color]["mu"])
        sigma = np.array(data[color]["sigma"])
        idx = np.arange(len(mu), dtype=np.float_) + bar_width / 2

        ax.bar(idx, mu, width, yerr=np.stack((sigma, sigma)))

    it = data[color]["iterations"]

    ax.set_xticks(it)
    ax.set_xticklabels(it)

    ax.set_title(f"{board_size}x{board_size} Board")
    ax.set_xlabel("Iteration/Step")
    ax.set_ylabel("Trueskill Score")

    ax.legend(["Black", "White"], title="Agent Color")

    plt.savefig(out_file)

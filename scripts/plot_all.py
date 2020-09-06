import matplotlib.pyplot as plt
import json
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read('ExIt.ini')
board_size = int(config["GLOBAL"]["board_size"])


try:
    score_file = config["apprenticeEval"]["eval_file"].format(
        board_size=board_size
    )
    with open(score_file, "r") as file:
        scores = json.load(file)

    mu = np.array(scores["mu"])
    sigma = np.array(scores["sigma"])

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(mu)), mu, yerr=np.stack((sigma, sigma)))

    ax.set_title("Apprentice Score")
    ax.set_xlabel("ExIt Iteration")
    ax.set_ylabel("Trueskill Score")

    fig_path = config["apprenticeEval"]["plot_file"]
    plt.savefig(fig_path)
except FileNotFoundError:
    print("Skipping: Apprentice Score Plot")


try:
    score_file = config["expertEval"]["depth_eval_file"].format(
        board_size=board_size
    )
    with open(score_file, "r") as file:
        scores = json.load(file)

    mu = np.array(scores["mu"])
    sigma = np.array(scores["sigma"])
    depth = np.array(scores["depth"])

    fig, ax = plt.subplots()
    ax.fill_between(
        depth, (mu - sigma), (mu + sigma),
        color='b',
        alpha=.15
    )
    ax.plot(depth, mu, marker='.')

    ax.set_title("Expert Score")
    ax.set_xlabel("Number of Expansions")
    ax.set_ylabel("Trueskill Score")

    fig_path = config["expertEval"]["depth_plot_file"]
    plt.savefig(fig_path)

except FileNotFoundError:
    print("Skipping: Expert Depth Plot")

try:
    score_file = config["expertEval"]["iter_eval_file"].format(
        board_size=board_size
    )
    with open(score_file, "r") as file:
        scores = json.load(file)

    mu = np.array(scores["mu"])
    sigma = np.array(scores["sigma"])

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(mu)), mu, yerr=np.stack((sigma, sigma)))

    ax.set_title("Expert Score")
    ax.set_xlabel("ExIt Iteration")
    ax.set_ylabel("Trueskill Score")

    fig_path = config["expertEval"]["iter_plot_file"]
    plt.savefig(fig_path)

except FileNotFoundError:
    print("Skipping: Expert Iteration Plot")

try:
    score_file = config["mctsEval"]["eval_file"].format(
        board_size=board_size
    )
    with open(score_file, "r") as file:
        scores = json.load(file)

    mu = np.array(scores["mu"])
    sigma = np.array(scores["sigma"])
    depth = np.array(scores["depth"])

    fig, ax = plt.subplots()
    ax.fill_between(
        depth, (mu - sigma), (mu + sigma),
        color='b',
        alpha=.15
    )
    ax.plot(depth, mu, marker='.')

    ax.set_title("MCTS Score")
    ax.set_xlabel("Number of Expansions")
    ax.set_ylabel("Trueskill Score")

    fig_path = config["mctsEval"]["plot_file"]
    plt.savefig(fig_path)

except FileNotFoundError:
    print("Skipping: MCTS Plot")

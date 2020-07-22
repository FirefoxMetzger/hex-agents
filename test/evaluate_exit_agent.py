from Agent import RandomAgent
from anthony_net.NNAgent import NNAgent
from mcts.MCTSAgent import MCTSAgent
import gym
import minihex
from minihex import player
from minihex.HexGame import HexEnv, HexGame
import numpy as np
import tqdm
import random
from nmcts.NMCTSAgent import NMCTSAgent
import trueskill
import json
from utils import step_and_rollout
from anthony_net.utils import convert_state_batch
from nmcts.NeuralSearchNode import NeuralSearchNode
from multiprocessing import Pool, cpu_count


def play_match(mcts_depth, nmcts_depth, model_file, board_size):
    player_color = player.BLACK if random.random() > 0.5 else player.WHITE

    if mcts_depth == 0:
        opponent = RandomAgent(board_size)
    else:
        opponent = MCTSAgent(depth=mcts_depth, board_size=board_size)
    agent = NMCTSAgent(model_file=model_file,
                       depth=nmcts_depth, board_size=board_size)

    env = gym.make("hex-v0", player_color=player_color,
                   opponent_policy=opponent.act, board_size=board_size)
    state, info = env.reset()
    yield ("update_env", env.simulator)

    done = False
    while not done:
        yield from agent.update_root_state_deferred(info)
        yield from agent.deferred_plan()
        action = agent.act_greedy(state[0], state[1], None)

        # opponent_info = info.copy()
        # opponent_info["last_move"] = opponent_info["last_move_opponent"]
        # opponent_info["last_move_opponent"] = action
        # yield from opponent.update_root_state_deferred(opponent_info)
        # yield from opponent.deferred_plan()

        state, reward, done, info = env.step(action)
        yield ("update_env", env.simulator)

    yield ("done", reward)


def evaluate_agent(num_matches, workers):
    board_size = 5
    nn_agent = NNAgent("best_model.h5")
    eval_agent = NMCTSAgent(agent=nn_agent,
                            depth=30, board_size=board_size)

    with open(f"mcts_eval({board_size}x{board_size})big.json", "r") as rating_file:
        mcts_ratings = json.load(rating_file)

    depths = [
        0, 50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    matches = [depths[random.randint(0, len(depths)-1)]
               for _ in range(num_matches)]

    tasks = [(idx, "init", play_match(depth, 30, "best_model.h5", board_size), None)
             for idx, depth in enumerate(matches)]
    initial_sims = dict()

    pbar = tqdm.tqdm(total=num_matches, desc="Games Played")
    while tasks:
        # scheduler
        new_tasks = list()
        nmcts_expand = list()
        mcts_expand = list()
        for task in tasks:
            idx, job, gen, args = task

            if job == "init":
                job, args = gen.send(None)
                new_tasks.append((idx, job, gen, args))
            elif job == "update_env":
                new_env = args
                initial_sims[idx] = new_env
                job, args = gen.send(None)
                new_tasks.append((idx, job, gen, args))
            elif job == "expand_and_simulate":
                nmcts_expand.append((idx, job, gen, args))
            elif job == "mcts_expand_and_simulate":
                nmcts_expand.append((idx, job, gen, args))
            elif job == "done":
                result = args
                depth = matches[idx]

                if depth == 0:
                    opponent = RandomAgent(board_size)
                else:
                    opponent = MCTSAgent(depth=depth, board_size=board_size)

                rating = mcts_ratings[str(opponent)]
                op_rating = trueskill.Rating(rating["mu"], rating["sigma"])
                if result == -1:
                    _, eval_agent.rating = trueskill.rate_1vs1(
                        op_rating, eval_agent.rating)
                elif result == 1:
                    eval_agent.rating, _ = trueskill.rate_1vs1(
                        eval_agent.rating, op_rating)
                else:
                    eval_agent.rating, _ = trueskill.rate_1vs1(
                        eval_agent.rating, op_rating, drawn=True)
                pbar.update(1)

        # handle nmcts expansions
        if nmcts_expand:
            sim_batch = list()
            history_batch = list()
            for task in nmcts_expand:
                idx, _, _, action_history = task
                sim_batch.append(initial_sims[idx])
                history_batch.append(action_history)
            results = workers.starmap(
                step_and_rollout, zip(sim_batch, history_batch))
            envs = list()
            players = list()
            winners = list()
            for env, winner in results:
                envs.append(env)
                players.append(env.active_player)
                winners.append(winner)
            board_batch = np.stack(convert_state_batch(envs))
            players = np.stack(players)
            policies = nn_agent.get_scores(board_batch, players)

            result_iter = zip(nmcts_expand, policies, winners, envs)
            for task, policy, winner, env in result_iter:
                idx, _, gen, action_history = task
                node = NeuralSearchNode(env, network_policy=policy)
                job, args = gen.send((node, winner))
                new_tasks.append((idx, job, gen, args))

        # handle mcts expansions
        if mcts_expand:
            sim_batch = list()
            history_batch = list()
            for task in mcts_expand:
                idx, _, _, action_history = task
                sim_batch.append(initial_sims[idx])
                history_batch.append(action_history)
            results = workers.starmap(
                step_and_rollout, zip(sim_batch, history_batch))
            winners = list()
            for env, winner in results:
                winners.append(winner)

            result_iter = zip(mcts_expand, winners)
            for task, winner, env in result_iter:
                idx, _, gen, action_history = task
                node = SearchNode(env)
                job, args = gen.send((node, winner))
                new_tasks.append((idx, job, gen, args))

        tasks = new_tasks

    return eval_agent


if __name__ == "__main__":
    trueskill.setup(mu=1200,
                    sigma=200.0,
                    beta=100.0,
                    tau=2.0,
                    draw_probability=0.01,
                    backend="scipy")

    num_matches = 2

    with Pool(3) as workers:
        eval_agent = evaluate_agent(num_matches, workers)
        print(eval_agent.rating)

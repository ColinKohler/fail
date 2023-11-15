import os
import sys

sys.path.insert(0, os.path.abspath("."))

import copy
import numpy as np
import numpy.random as npr
from tqdm import tqdm
import pickle
import argparse

from pydrake.geometry import StartMeshcat

from peg_insertion_envs.robot_insertion_3d_env import createPegInsertionEnv
from dataset.replay_buffer import ReplayBuffer
from expert.peg_insertion_3d_policy import getAction


def generateExpertData(num_eps, filepath, render=False):
    meshcat = StartMeshcat() if render else None
    env = createPegInsertionEnv(meshcat=meshcat)
    buffer = ReplayBuffer()

    pbar = tqdm(total=num_eps)
    pbar.set_description("0/0 (0%)")
    sr = 0

    eps = 0
    while eps < num_eps:
        goal, obs = env.reset()
        terminate = False
        offset = npr.uniform([-0.010, -0.010], [0.010, 0.010])
        stage = 0

        episode_data = {
            "state": [obs],
            "goal": [goal],
            "action": [[0, 0, 0]],
            "reward": [0],
        }

        while not terminate:
            stage, expert_action = getAction(
                obs[:, :3], obs[:, 3:], goal, env.action_space, offset, stage
            )
            obs_, reward, terminate, timeout = env.step(expert_action)

            episode_data["state"].append(obs_)
            episode_data["goal"].append(goal)
            episode_data["action"].append(expert_action)
            episode_data["reward"].append(reward)
            obs = obs_

        # Don't save the episode if it failed
        if reward == 0:
            continue

        for key, value in episode_data.items():
            episode_data[key] = np.array(value)
        buffer.addEpisode(episode_data)

        eps += 1
        sr += reward
        pbar.set_description("{}/{} ({:.2f}%)".format(sr, eps, (sr / eps) * 100))
        pbar.update(1)
    pbar.close()

    with open("{}.pkl".format(filepath), "wb") as fh:
        pickle.dump(buffer.getSaveState(), fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_eps", type=int, help="Number of episodes to generate")
    parser.add_argument(
        "filepath", type=str, help="Filepath to save the expert data to."
    )
    parser.add_argument(
        "--render",
        default=False,
        action="store_true",
        help="Render the data generation.",
    )
    args = parser.parse_args()

    generateExpertData(args.num_eps, args.filepath, render=args.render)

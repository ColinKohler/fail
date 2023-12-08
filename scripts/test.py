import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

import copy
import time
import hydra
import numpy as np
import numpy.random as npr
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import dill
from skimage.transform import resize
import argparse
import collections

from pydrake.geometry import StartMeshcat

#from drake_ws.peg_insertion_envs.robot_insertion_3d_env import createPegInsertionEnv
from drake_ws.envs.block_pushing_env import createBlockPushingEnv
from drake_ws.envs.block_touching_env import createBlockTouchingEnv
from fail.workflow.base_workflow import BaseWorkflow
from fail.utils import torch_utils


def test(checkpoint, num_eps=100, render=False):
    meshcat = StartMeshcat() if render else None
    #env = createPegInsertionEnv(meshcat=meshcat)
    env = createBlockTouchingEnv(meshcat=meshcat)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    config = payload["config"]
    cls = hydra.utils.get_class(config._target_)

    workflow = cls(config)
    workflow: BaseWorkflow
    workflow.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workflow.model
    policy.to(device)
    policy.eval()

    pbar = tqdm(total=num_eps)
    pbar.set_description("0/0 (0%)")
    sr = 0

    for eps in range(num_eps):
        goal, obs = env.reset()
        obs_deque = collections.deque([obs] * config.obs_horizon, maxlen=config.obs_horizon)
        terminate = False
        while not terminate:
            action = policy.get_action(obs_deque, goal, device)
            obs_, reward, terminate, timeout = env.step(action)

            obs = obs_
            obs_deque.append(obs)
        sr += reward

        pbar.set_description(
            "{}/{} ({:.2f}%)".format(sr, eps + 1, (sr / (eps + 1)) * 100)
        )
        pbar.update(1)
    pbar.close()

    print("{}/{} ({:.2f}%)".format(int(sr), num_eps, (sr / num_eps) * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Filepath to save the model to.")
    parser.add_argument(
        "--num_eps",
        type=int,
        default=100,
        help="Number of episodes to test the policy on.",
    )
    parser.add_argument(
        "--render",
        default=False,
        action="store_true",
        help="Render the env using meshcat.",
    )

    args = parser.parse_args()
    test(args.model_path, args.num_eps, render=args.render)

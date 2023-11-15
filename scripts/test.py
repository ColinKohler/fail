import os
import sys

sys.path.insert(0, os.path.abspath("."))

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
from utils import torch_utils
import collections

from pydrake.geometry import StartMeshcat

from drake_ws.peg_insertion_envs.robot_insertion_3d_env import createPegInsertionEnv
from fail.workflows.base_workflow import BaseWorkflow


def test(checkpoint, num_eps=100, render=False):
    meshcat = StartMeshcat() if render else None
    env = createPegInsertionEnv(meshcat=meshcat)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    config = payload["config"]
    cls = hydra.utils.get_class(config._target_)

    workflow = cls(config)
    workflow: BaseWorkflow
    workflow.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workflow.model
    normalizer = policy.normalizer
    policy.to(device)
    policy.eval()


    pbar = tqdm(total=num_eps)
    pbar.set_description("0/0 (0%)")
    sr = 0

    for eps in range(num_eps):
        goal, obs = env.reset()
        ngoal = normalizer["goal"].normalize(goal)
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
        terminate = False
        # hole_noise = npr.uniform([-0.010, -0.010, 0.0], [0.010, 0.010, 0])
        hole_noise = 0
        # print(hole_noise)
        while not terminate:
            obs_seq = np.stack(obs_deque)
            nobs = normalizer["obs"].normalize(obs_seq)
            policy_obs = nobs.unsqueeze(0).flatten(1, 2)
            # policy_obs = torch.concat((ngoal.view(1,1,3).repeat(1,20,1), policy_obs), dim=-1)
            policy_obs[:, :, :3] = ngoal.view(1, 1, 3).repeat(1, seq_len, 1) - (
                policy_obs[:, :, :3] + hole_noise
            )
            policy_obs = policy_obs.to(device)

            with torch.no_grad():
                _, _, policy_action = policy.sample(policy_obs)
                policy_action = policy_action.cpu().squeeze().numpy()
                # attn_maps = policy.encoder.transformer.getAttnMaps(policy_obs)
                # torch_utils.plotAttnMaps(torch.arange(seq_len).view(1,seq_len), attn_maps)
            action = normalizer["action"].unnormalize(policy_action)
            # print('force action: {}'.format(np.round(action, 3)))
            # f = obs[:,3:]
            # plt.plot(f[:,0], label='Mx')
            # plt.plot(f[:,1], label='My')
            # plt.plot(f[:,2], label='Mz')
            # plt.plot(f[:,3], label='Fx')
            # plt.plot(f[:,4], label='Fy')
            # plt.plot(f[:,5], label='Fz')
            # plt.legend()
            # plt.show()
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

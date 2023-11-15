import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import copy
import time
import numpy as np
import numpy.random as npr
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from skimage.transform import resize
import argparse
from utils import torch_utils
import collections

from pydrake.geometry import StartMeshcat

from peg_insertion_envs.robot_insertion_3d_env import createPegInsertionEnv
from models.policy import StochasticPolicy

def test(num_eps, model_path, obs_horizon=2, obs_key_points=10, render=False):
  meshcat = StartMeshcat() if render else None
  env = createPegInsertionEnv(meshcat=meshcat)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  seq_len = obs_horizon * obs_key_points
  model_save_state = pickle.load(open(model_path, 'rb'))
  policy = StochasticPolicy(n_act=3, seq_len=seq_len)
  policy.load_state_dict(model_save_state['weights'])
  policy = policy.to(device)
  policy.eval()
  normalizer = model_save_state['normalizer']

  pbar = tqdm(total=num_eps)
  pbar.set_description('0/0 (0%)')
  sr = 0

  for eps in range(num_eps):
    goal, obs = env.reset()
    ngoal = normalizer['goal'].normalize(goal)
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    terminate = False
    #hole_noise = npr.uniform([-0.010, -0.010, 0.0], [0.010, 0.010, 0])
    hole_noise =  0
    #print(hole_noise)
    while not terminate:
      obs_seq = np.stack(obs_deque)
      nobs = normalizer['obs'].normalize(obs_seq)
      policy_obs = nobs.unsqueeze(0).flatten(1,2)
      #policy_obs = torch.concat((ngoal.view(1,1,3).repeat(1,20,1), policy_obs), dim=-1)
      policy_obs[:,:,:3] = ngoal.view(1,1,3).repeat(1,seq_len,1) - (policy_obs[:,:,:3] + hole_noise)
      policy_obs = policy_obs.to(device)

      with torch.no_grad():
        _, _, policy_action = policy.sample(policy_obs)
        policy_action = policy_action.cpu().squeeze().numpy()
        #attn_maps = policy.encoder.transformer.getAttnMaps(policy_obs)
        #torch_utils.plotAttnMaps(torch.arange(seq_len).view(1,seq_len), attn_maps)
      action = normalizer['action'].unnormalize(policy_action)
      #print('force action: {}'.format(np.round(action, 3)))
      #f = obs[:,3:]
      #plt.plot(f[:,0], label='Mx')
      #plt.plot(f[:,1], label='My')
      #plt.plot(f[:,2], label='Mz')
      #plt.plot(f[:,3], label='Fx')
      #plt.plot(f[:,4], label='Fy')
      #plt.plot(f[:,5], label='Fz')
      #plt.legend()
      #plt.show()
      obs_, reward, terminate, timeout = env.step(action)

      obs = obs_
      obs_deque.append(obs)
    sr += reward

    pbar.set_description('{}/{} ({:.2f}%)'.format(sr, eps+1, (sr / (eps+1)) * 100))
    pbar.update(1)
  pbar.close()

  print('{}/{} ({:.2f}%)'.format(int(sr), num_eps, (sr / num_eps) * 100))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'model_path',
    type=str,
    help='Filepath to save the model to.'
  )
  parser.add_argument(
    '--num_eps',
    type=int,
    default=100,
    help='Number of episodes to test the policy on.'
  )
  parser.add_argument(
    '--render',
    default=False,
    action='store_true',
    help='Render the env using meshcat.'
  )
  parser.add_argument(
    '--obs_horizon',
    type=int,
    default=2,
    help='Number of timsteps to include in an observation.'
  )
  parser.add_argument(
    '--obs_key_points',
    type=int,
    default=10,
    help='Number of points in each action trajectory.'
  )

  args = parser.parse_args()
  test(args.num_eps, args.model_path, obs_horizon=args.obs_horizon, obs_key_points=args.obs_key_points, render=args.render)

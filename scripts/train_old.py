import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import copy
import time
import numpy as np
import numpy.random as npr
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from skimage.transform import resize
import argparse

from pydrake.geometry import StartMeshcat

from dataset.base_dataset import BaseDataset
from models.policy import StochasticPolicy
from utils import torch_utils

def train(epochs, data_path, model_path, lr=1e-3, batch_size=64, obs_horizon=2, obs_key_points=10):
  dataset = BaseDataset(data_path, horizon=obs_horizon)
  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=1,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True
  )

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  seq_len = obs_horizon * obs_key_points
  policy = StochasticPolicy(n_act=3, seq_len=seq_len).to(device)
  optimizer = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-3)
  lr_scheduler = torch_utils.CosineWarmupScheduler(optimizer, 500, len(dataloader) * epochs)
  criterion = nn.MSELoss()

  policy.train()
  with tqdm(range(epochs), desc='Epoch') as tglobal:
    for epoch_idx in tglobal:
      epoch_loss = list()
      with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
        for nbatch in tepoch:
          nobs, naction = nbatch['obs'].float().to(device), nbatch['action'].float().to(device)
          ngoal = nbatch['goal'].float().to(device)
          B = nobs.shape[0]
          obs = nobs.flatten(1,2)
          #obs = torch.concat((ngoal[:,0,:].unsqueeze(1).repeat(1,20,1), obs), dim=-1)
          obs[:,:,:3] = ngoal[:,0,:].unsqueeze(1).repeat(1,seq_len,1) - obs[:,:,:3]

          mean, log_prob, pred_action = policy.sample(obs)
          loss = criterion(mean, naction[:,-1])

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          lr_scheduler.step()

          loss_cpu = loss.item()
          epoch_loss.append(loss_cpu)
          tepoch.set_postfix(loss=loss_cpu)
      tglobal.set_postfix(loss=np.mean(epoch_loss))

  save_state = {
    'weights' : policy.cpu().state_dict(),
    'normalizer' : dataset.normalizer
  }
  with open('{}.pkl'.format(model_path), 'wb') as fh:
    pickle.dump(save_state, fh, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'epochs',
    type=int,
    help='Number of epochs to train the policy.'
  )
  parser.add_argument(
    'data_path',
    type=str,
    help='Filepath to the expert data.'
  )
  parser.add_argument(
    'model_path',
    type=str,
    help='Filepath to save the model to.'
  )
  parser.add_argument(
    '--lr',
    type=float,
    default=1e-3,
    help='Optimizer learning rate.'
  )
  parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='Minibatch size.'
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
  train(
    args.epochs,
    args.data_path,
    args.model_path,
    lr=args.lr,
    batch_size=args.batch_size,
    obs_horizon=args.obs_horizon,
    obs_key_points=args.obs_key_points,
  )

import numpy as np
import torch
import matplotlib.pyplot as plt

def circular_harmonics(L, theta):
    B = [torch.tensor([1 / np.sqrt(2 * np.pi)] * theta.size(0), device=theta.device).view(-1, 1)]
    for l in range(1, L+1):
        B.append(l * torch.cos(theta) / np.sqrt(np.pi))
        B.append(l * torch.sin(theta) / np.sqrt(np.pi))

    return torch.stack(B).permute(1,0,2).float()

def spherical_harmonics(L, theta):
    pass

def plot_energy_circle(E):
  fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
  ax.plot(np.linspace(0, 2*np.pi, 36), E)
  ax.set_rmax(np.max(E) + 0.2)
  ax.set_rticks(np.round(np.linspace(np.min(E),np.max(E), 5), 1))
  ax.grid(True)

  ax.set_title("Energy", va='bottom')
  plt.show()

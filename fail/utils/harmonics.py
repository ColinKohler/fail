import numpy as np
import torch

def circular_harmonics(L, theta):
    B = [torch.tensor([1 / np.sqrt(2 * np.pi)] * theta.size(0), device=theta.device).view(-1, 1)]
    for l in range(1, L+1):
        B.append(l * torch.cos(theta) / np.sqrt(np.pi))
        B.append(l * torch.sin(theta) / np.sqrt(np.pi))

    return torch.stack(B).permute(1,0,2).float()

def spherical_harmonics(L, theta):
    pass

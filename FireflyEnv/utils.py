import os, sys

import torch
from torch.autograd import Variable
import numpy as np
from numpy import pi

def row_vector(vector):
    # input, np array or torch tensor, ouput to tensor row vector
    if type(vector) == torch.Tensor:
        return vector.view(1,-1)
    elif type(vector) == np.ndarray:
        return torch.Tensor(vector).view(1,-1)

def col_vector(vector):
    # input, np array or torch tensor, ouput to tensor col vector
    if type(vector) == torch.Tensor:
        return vector.view(-1,1)
    elif type(vector) == np.ndarray:
        return torch.Tensor(vector).view(-1,1)

def shrink(a):
    #a = a.t()
    max_norm = torch.sqrt(1 + torch.min(a**2)/(torch.max(a**2) + 1e-6))
    return a/max_norm

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

def variable(x, **kwargs):
    if torch.cuda.is_available():
        return Variable(x, **kwargs).cuda()
    return Variable(x, **kwargs)

def soft_update(target, source, tau):
    # update using tau
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    # just swap
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def next_path(path_pattern):
    """
    path_pattern = 'file-%s.txt':
    """
    i = 1
    while os.path.exists(path_pattern % i):
        i = i * 2

    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)

    return path_pattern % b


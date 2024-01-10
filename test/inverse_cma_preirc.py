# update 1025, for monkey neural data inverse.

# high level imports
import sys
import os
from pathlib import Path
import configparser
config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
resdir = Path(config['Datafolder']['data'])
workdir = Path(config['Codefolder']['workspace'])
sys.path.append(os.path.abspath(workdir))
sys.path.append(os.path.abspath(workdir/'test'))
os.chdir(workdir)

# analysis related imports
from pathlib import Path
import ray
from Config import Config
from firefly_task import ffacc_real
from monkey_functions import *
from InverseFuncs import *
from numpy import pi
from stable_baselines3 import TD3
import time
from matplotlib import pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
import pickle
import warnings
import matplotlib.pyplot as plt
import copy
from cmaes import CMA
import numpy as np
from numpy.lib.npyio import save
import pandas as pd
from notification import notify
warnings.filterwarnings('ignore')
torch.manual_seed(42)
arg = Config()


print('loading data')
datapath = Path(resdir/'m51_mat_ruiyi\preirc_den_3')
idensity=datapath.name[-1]
with open(datapath, 'rb') as f:
    (states, actions, tasks) = pickle.load(f)

ithden = idensity
savename = datapath.parent/(f'm51_100_{ithden}'+datapath.name)

fix_theta={}
fix_theta={
    4:0.5,
    5:0.5,
    6:0.13,
    7:0.19,
    8:0.15,
    9:0.19,
    10:0.35,
}

print('done process data')

# decide if to continue
optimizer = None
log = []
if savename.is_file():
    print('continue on previous inverse...')
    with open(savename, 'rb') as f:
        log = pickle.load(f)
    optimizer = log[-1][0]

env = ffacc_real.FireFlyPaper(arg)
phi = torch.tensor([[0.5],
                    [pi/2],
                    [0.001],
                    [0.001],
                    [0.001],
                    [0.001],
                    [0.13],
                    [0.001],
                    [0.001],
                    [0.001],
                    [0.001],
                    ])
agent_ = TD3.load('trained_agent/paper.zip')
agent = agent_.actor.mu.cpu()


# use cluster
# ray.init(address="ray://192.168.0.177:6379", _redis_password='5241590000000000',
# log_to_driver=False,ignore_reinit_error=True,_node_ip_address='192.168.0.119')

# use localhost
ray.init(log_to_driver=False, ignore_reinit_error=True)


@ray.remote
def getlogll(x):
    with torch.no_grad():
        return monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=0.01, num_iteration=1, states=states, samples=5, gpu=False).item()


# fresh start
if not optimizer:
    # init condition, we want to at least cover some dynamic range
    init_theta = torch.tensor([[0.5],
                              [1.6],
                               [0.5],
                               [0.5],
                               [0.5],
                               [0.5],
                               [0.13],
                               [0.5],
                               [0.5],
                               [0.5],
                               [0.5]])

    dim = init_theta.shape[0]
    init_cov = torch.diag(torch.ones(dim))*0.3
    cur_mu = init_theta.view(-1)
    cur_cov = init_cov
    optimizer = CMA(mean=np.array(cur_mu), sigma=0.5, population_size=14)
optimizer.set_bounds(np.array([
    [0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.129, 0.01, 0.01, 0.01, 0.01],
    [1.5, 2, 1.5, 1.5, 1.5, 1.5, 0.131, 1, 1, 1, 1]], dtype='float32').transpose())


for generation in range(len(log), len(log)+99):
    solutions = []
    xs = []
    for _ in range(optimizer.population_size):
        x = optimizer.ask().astype('float32')
        if fix_theta:
            for k,v in fix_theta.items():
                x[k] = v
        xs.append(x)
    solutions = ray.get([getlogll.remote(p) for p in xs])
    meanlogll = np.mean(solutions)
    solutions = [[x, s] for x, s in zip(xs, solutions)]
    optimizer.tell(solutions)
    log.append([copy.deepcopy(optimizer), xs, solutions])
    # plt.imshow(optimizer._C)
    # plt.colorbar()
    # plt.show()
    with open(savename, 'wb') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("generation: ", generation, 'logll: ', meanlogll)
    stdout=["{0:0.2f}".format(optimizer._mean[i]) for i in range(len(optimizer._mean)) if i not in fix_theta.keys() ]
    print(stdout)
    notify(stdout)
    stdout=["{0:0.2f}".format((np.diag(optimizer._C)**0.5)[i]) for i in range(len(optimizer._mean)) if i not in fix_theta.keys() ]
    print(stdout)

    if optimizer.should_stop():
        print('stop at {}th generation'.format(str(generation)))

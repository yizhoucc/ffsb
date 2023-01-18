import os
import pandas as pd
from numpy.lib.npyio import save
import numpy as np
from cmaes import CMA
import copy
import matplotlib.pyplot as plt
# from numpy.core.defchararray import array
# from FireflyEnv.env_utils import is_pos_def
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import torch
import heapq
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt
import time
from stable_baselines3 import TD3
torch.manual_seed(42)
from numpy import pi
from InverseFuncs import *
from monkey_functions import *
from FireflyEnv import ffacc_real
from Config import Config
# from cma_mpi_helper import run
import ray
from pathlib import Path
arg = Config()
import os

import requests
import configparser
config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
token=config['Notification']['token']

def notify(group='lab',title='plot',msg='plots ready'):
    notification="https://api.day.app/{}/{}/{}?group={}".format(token,title, msg, group)
    requests.get(notification)


print('loading data')
datapath=Path("/data/neuraltest/1208pack")
savename=datapath.parent/('nouvar_dataprepro_forcenoise_test'+datapath.name)
# savename=datapath.parent/('fullfixre{}'.format(str(ith))+datapath.name)
with open(datapath,'rb') as f:
    states, actions, tasks = pickle.load(f)
print('done process data')


# 1227, some data selection ------------------
maskind=[]
# remove very large error trials
err=np.array([torch.norm(resp[-1,:2]-tar) for resp, tar in zip(states, tasks)])
maskind+=list(np.where(err>0.3)[0])
# remove very short trials
ts=np.array([len(s) for s in states])
maskind+=list(np.where(ts<6)[0])

mask = np.ones(len(tasks), dtype=bool)
mask[maskind]=False

states=[states[i] for i in range(len(mask)) if mask[i]==1]
actions=[actions[i] for i in range(len(mask)) if mask[i]==1]
tasks=tasks[mask]

# decide if to continue
optimizer=None
log=[]
if savename.is_file():
    print('continue on previous inverse...')
    with open(savename, 'rb') as f:
        log = pickle.load(f)
    optimizer=log[-1][0]

env=ffacc_real.FireFlyPaper(arg)
phi=torch.tensor([[0.4],
            [pi/2],
            [0.001],
            [0.001],
            [0.001],
            [0.001],
            [0.13],
            [0.001],
            [0.001],
            [0.001],
            # [0.001],
    ])
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()



# use cluster
# ray.init(address='192.168.0.177:6379', _redis_password='5241590000000000', 
# log_to_driver=False,ignore_reinit_error=True,_node_ip_address='192.168.0.119')

# ray.init(address="ray://192.168.0.177:6379", _redis_password='5241590000000000', 
# log_to_driver=False,ignore_reinit_error=True,_node_ip_address='192.168.0.119')

# use localhost
ray.init(log_to_driver=False,ignore_reinit_error=True,num_cpus=8)

@ray.remote
def getlogll(x,action_var=0.01):
    xx=np.copy(x)
    xx[2:4]=xx[4:6] #1227 force noise to be the same
    # x,action_var=x[:-1],x[-1]
    with torch.no_grad():
        return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(xx).t(), env, action_var=action_var,num_iteration=1, states=states, samples=5,gpu=False).item()

# fresh start
if not optimizer:
    # init condition, we want to at least cover some dynamic range
    init_theta=torch.tensor([[0.4],   
            [1.6],   
            [0.5],   
            [0.5],   
            [0.5],   
            [0.5],   
            [0.13],   
            [0.3],   
            [0.3],   
            [0.1],   
            [0.1],
            # [0.01]
            ])
    # init_theta=torch.tensor([0.8901243,  1.6706846,  0.858392,   0.37444177, 0.05250579, 0.08552641, 0.12986924, 0.20578894, 0.7251345,  0.4741538,  0.40967906]).view(-1,1)
    dim=init_theta.shape[0]
    init_cov=torch.diag(torch.ones(dim))*0.3
    cur_mu=init_theta.view(-1)
    cur_cov=init_cov
    optimizer = CMA(mean=np.array(cur_mu), sigma=0.5,population_size=16)
    # optimizer.set_bounds(np.array([
    # [0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.129, 0.01, 0.01, 0.01, 0.01,1e-8],
    # [1.,2,1.5,1.5,1.5,1.5,0.131,1,1,1,1,0.01]],dtype='float32').transpose())
    optimizer.set_bounds(np.array([
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.129, 0.01, 0.01, 0.01, 0.01],
    [1.,2,1.1,1.1,1.1,1.1,0.131,1,1,1,1]],dtype='float32').transpose())

for generation in range(len(log),len(log)+99):
    tic=time.time()
    solutions = []
    xs=[]
    for _ in range(optimizer.population_size):
        x = optimizer.ask().astype('float32')
        xs.append(x)
    solutions=ray.get([getlogll.remote(p) for p in xs])
    meanlogll=np.mean(solutions)
    solutions=[[x,s] for x,s in zip(xs,solutions)]
    optimizer.tell(solutions)
    log.append([copy.deepcopy(optimizer), xs, solutions])
    # plt.imshow(optimizer._C)
    # plt.colorbar()
    # plt.show()
    with open(savename, 'wb') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("generation: ",generation,'logll: ',meanlogll)
    print(["{0:0.2f}".format(i) for i in optimizer._mean])
    print(["{0:0.2f}".format(i) for i in np.diag(optimizer._C)**0.5])

    toc=time.time()-tic
    notify(group='lab',title='inv',msg='generation {}, {:.0f}'.format(generation, toc))

    if optimizer.should_stop():
        print('stop at {}th generation'.format(str(generation)))



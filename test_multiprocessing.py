import time
import ray
import warnings
warnings.filterwarnings('ignore')
import multiprocessing
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
from cma_mpi_helper import run
import multiprocessing as mp
arg = Config()

import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager, ExitStack
@contextmanager
def suppress(out=True, err=False):
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield
            

env=ffacc_real.FireFlyPaper(arg)
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()
print('loading data')
note='testdcont'
with open("C:/Users/24455/Desktop/bruno_pert_downsample",'rb') as f:
        df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
# df=df[df.target_r>250]
df=df[df.floor_density==0.005]
# floor density are in [0.0001, 0.0005, 0.001, 0.005]
df=df[:-100]
print('process data')
states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
print('done process data')
# misc
phi=torch.tensor([[0.5],
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
# init condition, we want to at least cover some dynamic range
init_theta=torch.tensor([[0.5],   
        [1.6],   
        [0.5],   
        [0.5],   
        [0.5],   
        [0.5],   
        [0.13],   
        [0.3],   
        [0.3],   
        [0.3],   
        [0.3]])
dim=init_theta.shape[0]
def boundary(theta):
    return theta.clamp(1e-3,999)
init_cov=torch.diag(torch.ones(dim))*0.3
selection_ratio=0.25 # ratio of evolution
steps=20 # number of updates
samplesize=10 # population size
ircsample=20 # irc sample per mk trial
action_var=0.01 # assumed monkey action var
batchsize=20 # number of mk trials in  batch
cur_mu=init_theta
cur_cov=init_cov

# define sampling here, to sample parameters
def samplingparam(offset,cov,samplesize=10):
    dist=MultivariateNormal(loc=offset, covariance_matrix=cov)
    return dist.sample_n(samplesize)

def inversesolver(x):
    with torch.no_grad():
        return [getlogll(x).item(),x]

# start solving

# _batchsize=int(epoch/steps*(len(tasks)-batchsize))+batchsize
_batchsize=batchsize
batch_states, batch_actions, batch_tasks = next(data_iter(_batchsize,states, actions, tasks))

sampledparam=samplingparam(cur_mu.flatten(),cur_cov,samplesize=samplesize).tolist()
# getlogll=lambda x: monkeyloss(agent, batch_actions, batch_tasks, phi, torch.tensor(x).t(), env, action_var=action_var,num_iteration=1, states=batch_states, samples=ircsample,gpu=False).item()

@ray.remote
def getlogll(x):
    with torch.no_grad():
        return  monkeyloss(agent, batch_actions, batch_tasks, phi, torch.tensor(x).t(), env, action_var=action_var,num_iteration=1, states=batch_states, samples=ircsample,gpu=False).item()

def getlogll_(x):
    with torch.no_grad():
        return  monkeyloss(agent, batch_actions, batch_tasks, phi, torch.tensor(x).t(), env, action_var=action_var,num_iteration=1, states=batch_states, samples=ircsample,gpu=False).item() 

# solve for logll
print('starting ray')
start=time.time()
with suppress():
    res1=ray.get([getlogll.remote(p) for p in sampledparam])
print('ray finished in {:.1f}'.format(time.time()-start),res1)

print('starting normal')
start=time.time()
with suppress():
    res2=[getlogll_(p) for p in sampledparam]
print('normal finished in {:.1f}'.format(time.time()-start),res2)



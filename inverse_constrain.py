# inverse the schro
# constrain: no obs only prediction
# now the prediction is the integrated belief, the preception.


import sys
sys.path.append('..')
import numpy as np
from cmaes import CMA
import copy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt
import time
from stable_baselines3 import TD3
torch.manual_seed(0)
from numpy import pi
from InverseFuncs import *
from firefly_task import ffacc_real
from env_config import Config
import ray
from pathlib import Path
arg = Config()
import os
from timeit import default_timer as timer
from plot_ult import process_inv, run_trial
from notification import notify
from monkey_functions import *
# --------------------------------------------------------


# load agent and task
env=ffacc_real.FireFlyPaper2(arg)
env.episode_len=50
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
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()

# create the dataset
path=Path('data/')
datafiles=list(path.rglob('*normal/packed'))

states, actions, tasks=[],[],[]
for file in datafiles:
    with open(file, 'rb') as f:
        df= pickle.load(f)
    df=datawash(df)
    df=df[df.category=='normal']
    df=df[df.target_r>250]
    s, a, t=monkey_data_downsampled(df,factor=0.0025)
    states+=s
    actions+=a
    tasks+=t
del df



resfile=Path('res/inv_schroall_constrain_nopert_part2')



# decide if to continue
optimizer=None
log=[]
if resfile.is_file():
    print('continue on previous inverse...')
    with open(resfile, 'rb') as f:
        log = pickle.load(f)
    optimizer=log[-1][0]
else:
    print('starting new inverse ...')

# use localhost
ray.init(log_to_driver=False,ignore_reinit_error=True)

@ray.remote
def getlogll(x):
    with torch.no_grad():
        return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=1e-3,num_iteration=1, states=states, samples=5,gpu=False).item()

# fresh start
if not optimizer:
    # init condition, we want to at least cover some dynamic range
    init_theta=torch.tensor([[0.5],   
            [1.0],   
            [0.5],   
            [0.5],   
            [0.5],   
            [0.5],   
            [0.13],   
            [0.5],   
            [0.5],   
            [0.5],   
            [0.5]])
    dim=init_theta.shape[0]
    init_cov=torch.diag(torch.ones(dim))*0.3
    cur_mu=init_theta.view(-1)
    cur_cov=init_cov
    optimizer = CMA(mean=np.array(cur_mu), sigma=0.5,population_size=14)
    optimizer.set_bounds(np.array([
    [0.2, 0.7, 0.1, 0.49, 0.1, 0.49, 0.129, 0.1, 0.1, 0.1, 0.1],
    [1.,2,2,0.5,2,0.5,0.131,0.9, 0.9, 0.9, 0.6]],dtype='float32').transpose())

optimizer.set_bounds(np.array([
    [0.2, 0.7, 0.1, 0.49, 0.1, 0.49, 0.129, 0.1, 0.1, 0.1, 0.1],
    [1.,2,2,0.5,2,0.5,0.131,0.9, 0.9, 0.9, 0.6]],dtype='float32').transpose())

for generation in range(len(log),len(log)+399):
    start=timer()
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

    with open(resfile, 'wb+') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('done, ',timer()-start)
    print("generation: ",generation,'-logll: ',meanlogll)
    print('cur estimatation ',["{0:0.2f}".format(i) for i in optimizer._mean])
    print('cur uncertainty ',["{0:0.2f}".format(i) for i in np.diag(optimizer._C)**0.5])
    notify(msg="".join(['{:0.1f} '.format(i) for i in optimizer._mean]))
    if optimizer.should_stop():
        print('stop at {}th generation'.format(str(generation)))




    
# inverse human subject individually

import enum
import pickle
import os
import pandas as pd
# os.chdir(r"C:\Users\24455\iCloudDrive\misc\ffsb")
import numpy as np
from cmaes import CMA
import copy
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import torch
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
from timeit import default_timer as timer
from plot_ult import process_inv

env=ffacc_real.FireFlyPaper(arg)
phi=torch.tensor([[1],
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
agent_=TD3.load('trained_agent/paper')
agent=agent_.actor.mu.cpu()

datafolder=Path('/data/human/turnoffdata')

for filename in datafolder.glob("*.p"):
    print(filename)
    # if filename.name=='Alex.p': continue
    savename=datafolder/"inv{}".format(Path(filename).name)

    # load data
    with open(filename, 'rb') as f:
        states, actions, tasks, stimdur = pickle.load(f)
    print('inverse ',filename.name)

    # load inv
    optimizer=None
    log=[]
    init_theta=torch.tensor([[2],   
        [1.6],   
        [0.5],   
        [0.5],   
        [0.5],   
        [0.5],   
        [0.13],   
        [0.1],   
        [0.1],   
        [0.1],   
        [0.1]])
    dim=init_theta.shape[0]
    init_cov=torch.diag(torch.ones(dim))*0.3
    # if savename.is_file():
    #     print('continue on previous inverse...')
    #     with open(savename, 'rb') as f:
    #         log = pickle.load(f)
    #     optimizer=log[-1][0]
    # else: # init condition from the common inverse
    print('starting new inverse ...')
    # init_theta,init_cov,_=process_inv(logs[invtag],removegr=False)

    optimizer = CMA(mean=np.array(init_theta.view(-1)), sigma=max(torch.diag(init_cov)).item()**0.5,population_size=14)
    # optimizer.set_bounds(np.array([
    # [0.1, 0.1, 0.01, 0.01,0.499, 0.499, 0.129, 0.01, 0.01, 0.01, 0.01],
    # [2.,2,2,2,0.5,0.5,0.131,2,2,1,1]],dtype='float32').transpose())
    optimizer.set_bounds(np.array([
    [0.1, 0.1, 0.01, 0.01,0.01, 0.01, 0.129, 0.01, 0.01, 0.01, 0.01],
    [2.,2,2,2,1.5,1.5,0.131,2,2,1,1]],dtype='float32').transpose())

    # use localhost
    ray.init(log_to_driver=False,ignore_reinit_error=True)

    @ray.remote
    def getlogll(x):
        with torch.no_grad():
            return  monkeyloss_turnoffdata(agent, actions, tasks, phi, torch.tensor(x).t(), env, stimdur=stimdur, action_var=0.01,num_iteration=1, states=states, samples=10,gpu=False).item()

    for generation in range(len(log),len(log)+30):
        start=timer()
        solutions = []
        xs=[]
        for _ in range(optimizer.population_size):
            x = optimizer.ask().astype('float32')
            xs.append(x)
        solutions=ray.get([getlogll.remote(p) for p in xs])
        
        solutions=[[x,s] for x,s in zip(xs,solutions)]
        optimizer.tell(solutions)
        log.append([copy.deepcopy(optimizer), xs, solutions])
        with open(savename, 'wb') as handle:
            pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('done, ',timer()-start)
        print(filename.name," generation: ",generation)
        print(["{0:0.2f}".format(i) for i in optimizer._mean])
        print(["{0:0.2f}".format(i) for i in np.diag(optimizer._C)**0.5])
        if optimizer.should_stop():
            print('stop at {}th generation'.format(str(generation)))










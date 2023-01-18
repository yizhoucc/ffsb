# eval the persub inverse

import pickle
import os
import pandas as pd
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


# use localhost
ray.init(log_to_driver=False,ignore_reinit_error=True)

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

numhsub,numasub=25,14


logs={'a':'/data/human/fixragroup','h':'/data/human/clusterpaperhgroup'}

for invtag in ['h','a']:
        
    for isub in range(numhsub):
        
        dataname="{}sub{}".format(invtag,str(isub))
        foldername='persub3of5dp'
        trainname=Path("/data/human/{}/inv{}sub{}".format(foldername,invtag,str(isub)))
        evalname=Path("/data/human/{}/evaltrain_inv{}sub{}".format(foldername,invtag,str(isub)))
        
        
        # load data
        with open('/data/human/{}'.format(dataname), 'rb') as f:
            states, actions, tasks = pickle.load(f)
        print(len(states))

        # select shorts
        taskdist=np.array([np.linalg.norm(x) for x in tasks])
        distsortind=np.argsort(taskdist)
        trainind=distsortind[:int(len(distsortind)*3/4)]
        states, actions, tasks = [states[t] for t in trainind], [actions[t] for t in trainind], tasks[trainind]
        
        

        # load inv
        optimizer=None
        log=[]
        if trainname.is_file():
            print('eval the previous inverse...')
            with open(trainname, 'rb') as f:
                log = pickle.load(f)
            optimizer=log[-1][0]
        else:
            continue
        traininglogll = [ t[-1] for t in log[-1][-1] ]
        
        # optimizer.set_bounds(np.array([
        # [0.1, 0.1, 0.01, 0.01,0.499, 0.499, 0.129, 0.01, 0.01, 0.01, 0.01],
        # [2.,2,2,2,0.5,0.5,0.131,2,2,1,1]],dtype='float32').transpose())
        optimizer.set_bounds(np.array([
        [0.1, 0.1, 0.01, 0.01,0.01, 0.01, 0.129, 0.01, 0.01, 0.01, 0.01],
        [2.,2,2,2,1.5,1.5,0.131,2,2,1,1]],dtype='float32').transpose())

        @ray.remote
        def getlogll(x):
            with torch.no_grad():
                return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=0.01,num_iteration=1, states=states, samples=5,gpu=False).item()
        
        start=timer()
        evallogll = []
        xs=[]
        for _ in range(optimizer.population_size):
            x = optimizer.ask().astype('float32')
            xs.append(x)
        evallogll=ray.get([getlogll.remote(p) for p in xs])

        log.append([traininglogll, evallogll])
        with open(evalname, 'wb') as handle:
            pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('done, ',timer()-start)
        print(isub,"sub: ")
        print(traininglogll, evallogll)
        print("{0:0.2f}".format(np.mean(traininglogll)),"{0:0.2f}".format(np.mean(evallogll)))
        # print(["{0:0.2f}".format(i) for i in optimizer._mean])
        # print(["{0:0.2f}".format(i) for i in np.diag(optimizer._C)**0.5])


import requests
group='lab'
title='inverse'
msg='job done'
notification="https://api.day.app/mWBcxqxVNZRUzECXxiLxs5/{}/{}?group={}".format(title, msg, group)
requests.get(notification)




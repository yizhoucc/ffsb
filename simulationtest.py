# validate the inverse on simulation data.
# generate simulatino data and inverse
import pandas as pd
from numpy.lib.npyio import save
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
import heapq
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt
import time
from stable_baselines3 import TD3
torch.manual_seed(0)
from numpy import pi
from InverseFuncs import *
from firefly_task import ffacc_real
from env_config import Config
# from cma_mpi_helper import run
import ray
from pathlib import Path
arg = Config()
import os
from timeit import default_timer as timer
from plot_ult import process_inv, run_trial
from notification import notify

# --------------------------------------------------------


# load agent and task
env=ffacc_real.FireFlyPaper(arg)
env.episode_len=50
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
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()

# create the dataset
n_normaltrial=0
n_perttrial=1000
path=Path('Z:/simulation')
datafile=path/'0119simulation1000'
resfile=datafile.parent/'inv2_{}'.format(datafile.name)

if datafile.is_file():
    print('use previous simulation trajectory')
    with open(datafile, 'rb') as f:
        states, actions, tasks, groundtruth = pickle.load(f)
else:
    print('generate new simulation trajectory')

    alltruth=[]
    groundtruth=torch.tensor([
        [0.5],
        [0.9],
        [0.5],
        [0.1],
        [0.1],
        [0.5],
        [1.3000e-01],
        [0.5],
        [0.5],
        [0.1],
        [0.1]])
    groundtruth[6]=0.13
   
    states, actions, tasks=[],[],[]
    perts=[]

    while len(actions)<n_normaltrial:
        env.reset(phi=phi, theta=groundtruth, pro_traj=None )
        epactions,_,_,epstates=run_trial(agent,env,given_action=None, given_state=None, action_noise=0.1)
        if len(epactions)>5:
            tasks.append([env.goalx.item(),env.goaly.item()])
            actions.append(torch.stack(epactions))
            states.append(torch.stack(epstates)[:,:,0])

    while len(actions)<n_normaltrial+n_perttrial:
        pertpeakt=np.random.randint(0,20)
        pertstd=np.random.randint(1,5)
        pertampv=np.random.random(1)*2-1
        pertampw=np.random.random(1)*2-1
        ts=np.linspace(0,99,66)
        rawpert=np.exp(-0.5*(ts-pertpeakt)**2/(pertstd)**2)
        pert=np.vstack([rawpert*pertampv, rawpert*pertampw]).astype('float32').T

        env.reset(phi=phi, theta=groundtruth, )
        env.pro_traj=pert
        epactions,_,_,epstates=run_trial(agent,env,given_action=None, given_state=None, action_noise=0.1)
        if len(epactions)>5:
            tasks.append([env.goalx.item(),env.goaly.item()])
            actions.append(torch.stack(epactions))
            states.append(torch.stack(epstates)[:,:,0])
            perts.append(pert)

    with open(datafile,'wb+') as f:
        pickle.dump((states, actions, tasks, groundtruth), f, protocol=pickle.HIGHEST_PROTOCOL)

    notify(msg='data ready')

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
    [0.2, 0.2, 0.01, 0.01, 0.01, 0.01, 0.129, 0.5-0.01, 0.5-0.01, 0.1-0.01, 0.1-0.01],
    [1.,2,1,1,1,1,0.131,0.5+0.01, 0.5+0.01, 0.1+0.01, 0.1+0.01]],dtype='float32').transpose())


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
    # plt.imshow(optimizer._C)
    # plt.colorbar()
    # plt.show()
    with open(resfile, 'wb') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('done, ',timer()-start)
    print("generation: ",generation,'-logll: ',meanlogll)
    # print('ground truth ',["{0:0.2f}".format(i) for i in groundtruth])
    print('cur estimatation ',["{0:0.2f}".format(i) for i in optimizer._mean])
    print('cur uncertainty ',["{0:0.2f}".format(i) for i in np.diag(optimizer._C)**0.5])
    notify(msg="".join(['{:0.1f} '.format(i) for i in optimizer._mean]))
    if optimizer.should_stop():
        print('stop at {}th generation'.format(str(generation)))




# check res
# ---------------------------

# path=Path('Z:/simulation')
# datafile=path/'simulationnopert10group50normfixgrpert'
# resfile=datafile.parent/'invblief{}'.format(datafile.name)
# if resfile.is_file():
#     with open(resfile, 'rb') as f:
#         log = pickle.load(f)
#     optimizer=log[-1][0]

# if datafile.is_file():
#     print('use previous simulation trajectory')
#     with open(datafile, 'rb') as f:
#         states, actions, tasks, alltruth = pickle.load(f)

# theta,cov,err=process_inv(resfile,removegr=False,usingbest=True)
# err=torch.diag(cov)**0.5
# for t in alltruth:
#     plt.scatter(list(range(11)),t,color='b',alpha=0.2)
# plt.errorbar(list(range(11)),theta.view(-1),yerr=err,color='r',ls='none')

# with initiate_plot(6,4,300) as f:
#     ax=f.add_subplot(111)
#     xs=np.array(list(range(11)))
#     for t in alltruth:
#         plt.scatter(xs,t,color='b',alpha=0.2,s=9,label='ground truth')
#     plt.errorbar(xs+0.1,theta.view(-1),yerr=err,color='r',ls='none',label='group inverse')
#     quickleg(ax)
#     quickspine(ax)
#     ax.set_xticks(list(range(11)))
#     ax.set_xticklabels(theta_names_, rotation=45, ha='right')
#     ax.set_ylabel('inferred param value')
    
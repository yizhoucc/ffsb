# test if log ll evaulation is biased by distance and angle
# the hypotheis: longer trial takes longer time, thus more variable, and logll will be smaller


# imports
# ---------------------------------------------------------------------------------
import os
import pandas as pd
from numpy.lib.npyio import save
os.chdir(r"C:\Users\24455\iCloudDrive\misc\ffsb")
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
from monkey_functions import *
from FireflyEnv import ffacc_real
from Config import Config
# from cma_mpi_helper import run
import ray
from pathlib import Path
arg = Config()
import os
from timeit import default_timer as timer
from plot_ult import run_trial, sample_all_tasks, initiate_plot

# -------------------------------------------------------------------------------


# load agent and task --------------------------------------------------------
env=ffacc_real.FireFlyPaper(arg)
env.episode_len=50
env.debug=1
env.terminal_vel=0.1
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


# generate some simulation trajectory, vary by xxx----------------------------------

# settings
usepert=False
n_perttrial=100
gridreso=30
path=Path('Z:/simulation')
datafile=path/'simulationpert100'
resfile=datafile.parent/'invblief{}'.format(datafile.name)




gridmap=np.array(np.meshgrid(np.linspace(0,1,gridreso), np.linspace(-0.7,0.7,gridreso))).reshape(2,-1).T
tasks=[]
for task in gridmap:
    d,a=xy2pol(task, rotation=False)
    if env.min_distance<=d<=env.max_distance and env.min_angle<=a<env.max_angle:
        tasks.append(task)
tasks=np.array(tasks)

# check tasks on map
plt.scatter(gridmap[:,0],gridmap[:,1],s=2)
plt.scatter(tasks[:,0],tasks[:,1],s=2)
# res=np.array(res)
# plt.scatter(res[:,0],res[:,1],s=2,color='orange')
plt.axis('equal')
len(tasks)

groundtruth=env.reset_task_param()
theta=env.reset_task_param()
states, actions=[],[]
perts=[]
for task in tasks:
    pertpeakt=np.random.randint(0,20)
    pertstd=np.random.randint(1,5)
    if usepert:
        pertampv=np.random.random(1)*2-1;pertampw=np.random.random(1)*2-1
    else:
        pertampv=0;pertampw=0

    ts=np.linspace(0,99,66)
    rawpert=np.exp(-0.5*(ts-pertpeakt)**2/(pertstd)**2)
    pert=np.vstack([rawpert*pertampv, rawpert*pertampw]).astype('float32').T

    env.reset(phi=phi, theta=groundtruth,goal_position=task )
    env.pro_traj=pert
    epactions,_,_,epstates=run_trial(agent,env,given_action=None, given_state=None, action_noise=0.1)
    if len(epactions)>3:
        # tasks.append([env.goalx.item(),env.goaly.item()])
        actions.append(torch.stack(epactions))
        states.append(torch.stack(epstates)[:,:,0])
        perts.append(pert)



# eval the log ll of the trials--------------------------------------------------------

def getlogll(states, actions, tasks):
    with torch.no_grad():
        return  monkeyloss_(agent, actions, tasks, phi, theta, env, action_var=0.001,num_iteration=1, states=states, samples=9,gpu=False,debug=True)

loglls=getlogll(states, actions, tasks)



# ploting --------------------------------------------------------------------------

with initiate_plot(9,3,300) as fig:
    ax=fig.add_subplot(131)
    c=ax.scatter(tasks[:,0],tasks[:,1],c=loglls,s=10,cmap='bwr',norm=getcbarnorm(np.min(loglls), np.mean(loglls), np.max(loglls)))
    plt.colorbar(c, label='- log likelihood')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.axes.xaxis.set_ticks([0,0.5,1])
    ax.set_xlabel('world x [2m]')
    ax.set_ylabel('world y [2m]' )
    ax.set_aspect('equal')
    ax.set_title('trial likelihoods')

    ax=fig.add_subplot(132)
    c=ax.scatter(tasks[:,0],tasks[:,1],c=[len(a)/10 for a in actions],s=10,cmap='bwr',norm=getcbarnorm(np.min([len(a)/10 for a in actions]), np.mean([len(a)/10 for a in actions]), np.max([len(a)/10 for a in actions])))
    plt.colorbar(c, label='trial length [s]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.axes.xaxis.set_ticks([0,0.5,1])
    ax.set_xlabel('world x [2m]')
    ax.set_ylabel('world y [2m]' )
    ax.set_aspect('equal')
    ax.set_title('trial length')

    ax=fig.add_subplot(133)
    c=ax.scatter(tasks[:,0],tasks[:,1],c=loglls/np.array([len(a)/10 for a in actions]),s=10,cmap='bwr',norm=getcbarnorm(np.min(loglls/np.array([len(a)/10 for a in actions])), np.mean(loglls/np.array([len(a)/10 for a in actions])), np.max(loglls/np.array([len(a)/10 for a in actions]))))
    plt.colorbar(c, label='1/trial length [s]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.axes.xaxis.set_ticks([0,0.5,1])
    ax.set_xlabel('world x [2m]')
    ax.set_ylabel('world y [2m]' )
    ax.set_aspect('equal')
    ax.set_title('trial d logll / dt')
    plt.tight_layout()
    plt.show()








with open('sim_logll', 'wb+') as f:
    pickle.dump(
        [states, actions, tasks, loglls,],
        f,  protocol=pickle.HIGHEST_PROTOCOL
    )



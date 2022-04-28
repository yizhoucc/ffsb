# inverse obs alone 

# distribution of likelihood of each desnity-------------------------------------



import os
import pandas as pd
from numpy.lib.npyio import save
os.chdir(r"C:\Users\24455\iCloudDrive\misc\ffsb")
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
from plot_ult import *
arg = Config()
ray.init(log_to_driver=False,ignore_reinit_error=True)


env=ffacc_real.FireFlyPaper(arg)
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



name='bruno'
# load the final theta
with open('Z:/{}_normal/normalallpacked'.format(name), 'rb') as f:
    log = pickle.load(f)
finaltheta=torch.tensor(log[-1][0]._mean)
finaltheta=torch.tensor([0.5, 1.57, 1.3387, 0.5589, 0.0433, 0.0616, 0.1304, 0.1614, 0.7076,
        0.1921, 0.2036])
# load the testdata of each density
print('loading data')
datapath=Path("Z:\\{}_normal\\packed".format(name))
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df=df[df.target_r>200]
densities=sorted(pd.unique(df.floor_density))


log=[]
samples=500
numtrial=20
obs_trajectory=np.random.normal(0,1,size=(numtrial,samples,env.episode_len,2)) # 2 for action dim

for d in densities:
    densitydf=df[df.floor_density==d]
    print('process data ',d)
    states, actions, tasks=monkey_data_downsampled(densitydf[-numtrial:],factor=0.0025)
    print('done process data')

    @ray.remote
    def getlogll(x,obs_trajectory):
        with torch.no_grad():
            obs_trajectory_reparam=obs_trajectory*np.array(x)[4:6]
            return  monkeyloss_fixobs(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=0.01,num_iteration=1, states=states,gpu=False, obs_trajectory=obs_trajectory_reparam,usestop=False).item()

    # @ray.remote
    # def getlogll(x,obs_trajectory):
    #     with torch.no_grad():
    #         return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=0.01,num_iteration=1, states=states,gpu=False, samples=5).item()


    obsvls=np.linspace(1e-4,0.5, num=256)
    thetas=[]
    for obs in obsvls:
        theta=finaltheta.clone().detach()
        theta[4]=obs
        thetas.append(theta)
    likelihoods=ray.get([getlogll.remote(theta,obs_trajectory) for theta in thetas])

    likelihoods=np.array(likelihoods)*-1 # log likelihood
    log.append(likelihoods)

    # likelihoods=np.exp(likelihoods) # likelihood
    # pdf=[1/sum(likelihoods) * p for p in likelihoods]

l=np.array(log)
l=np.exp(l)
p=l*1/np.sum(l, axis=1).reshape(-1,1)


with open('{}obsdenslog'.format(name), 'wb') as handle:
        pickle.dump((obsvls, p), handle, protocol=pickle.HIGHEST_PROTOCOL)


basecolor=hex2rgb(color_settings['o'])
colors=colorshift(basecolor, basecolor, 1, len(densities))


with initiate_plot(9,3) as fig:
    for i in range(len(densities)):
        if i!=0:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax=fig.add_subplot(1,4,i+1, sharey=ax)
        else:
            ax=fig.add_subplot(1,4,1)
            ax.set_ylabel('probability')
        ax.set_xlim(0,max(obsvls))
        ax.set_xticks([0,0.1,0.2,0.5])
        pdd,d=p[i], densities[i]
        ax.fill_between(obsvls,pdd, label='density {}'.format(str(d)),color=colors[i],alpha=0.8,edgecolor='none')
        ax.set_xlabel('observation noise std [m/s]')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0,np.max(p)*1.2)
        ax.legend()
        

# np.random.normal(0,1,size=(5,env.episode_len,2)).shape


# env.obs_traj=np.random.normal(0,1,size=(5,env.episode_len,2))




# env.obs_traj[int(env.trial_timer.item())]*env.noise_scale

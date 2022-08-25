

print(
'''
varying two parameters and plot the log likelihood surface.

oringally for showing the ridge of process noise and observation noise.

''')

# imports
# -------------------------------------------------------------
import matplotlib
from playsound import playsound
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import os
import pandas as pd
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
from numpy import linspace, pi
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
from plot_ult import ll2array, normalizematrix, npsummary, quickleg, quicksave, run_trial, sample_all_tasks, initiate_plot,process_inv, eval_curvature, xy2pol,suppress, quickspine, select_tar


# load agent and task --------------------------------------------------------
env=ffacc_real.FireFlyPaper(arg)
env.episode_len=50
env.debug=1
env.terminal_vel=0.2
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


# load the individual subject logs --------------------
numhsub,numasub=25,14
foldername='persub1cont'
logs={'a':'Z:/human/fixragroup','h':'Z:/human/clusterpaperhgroup'}

invres={'a':[],'h':[]}
for isub in range(numhsub):
    dataname="hsub{}".format(str(isub))
    savename=Path("Z:/human/{}".format(foldername))/"invhsub{}".format(str(isub))
    if savename.is_file():
        invres['h'].append(process_inv(savename,ind=31, usingbest=True, removegr=False))
for isub in range(numasub):
    dataname="asub{}".format(str(isub))
    savename=Path("Z:/human/{}".format(foldername))/"invasub{}".format(str(isub))
    if savename.is_file():
        invres['a'].append(process_inv(savename,ind=31, usingbest=True,removegr=False))

# pick a subject to vary 2 pram -----------------------
for asubind in range(numasub):
# asubind=1

    sub=invres['h'][asubind]
    subtheta=sub[0]
    basetheta=np.array(subtheta).reshape(-1)
    dataname="hsub{}".format(str(asubind))
    with open('Z:/human/{}'.format(dataname), 'rb') as f:
        states, actions, tasks = pickle.load(f)
    print(len(states))



    ray.init(log_to_driver=False,ignore_reinit_error=True)

    @ray.remote
    def getlogll(x):
        if torch.any(x>2) or torch.any(x<=0):
            return None
        with torch.no_grad():
            return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).view(-1,1), env, action_var=0.01,num_iteration=1, states=states, samples=5,gpu=False).item()

    # vary two noises in the full range
    gridreso=11 # num prameters in theta
    dx,dy=np.zeros((gridreso)),np.zeros((gridreso))
    dx[2]=1 # process noise
    dy[4]=1 # observation noise
    basetheta[2]=0
    basetheta[4]=0
    X,Y=np.linspace(1e-3,2,gridreso),np.linspace(1e-3,2,gridreso)
    paramls=[]
    for i in range(gridreso):
        for j in range(gridreso):
            theta=torch.tensor(dx*X[i]+dy*Y[j]+basetheta).float().view(-1,1)
            paramls.append(theta)


    Z=ray.get([getlogll.remote(each) for each in paramls])
    with open('vary2noises_h{}sub'.format(asubind), 'wb+') as f:
            pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)



print(
    '''
    results shows that process noise and obs noise have a ridge
    it is like a y=1/x relationship, as expected
    one other thing to note is, we can vary obs noise quite a bit, but not process noise. this is because we did not vary process gain with the process noise. so the process noise paramter has constraints.
    ''')
gridreso=11 # num prameters in theta
X,Y=np.linspace(1e-3,2,gridreso),np.linspace(1e-3,2,gridreso)

for asubind in range(numasub):
    with open('vary2noises_asd{}sub'.format(asubind), 'rb') as f:
        paramls,Z= pickle.load(f)
    Z=[a if a else np.nan for a in Z]
    formatedZ=np.array(Z).reshape(int(len(Z)**0.5),int(len(Z)**0.5)).T
    sub=invres['a'][asubind]
    subtheta=sub[0]
    basetheta=np.array(subtheta).reshape(-1)

    with initiate_plot(3,3,300) as f:
        ax=f.add_subplot(111)
        cmap = matplotlib.cm.jet
        cmap.set_bad('black',1.)
        im=ax.contourf(formatedZ,origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),aspect='auto',cmap=cmap)    
        ax.set_aspect('equal')
        plt.colorbar(im,label='log likelihood') 
        ax.set_xlabel('process noise')
        ax.set_ylabel('observation noise')
        ax.scatter(basetheta[5],basetheta[8],label='inferred theta') # inferred delta
        quickleg(ax,bbox_to_anchor=(-0.1,-0.2))
        quickspine(ax)
        ax.set_title('agroup subject {}'.format(asubind))
        quicksave('agroup subject {}'.format(asubind))



with initiate_plot(3,3,300) as f:
    ax=f.add_subplot(111)
    plt.bar(list(range(11)),clf.coef_[0])
    plt.xticks(list(range(11)))
    ax.set_xticklabels(theta_names_)

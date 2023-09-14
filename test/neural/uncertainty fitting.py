

import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
os.chdir('../..')
print(os.getcwd())
import numpy as np
import torch
from numpy import pi
from matplotlib import pyplot as plt
from firefly_utils.data_handler import data_handler
from firefly_utils.spike_times_class import spike_counts
from firefly_utils.behav_class import *
from firefly_utils.lfp_class import lfp_class
from copy import deepcopy
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.io import loadmat
import pickle
import pandas as pd
from numpy.lib.npyio import save
from cmaes import CMA
import copy
import matplotlib.pyplot as plt
import warnings
import heapq
from torch.distributions.multivariate_normal import MultivariateNormal
import time
from stable_baselines3 import TD3
from InverseFuncs import *
from monkey_functions import *
from firefly_task import ffacc_real
from env_config import Config
from notification import notify
from pathlib import Path
from sklearn import linear_model
import configparser
from plot_ult import *
config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
resdir=Path(config['Datafolder']['data'])

dat = loadmat(resdir/'neuraltest/m53s31.mat') # schro
# dat = loadmat(resdir/'neural/m53s36.mat') 
behav_stat_key = 'behv_stats'
spike_key = 'units'
behav_dat_key = 'trials_behv'
lfp_key = 'lfps'
pre_trial_dur = 0.5
post_trial_dur = 0.5
exp_data = data_handler(dat, behav_dat_key, spike_key, lfp_key, behav_stat_key, pre_trial_dur=pre_trial_dur, extract_fly_and_monkey_xy=True,
                        post_trial_dur=post_trial_dur,extract_cartesian_eye_and_firefly=True,
                        lfp_beta=None, lfp_alpha=None, extract_lfp_phase=True)
exp_data.set_filters('all', True)
ts = exp_data.rebin_time_stamps(0.1)
t_targ=dict_to_vec(exp_data.behav.events.t_targ)
t_start=t_targ
t_stop = dict_to_vec(exp_data.behav.events.t_stop)
var_names = 'rad_vel','ang_vel','x_monk','y_monk', 'eye_hori', 'eye_vert','x_fly_screen','z_fly_screen','x_eye_screen','z_eye_screen',"x_fly_rel","y_fly_rel","ang_vel"
y,X,trial_idx = exp_data.concatenate_inputs(*var_names,t_start=t_start,t_stop=t_stop, time_stamps=ts)
trials=np.unique(trial_idx)


with open(resdir/'neuraltest/res/m53s31_0223newformatbelief', 'rb') as f:
    res = pickle.load(f)

y_ = res['y']
X = {k: res[k] for k in ['rad_vel', 'ang_vel', 'x_monk', 'y_monk']}
trial_idx = res['trial_idx']
beliefs = res['belief']
covs = res['covs']
s = np.vstack([v for v in X.values()])
s = s.T
# define a b-spline
kernel_len = 7 # should be about +- 325ms 
knots = np.hstack(([-1.001]*3, np.linspace(-1.001,1.001,5), [1.001]*3))
tp = np.linspace(-1.,1.,kernel_len)
bX = splineDesign(knots, tp, ord=4, der=0, outer_ok=False)
with initiate_plot(3,2,200) as f:
    ax=f.add_subplot(111)
    plt.plot(bX)
    plt.title('B-spline kernel')
    quickspine(ax)
    plt.xticks([0,kernel_len-1])
    ax.set_xticklabels([-kernel_len*50, kernel_len*50])
    plt.xlabel('time, ms')
    plt.ylabel('coef')

with suppress():
    modelX = convolve_loop(y_.T, trial_idx, bX) # ts, neurons
pos_xy = np.hstack((X['x_monk'].reshape(-1,1), X['y_monk'].reshape(-1,1))) # ts, xy
# remove bad data
non_nan = ~np.isnan(pos_xy.sum(axis=1))
modelX = modelX[non_nan]
pos_xy = pos_xy[non_nan]
belief_xy = beliefs[:,[0,1]][non_nan]

# calculate relative beliefs and states
states_rel, belief_rel=[],[]
for itrial in range(len(trials)):
    xr, yr=world2mk(X['x_monk'][trial_idx==trials[itrial]],X['y_monk'][trial_idx==trials[itrial]],X['ang_vel'][trial_idx==trials[itrial]],exp_data.behav.continuous.x_fly[trials[itrial]], exp_data.behav.continuous.y_fly[trials[itrial]])
    states_rel.append(np.vstack([xr,yr]))

    xr, yr=world2mk(beliefs[trial_idx==trials[itrial]][:,0],beliefs[trial_idx==trials[itrial]][:,1],X['ang_vel'][trial_idx==trials[itrial]],exp_data.behav.continuous.x_fly[trials[itrial]], exp_data.behav.continuous.y_fly[trials[itrial]])
    belief_rel.append(np.vstack([xr,yr]))

states_rel=np.hstack(states_rel).T
belief_rel=np.hstack(belief_rel).T
R = lambda theta : np.array([[np.cos(theta/180*np.pi),-np.sin(theta/180*np.pi)],[np.sin(theta/180*np.pi),np.cos(theta/180*np.pi)]])
belief_cov_rel=torch.tensor((R(180)@np.array(covs[:,:2,:2])@R(180).T))


# custom method -------------------
import torch
from torch.autograd import Variable

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, LBFGS, SGD
from torch.utils.data import Dataset, DataLoader


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def likelihoodloss(pred, mus, covs, ndim=2):
    p=torch.distributions.multivariate_normal.MultivariateNormal(mus,covariance_matrix=covs)
    genloss=-torch.mean(p.log_prob(pred))
    # genloss=-torch.mean(torch.clip(p.log_prob(target),-10,3))
    return genloss

mus=torch.tensor(belief_xy)
mus.shape
covs=torch.tensor(covs[:,:2,:2])
covs.shape
r=torch.tensor(modelX).float()

inputDim = r.shape[1]        # takes variable 'x' 
outputDim = mus.shape[1]       # takes variable 'y'
learningRate = 0.01 
epochs = 100

model = linearRegression(inputDim, outputDim)

criterion = likelihoodloss
optimizer = LBFGS(model.parameters(), history_size=9, max_iter=4)
max_norm=22
losslog=[]
for epoch in range(epochs):
    running_loss = 0.0


    x_ = Variable(r, requires_grad=True)


    def closure():
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x_)

        # Compute loss
        loss = criterion(y_pred,mus, covs)

        # Backward pass
        loss.backward()

        return loss

    # Update weights
    optimizer.step(closure)

    # Update the running loss
    loss = closure()

    print(loss)
    losslog.append(loss.clone().detach())
    # print('epoch {}, loss {}'.format(epoch, loss.item()))

plt.plot(losslog)
quickspine(plt.gca())
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


with torch.no_grad():
    if torch.cuda.is_available():
        pred = model(Variable(r.cuda())).cpu().data.numpy()
    else:
        pred = model(Variable(r)).data.numpy()

every=5
with initiate_plot(6, 3, 200) as f:
    ax=f.add_subplot(121)
    thispred,thistrue=pred[::every,0],(mus[::every,0]).numpy()
    thispred.shape
    thistrue.shape
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('world centeric x')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    
    ax=f.add_subplot(122)
    thispred,thistrue=pred[::every,1],(mus[::every,1]).numpy()
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('world centeric y')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    plt.tight_layout()


# -----------------
mus=torch.tensor(belief_rel)
mus.shape
covs=belief_cov_rel
covs.shape
r=torch.tensor(modelX).float()

inputDim = r.shape[1]        # takes variable 'x' 
outputDim = belief_rel.shape[1]       # takes variable 'y'
learningRate = 0.01 
epochs = 100

model = linearRegression(inputDim, outputDim)

criterion = likelihoodloss
optimizer = LBFGS(model.parameters(), history_size=9, max_iter=4)
max_norm=22
losslog=[]
for epoch in range(epochs):
    running_loss = 0.0


    x_ = Variable(r, requires_grad=True)


    def closure():
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x_)

        # Compute loss
        loss = criterion(y_pred,mus, covs)

        # Backward pass
        loss.backward()

        return loss

    # Update weights
    optimizer.step(closure)

    # Update the running loss
    loss = closure()

    print(loss)
    losslog.append(loss.clone().detach())
    # print('epoch {}, loss {}'.format(epoch, loss.item()))

plt.plot(losslog)
quickspine(plt.gca())
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


with torch.no_grad():
    if torch.cuda.is_available():
        pred = model(Variable(r.cuda())).cpu().data.numpy()
    else:
        pred = model(Variable(r)).data.numpy()

every=5
with initiate_plot(6, 3, 200) as f:
    ax=f.add_subplot(121)
    thispred,thistrue=pred[::every,0],(mus[::every,0]).numpy()
    thispred.shape
    thistrue.shape
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('belief egocenteric x')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    
    ax=f.add_subplot(122)
    thispred,thistrue=pred[::every,1],(mus[::every,1]).numpy()
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('belief egocenteric y')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    plt.tight_layout()


# model weights
plt.plot((model.linear.weight.clone().detach()[0]))
plt.plot((model.linear.weight.clone().detach()[1]))

plt.plot(sorted(model.linear.weight.clone().detach()[0]))
plt.plot(sorted(model.linear.weight.clone().detach()[1])) 


xinds=torch.argsort(torch.abs(model.linear.weight.clone().detach()[0]))
plt.plot((model.linear.weight.clone().detach()[1])[xinds])

yinds=torch.argsort(torch.abs(model.linear.weight.clone().detach()[1]))
plt.plot((model.linear.weight.clone().detach()[0])[yinds])

xyinds=list( (set(xinds[:99].tolist())).intersection(set(yinds[:99].tolist())) )
plt.plot((model.linear.weight.clone().detach()[0])[xyinds])
plt.plot((model.linear.weight.clone().detach()[1])[xyinds])





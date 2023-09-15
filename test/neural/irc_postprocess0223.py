
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
import configparser
from plot_ult import *

config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
resdir=config['Datafolder']['data']
resdir = Path(resdir)

def dict_to_vec(dictionary):
    return np.hstack(list(dictionary.values()))

def time_stamps_rebin(time_stamps, binwidth_ms=20):
    rebin = {}
    for tr in time_stamps.keys():
        ts = time_stamps[tr]
        tp_num = np.floor((ts[-1] - ts[0]) * 1000 / (binwidth_ms))
        rebin[tr] = ts[0] + np.arange(tp_num) * binwidth_ms / 1000.
    return rebin


print('start loading...')
dat = loadmat(resdir/'neuraltest/m53s31.mat')

# print(dat.keys())
behav_stat_key = 'behv_stats'
spike_key = 'units'
behav_dat_key = 'trials_behv'
lfp_key = 'lfps'

pre_trial_dur = 0.5
post_trial_dur = 0.5
exp_data = data_handler(dat, behav_dat_key, spike_key, lfp_key, behav_stat_key, pre_trial_dur=pre_trial_dur, extract_fly_and_monkey_xy=True,
                        post_trial_dur=post_trial_dur,
                        lfp_beta=None, lfp_alpha=None, extract_lfp_phase=True)

exp_data.set_filters('all', True)

# rebin to 0.1 sec
ts = exp_data.rebin_time_stamps(0.1)
t_targ = dict_to_vec(exp_data.behav.events.t_targ)
t_start = t_targ
t_stop = dict_to_vec(exp_data.behav.events.t_stop)

# concatenate a cuple of variables with the 0.2 binning
var_names = 'rad_vel', 'ang_vel', 'x_monk', 'y_monk'  # ,'t_move'
y, X, trial_idx = exp_data.concatenate_inputs(
    *var_names, t_start=t_start, t_stop=t_stop, time_stamps=ts)


# generate ts for a trial
id=np.unique(trial_idx)[0]
ts_=[]
for id in np.unique(trial_idx):
    trialts= ts[id]
    start = t_start[id]
    stop = t_stop[id]
    trialts = trialts[(trialts>=start) & (trialts<stop) ]
    ts_.append(trialts)
cutted_ts=np.hstack(ts_)



# reconstruct belief ----------------------------------------
warnings.filterwarnings('ignore')
torch.manual_seed(42)
arg = Config()

print('loading data')
datapath = Path(resdir/"neuraltest/1208pack")
with open(datapath, 'rb') as f:
    states, actions, tasks = pickle.load(f)

env = ffacc_real.FireFlyPaper2(arg)
env.debug = 1
phi = torch.tensor([[0.4],
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
agent_ = TD3.load('trained_agent/paper.zip')
agent = agent_.actor.mu.cpu()

invfile = Path(resdir/'neuraltest/inv_schroall_constrain_nopert_part2')
finaltheta, finalcov, err = process_inv(
    invfile, removegr=False, usingbest=False)

# run the agent
beliefs, covs = [], []
ntrial = 1
theta = finaltheta
removemask = []
for ind in range(len(tasks)):
    print(ind)
    if len(actions[ind]) < 5:
        removemask.append(list(set(trial_idx))[ind])
    else:
        _, _, ep_beliefs, ep_covs = run_trials(agent=agent, env=env, phi=phi, theta=theta, task=tasks[ind], ntrials=ntrial,
                                               pert=None, given_obs=None, return_belief=True, given_action=actions[ind], given_state=states[ind])
        beliefs.append(ep_beliefs[0]-ep_beliefs[0][0])
        covs.append(ep_covs[0])
        assert len(ep_beliefs[0]) == len(actions[ind])


# mask the invalid trial
mask = [True if i not in removemask else False for i in trial_idx]
pos_xy = np.hstack((X['x_monk'].reshape(-1,1), X['y_monk'].reshape(-1,1)))[mask,:] # ts, xy
# mask the nan value in state
non_nan = ~np.isnan(pos_xy.sum(axis=1))

# res, add existing variables
res={k:v[mask][non_nan] for k,v in X.items()}
res['y']=y[:, mask].T[non_nan]
res['trial_idx']=trial_idx[mask][non_nan]
res['ts']=cutted_ts[mask][non_nan]


# model unit to world unit, rotate model to world
b= np.vstack(beliefs)[:, [1,0,2,3,4], 0][non_nan].T # 1st step of rotation ccw 90: flip x and y 
b[[0,1,3]]=b[[0,1,3]]*500
b[[2,4]]=b[[2,4]]*180/pi
b[0]=b[0]*-1 # 2nd step of rotation ccw 90: flip x sign
b=b.T

covs=np.vstack(covs)
covs=covs[non_nan]
# rotate cov 90 degree and x500
R=np.array([[np.cos(-pi/2),-np.sin(-pi/2)],[np.sin(-pi/2),np.cos(-pi/2)]])*500
covs[:,:2,:2]=R.T@covs[:,:2,:2]@R
# v unit x500
covs[:,3,3]=covs[:,3,3]*500*500
# w unit 180/pi
covs[:,[2,4], [2,4]]=covs[:,[2,4], [2,4]]*180/pi*180/pi
res['covs']=covs

for id in np.unique(res['trial_idx']):
    # zero belief initial position
    b[res['trial_idx']==id][:,:2]=b[res['trial_idx']==id,:2]-b[res['trial_idx']==id][0,:2]
    # zero state inital position
    res['x_monk'][res['trial_idx']==id] = res['x_monk'][res['trial_idx']==id]-res['x_monk'][res['trial_idx']==id][0]
    res['y_monk'][res['trial_idx']==id] = res['y_monk'][res['trial_idx']==id]-res['y_monk'][res['trial_idx']==id][0]
    
res['belief']=b
    


resdir.mkdir(parents=True, exist_ok=True)
resdir = Path(resdir)/'neuraltest/res'

with open(resdir/'m53s31_0223newformatbelief', 'wb+') as f:
    pickle.dump(res, f)


# post check
# i=0
# i+=1
# id=np.unique(res['trial_idx'])[i]

# trialb=b[res['trial_idx']==id][:,:2]
# trialx=res['x_monk'][res['trial_idx']==id]
# trialy=res['y_monk'][res['trial_idx']==id]

# plt.plot(trialb[:,0], trialb[:,1])
# plt.plot(trialx, trialy)
# plt.axis('equal')

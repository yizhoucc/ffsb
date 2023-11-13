# sys.path.append(os.path.abspath('.'))
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
import os
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
# resdir = 'Z:/neuraltest/res'
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


# from data handler, load the data and cut ----------------------------------
print('start loading...')
dat = loadmat(resdir/'neuraltest/m53s31.mat')

# lfp_beta = loadmat('/Volumes/TOSHIBA EXT/dataset_firefly/lfp_beta_m53s50.mat')
# lfp_alpha = loadmat('/Volumes/TOSHIBA EXT/dataset_firefly/lfp_alpha_m53s50.mat')
# lfp_theta = loadmat('/Volumes/TOSHIBA EXT/dataset_firefly/lfp_theta_m53s50.mat')
print(dat.keys())
behav_stat_key = 'behv_stats'
spike_key = 'units'
behav_dat_key = 'trials_behv'
lfp_key = 'lfps'

pre_trial_dur = 0.5
post_trial_dur = 0.5
# exp_data = data_handler(dat,behav_dat_key,spike_key,lfp_key,behav_stat_key,pre_trial_dur=pre_trial_dur,post_trial_dur=post_trial_dur,
#                         lfp_beta=lfp_beta['lfp_beta'],lfp_alpha=lfp_alpha['lfp_alpha'],extract_lfp_phase=True)
exp_data = data_handler(dat, behav_dat_key, spike_key, lfp_key, behav_stat_key, pre_trial_dur=pre_trial_dur, extract_fly_and_monkey_xy=True,
                        post_trial_dur=post_trial_dur,
                        lfp_beta=None, lfp_alpha=None, extract_lfp_phase=True, dt=0.1)

exp_data.set_filters('all', True)

# rebin to 0.1 sec
ts = exp_data.rebin_time_stamps(0.1)
# ts=None
# select the stat/stop trial
# t_targ = dict_to_vec(exp_data.behav.events.t_targ)+0.3
t_targ = dict_to_vec(exp_data.behav.events.t_targ)
# t_move = dict_to_vec(exp_data.behav.events.t_move)
# t_start = np.min(np.vstack((t_targ)),axis=0) - pre_trial_dur
t_start = t_targ
t_stop = dict_to_vec(exp_data.behav.events.t_stop)

# concatenate a cuple of variables with the 0.2 binning
var_names = 'rad_vel', 'ang_vel', 'x_monk', 'y_monk'  # ,'t_move'
y, X, trial_idx = exp_data.concatenate_inputs(
    *var_names, t_start=t_start, t_stop=t_stop, time_stamps=ts)

len(ts)
len(t_start)
len(t_stop)
len(ts)
y.shape
len(np.unique(trial_idx))



# datapath=resdir/"neuraltest/1208pack"
# with open(datapath,'rb') as f:
#     states, actions, tasks = pickle.load(f)
    

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
agent_ = TD3.load(resdir/'trained_agent/paper')
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


# remove invalid inds
mask = [True if i not in removemask else False for i in trial_idx]

res={k:v[mask] for k,v in X.items()}
res['y']=y[:, mask].T
res['trial_idx']=trial_idx[mask]
b= np.vstack(beliefs)[:, :, 0].T
res['cov']= np.vstack(covs)

resdir.mkdir(parents=True, exist_ok=True)
resdir = Path(resdir)/'neuraltest/res'

b[[0,1,3]]=b[[0,1,3]]*500
b[[2,4]]=b[[2,4]]*180/pi
res['belief']=b.T
res['mask']=mask



with open(resdir/'0928collapsemodelbelief', 'wb+') as f:
    pickle.dump(res, f)



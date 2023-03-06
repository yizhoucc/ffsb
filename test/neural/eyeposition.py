
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

def eyepos2flypos_(beta_l,beta_r,alpha_l,alpha_r,z):
    '''
    beta_l: left eye elevation
    beta_r: right eye elevation
    alpha_l: left eye version
    alpha_r: right eye version
    '''
    beta = 0.5*(beta_l + beta_r)
    alpha = 0.5*(alpha_l + alpha_r)
    x_squared = z^2*[(np.tan(alpha)**2)/(np.tan(beta)**2)]*[(1 + np.tan(beta)**2)/(1 + np.tan(alpha)**2)]
    y_squared = (z^2)/(np.tan(beta)**2) - x_squared
   
    r = (x_squared + y_squared)**0.5
    theta = alpha
   
    return r, theta

def eyepos2flypos(hor_theta,ver_theta,agent_height=10):

    gaze_r = agent_height / np.tan(ver_theta)
    gaze_x =  gaze_r * np.cos( hor_theta)
    gaze_y =  gaze_r * np.sin( hor_theta)

    return gaze_x,gaze_y

def world2screen(x_rel,y_rel,height=10,screen_dist=32.5 ):
    screen_z = height - screen_dist * height / y_rel
    screen_x = screen_dist * x_rel / y_rel

    screen_z[y_rel <= 0] = np.nan
    screen_x[y_rel <= 0] = np.nan

    return screen_x,screen_z

def screen2world(screen_x,screen_z, height=10,screen_dist=32.5):
    y_rel=(screen_dist*height)/(height-screen_z)
    x_rel=screen_x*y_rel/screen_dist
    y_rel[height-screen_z <= 0] = np.nan
    return x_rel, y_rel

def world2mk(monkeyx,monkeyy, w,x_fly, y_fly,dt=0.1):
    x_fly_rel =x_fly - monkeyx
    y_fly_rel = y_fly - monkeyy
    phi = dt * np.cumsum(w)
    R = lambda theta : np.array([[np.cos(theta/180*np.pi),-np.sin(theta/180*np.pi)],[np.sin(theta/180*np.pi),np.cos(theta/180*np.pi)]])
    XY = np.zeros((2,x_fly_rel.shape[0]))
    XY[0,:] = x_fly_rel
    XY[1,:] = y_fly_rel
    rot = R(phi)
    XY = np.einsum('ijk,jk->ik', rot, XY)
    xfp_rel= XY[0, :]
    yfp_rel = XY[1, :]
    return xfp_rel,yfp_rel



datafiles=list((resdir/"neural").glob("*.mat"))
ind=1
# dat = loadmat(datafiles[ind])
dat = loadmat(resdir/'neuraltest/m53s31.mat')

# dat['trials_behv']['continuous'][0].zle
# dat['trials_behv']['continuous'][0].zre
# dat['trials_behv']['continuous'][0].yle
# dat['prs'][0]['height']
# dat['trials_behv']['continuous'].shape
# dat['trials_behv']['continuous'][0].shape
# dat['trials_behv']['continuous'][0][0].shape
# dat['trials_behv']['continuous'][0][0][0].shape
# dat['trials_behv']['continuous'][0][0][0]['zle'].shape
# dat['trials_behv']['continuous'][0][0][0]['zle'][0].shape
# dat['trials_behv']['continuous'][0][0]['zle'][0].shape
# dat['trials_behv']['continuous'][0][0][0][0]['zle'].shape
# plt.plot(dat['trials_behv']['continuous'][0][0][0][0]['zle'])
# continuous=dat['trials_behv']['continuous']
# prs=dat['prs']
# i=0
# r_belief,theta_belief= eyepos2flypos(continuous[i].zle,continuous[i].zre,continuous[i].yle,continuous[i].yre,prs.height)


# print(dat.keys())
behav_stat_key = 'behv_stats'
spike_key = 'units'
behav_dat_key = 'trials_behv'
lfp_key = 'lfps'

pre_trial_dur = 0.5
post_trial_dur = 0.5
exp_data = data_handler(dat, behav_dat_key, spike_key, lfp_key, behav_stat_key, pre_trial_dur=pre_trial_dur, extract_fly_and_monkey_xy=True,
                        post_trial_dur=post_trial_dur,extract_cartesian_eye_and_firefly=True,
                        lfp_beta=None, lfp_alpha=None, extract_lfp_phase=True)


# list(exp_data.behav.continuous.__dict__.keys())
exp_data.set_filters('all', True)
# rebin to 0.1 sec
ts = exp_data.rebin_time_stamps(0.1)
t_targ=dict_to_vec(exp_data.behav.events.t_targ)
t_start=t_targ
t_stop = dict_to_vec(exp_data.behav.events.t_stop)
var_names = 'rad_vel','ang_vel','x_monk','y_monk', 'eye_hori', 'eye_vert','x_fly_screen','z_fly_screen','x_eye_screen','z_eye_screen',"x_fly_rel","y_fly_rel","ang_vel"
y,X,trial_idx = exp_data.concatenate_inputs(*var_names,t_start=t_start,t_stop=t_stop, time_stamps=ts)


trials=np.unique(trial_idx)

itrial=0
itrial+=1


# # overhead
# with initiate_plot(2,2,200) as fig:
#     ax=fig.add_subplot(111)
#     plt.plot(X['x_monk'][trial_idx==trials[itrial]], X['y_monk'][trial_idx==trials[itrial]],label='monkey path')
#     plt.plot(X['x_fly_rel'][trial_idx==trials[itrial]], X['y_fly_rel'][trial_idx==trials[itrial]]-32.5,label='relative firefly path')
#     plt.scatter(exp_data.behav.continuous.x_fly[trials[itrial]], exp_data.behav.continuous.y_fly[trials[itrial]],label='target')
#     plt.axis('equal')
#     quickspine(ax)
#     ax.set_xlabel('world x, cm')
#     ax.set_ylabel('world y, cm')
#     quickleg(ax)
#     # quicksave('example overhead mk position relative itrial{}'.format(itrial))

# # world 2 screen, firefly position
# with initiate_plot(2,2,200) as fig:
#     ax=fig.add_subplot(111)
#     xx,yy=world2screen(X['x_fly_rel'][trial_idx==trials[itrial]], X['y_fly_rel'][trial_idx==trials[itrial]])
#     plt.plot(xx+3, yy, label='firefly position,shifted for vis')
#     plt.plot(X['x_fly_screen'][trial_idx==trials[itrial]], X['z_fly_screen'][trial_idx==trials[itrial]], label='reconstructed firefly on screen')
#     plt.axis('equal')
#     quickspine(ax)
#     ax.set_xlabel('screen x, cm')
#     ax.set_ylabel('screen z, cm')
#     quickleg(ax)
#     # quicksave('example screen mk position relative itrial{}'.format(itrial))

# # screen 2 world, firefly position
# with initiate_plot(2,2,200) as fig:
#     ax=fig.add_subplot(111)
#     xx,yy=screen2world(X['x_fly_screen'][trial_idx==trials[itrial]], X['z_fly_screen'][trial_idx==trials[itrial]])
#     plt.plot(xx+9, yy, label='reconstructed firefly position,shifted for vis')
#     plt.plot(X['x_fly_rel'][trial_idx==trials[itrial]], X['y_fly_rel'][trial_idx==trials[itrial]], label='recorded firefly in world')
#     plt.axis('equal')
#     quickspine(ax)
#     ax.set_xlabel('world x, cm')
#     ax.set_ylabel('world y, cm')
#     quickleg(ax)
#     # quicksave('example reconstructed world mk position relative itrial{}'.format(itrial))

#  eye position on screen
with initiate_plot(2,2,200) as fig:
    ax=fig.add_subplot(111)
    plt.plot(X['x_fly_screen'][trial_idx==trials[itrial]], X['z_fly_screen'][trial_idx==trials[itrial]], label='firefly on screen')
    plt.plot(X['x_eye_screen'][trial_idx==trials[itrial]], X['z_eye_screen'][trial_idx==trials[itrial]], label='eye focus on screen', color='red')
    plt.axis('equal')
    quickspine(ax)
    ax.set_xlabel('screen x, cm')
    ax.set_ylabel('screen z, cm')
    quickleg(ax)
    # quicksave('example eye position on screen relative itrial{}'.format(itrial))

#  eye position in world
with initiate_plot(2,2,200) as fig:
    ax=fig.add_subplot(111)
    plt.plot(X['x_fly_rel'][trial_idx==trials[itrial]], X['y_fly_rel'][trial_idx==trials[itrial]], label='recorded firefly in world')
    xx,yy=screen2world(X['x_eye_screen'][trial_idx==trials[itrial]], X['z_eye_screen'][trial_idx==trials[itrial]])
    plt.plot(xx, yy, label='reconstructed eye focus in world',color='red')
    plt.axis('equal')
    quickspine(ax)
    ax.set_xlabel('world x, cm')
    ax.set_ylabel('world y, cm')
    plt.xlim(-444,444)
    plt.ylim(0,666)
    quickleg(ax)
    # quicksave('example eye position in world relative itrial{}'.format(itrial))


# reconstruct relative path
with initiate_plot(2,2,200) as fig:
    ax=fig.add_subplot(111)
    plt.plot(X['x_monk'][trial_idx==trials[itrial]], X['y_monk'][trial_idx==trials[itrial]],label='monkey path')
    plt.plot(X['x_fly_rel'][trial_idx==trials[itrial]], X['y_fly_rel'][trial_idx==trials[itrial]]-32.5,label='relative firefly path')
    
    xr, yr=world2mk(X['x_monk'][trial_idx==trials[itrial]],X['y_monk'][trial_idx==trials[itrial]],X['ang_vel'][trial_idx==trials[itrial]],exp_data.behav.continuous.x_fly[trials[itrial]], exp_data.behav.continuous.y_fly[trials[itrial]])
    plt.plot(xr, yr-32.5,label='reconstructed relative firefly path')

    plt.scatter(exp_data.behav.continuous.x_fly[trials[itrial]], exp_data.behav.continuous.y_fly[trials[itrial]],label='target')
    plt.axis('equal')
    quickspine(ax)
    ax.set_xlabel('world x, cm')
    ax.set_ylabel('world y, cm')
    quickleg(ax)
    # quicksave('example overhead mk position relative itrial{}'.format(itrial))


# fitting check  ------------------
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import numpy as np
import pickle
from pathlib import Path
import os
import sys
import random
import configparser
from plot_ult import *
import scipy.interpolate as interpolate
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def splineDesign(knots, x, ord=4, der=0, outer_ok=False):
    """Reproduces behavior of R function splineDesign() for use by ns(). See R documentation for more information.

    Python code uses scipy.interpolate.splev to get B-spline basis functions, while R code calls C.
    Note that der is the same across x."""
    knots = np.array(knots, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    xorig = x.copy()
    not_nan = ~np.isnan(xorig)
    nx = x.shape[0]
    knots.sort()
    nk = knots.shape[0]
    need_outer = any(x[not_nan] < knots[ord - 1]) or any(x[not_nan] > knots[nk - ord])
    in_x = (x >= knots[0]) & (x <= knots[-1]) & not_nan

    if need_outer:
        if outer_ok:
            # print('knots do not contain the data range')

            out_x = ~all(in_x)
            if out_x:
                x = x[in_x]
                nnx = x.shape[0]
            dkn = np.diff(knots)[::-1]
            reps_start = ord - 1
            if any(dkn > 0):
                reps_end = max(0, ord - np.where(dkn > 0)[0][0] - 1)
            else:
                reps_end = np.nan  # this should give an error, since all knots are the same
            idx = [0] * (ord - 1) + list(range(nk)) + [nk - 1] * reps_end
            knots = knots[idx]
        else:
            raise ValueError("the 'x' data must be in the range %f to %f unless you set outer_ok==True'" % (
            knots[ord - 1], knots[nk - ord]))
    else:
        reps_start = 0
        reps_end = 0
    if (not need_outer) and any(~not_nan):
        x = x[in_x]
    idx0 = reps_start
    idx1 = len(knots) - ord - reps_end
    cycleOver = np.arange(idx0, idx1)
    m = len(knots) - ord
    v = np.zeros((cycleOver.shape[0], len(x)), dtype=np.float64)
    # v = np.zeros((m, len(x)))

    d = np.eye(m, len(knots))
    for i in range(cycleOver.shape[0]):
        v[i] = interpolate.splev(x, (knots, d[cycleOver[i]], ord - 1), der=der)
        # v[i] = interpolate.splev(x, (knots, d[i], ord - 1), der=der)

    # before = np.sum(xorig[not_nan] < knots[0])
    # after = np.sum(xorig[not_nan] > knots[-1])
    design = np.zeros((v.shape[0], xorig.shape[0]), dtype=np.float64)
    for i in range(v.shape[0]):
    #     design[i, before:xorig.shape[0] - after] = v[i]
        design[i,in_x] = v[i]


    return design.transpose()

def convolve_neuron(spk, trial_idx, bX):
    kernel_len  = bX.shape[1]
    agument_spk_size = spk.shape[0] + 2 * kernel_len * np.unique(trial_idx).shape[0]
    agument_spk = np.zeros(agument_spk_size)
    reverse = np.zeros(agument_spk.shape[0],dtype=bool)
    cc = 0
    for tr in np.unique(trial_idx):
        sel = trial_idx == tr
        agument_spk[cc:cc+sel.sum()] = spk[sel]
        reverse[cc:cc+sel.sum()] = True
        cc += sel.sum() + 2 * kernel_len
    modelX = np.zeros((spk.shape[0],bX.shape[1]))
    for k in range(bX.shape[1]):
        xsm = np.convolve(agument_spk, bX[:,k],'same')
        modelX[:,k] = xsm[reverse]
    return modelX

def convolve_loop(spks, trial_idx, bX):
    modelX = np.zeros((spks.shape[1], spks.shape[0]*bX.shape[1]))
    cc = 0
    for neu in range(spks.shape[0]):
        print(neu)
        modelX[:,cc:cc+bX.shape[1]] = convolve_neuron(spks[neu], trial_idx, bX)
        cc += bX.shape[1]
    return modelX

def splitdata(s, mask=None):
    n = len(s)
    if mask is not None:
        pass
    else:
        mask = np.array(random.sample(range(0, n), n//10*9))

    negmask = np.zeros(len(s), dtype=bool)
    negmask[mask] = True
    train = s[mask]
    test = s[negmask]
    return train, test, mask

# load belief and neural data -------------------
config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
resdir=Path(config['Datafolder']['data'])
dat = loadmat(resdir/'neuraltest/m53s31.mat')
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


with open(resdir/'neuraltest/res/m53s31_0223newformatbelief', 'rb') as f:
    res = pickle.load(f)

y_ = res['y']
X = {k: res[k] for k in ['rad_vel', 'ang_vel', 'x_monk', 'y_monk']}
trial_idx = res['trial_idx']
beliefs = res['belief']
covs = res['covs']
s = np.vstack([v for v in X.values()])
s = s.T


trials=np.unique(trial_idx)

itrial=0
itrial+=1

# reconstruct relative belief path
with initiate_plot(2,2,200) as fig:
    ax=fig.add_subplot(111)
    # plt.plot(X['x_monk'][trial_idx==trials[itrial]], X['y_monk'][trial_idx==trials[itrial]],label='monkey path')
    # plt.plot(X['x_fly_rel'][trial_idx==trials[itrial]], X['y_fly_rel'][trial_idx==trials[itrial]]-32.5,label='relative firefly path')
    
    xr, yr=world2mk(X['x_monk'][trial_idx==trials[itrial]],X['y_monk'][trial_idx==trials[itrial]],X['ang_vel'][trial_idx==trials[itrial]],exp_data.behav.continuous.x_fly[trials[itrial]], exp_data.behav.continuous.y_fly[trials[itrial]])
    plt.plot(xr, yr-32.5,label='reconstructed relative firefly path')

    xr, yr=world2mk(beliefs[trial_idx==trials[itrial]][:,0],beliefs[trial_idx==trials[itrial]][:,1],X['ang_vel'][trial_idx==trials[itrial]],exp_data.behav.continuous.x_fly[trials[itrial]], exp_data.behav.continuous.y_fly[trials[itrial]])
    plt.plot(xr, yr-32.5,label='reconstructed relative belief firefly path', color='red')

    plt.scatter(exp_data.behav.continuous.x_fly[trials[itrial]], exp_data.behav.continuous.y_fly[trials[itrial]],label='target')
    plt.axis('equal')
    quickspine(ax)
    ax.set_xlabel('world x, cm')
    ax.set_ylabel('world y, cm')
    quickleg(ax)
    # quicksave('example overhead mk position relative itrial{}'.format(itrial))



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

# world centric states
linreg = linear_model.LinearRegression()
linreg.fit(modelX, pos_xy)
print('linear regression score',linreg.score(modelX, pos_xy))
pred  = linreg.predict(modelX)
every=5
with initiate_plot(5, 3, 200) as f:
    ax=f.add_subplot(121)
    plt.scatter(pred[::every,1],pos_xy[::every,1],s=1, alpha=0.3)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('world pos x')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    ax=f.add_subplot(122)
    plt.scatter(pred[::every,0],pos_xy[::every,0],s=1, alpha=0.3)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('world pos y')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    plt.tight_layout()

# baseline, world centric
linreg = linear_model.LinearRegression()
linreg.fit(modelX, belief_xy)
print('linear regression score',linreg.score(modelX, belief_xy))
pred  = linreg.predict(modelX)
every=5
with initiate_plot(5, 3, 200) as f:
    ax=f.add_subplot(121)
    plt.scatter(pred[::every,1],belief_xy[::every,1],s=1, alpha=0.3)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('belief world pos x')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    ax=f.add_subplot(122)
    plt.scatter(pred[::every,0],belief_xy[::every,0],s=1, alpha=0.3)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('belief world pos y')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    plt.tight_layout()



# mk centric, states
linreg = linear_model.LinearRegression()
linreg.fit(modelX, states_rel)
print('linear regression score',linreg.score(modelX, states_rel))
pred  = linreg.predict(modelX)
every=5
with initiate_plot(5, 3, 200) as f:
    ax=f.add_subplot(121)
    plt.scatter(pred[::every,1],states_rel[::every,1],s=1, alpha=0.3)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff pos x')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    ax=f.add_subplot(122)
    plt.scatter(pred[::every,0],states_rel[::every,0],s=1, alpha=0.3)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff pos y')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    plt.tight_layout()


# mk centric, belief
linreg = linear_model.LinearRegression()
linreg.fit(modelX, belief_rel)
print('linear regression score',linreg.score(modelX, belief_rel))
pred  = linreg.predict(modelX)
every=5
with initiate_plot(5, 3, 200) as f:
    ax=f.add_subplot(121)
    plt.scatter(pred[::every,1],belief_rel[::every,1],s=1, alpha=0.3)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff belief_rel pos x')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    ax=f.add_subplot(122)
    plt.scatter(pred[::every,0],belief_rel[::every,0],s=1, alpha=0.3)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff belief_rel pos y')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    plt.tight_layout()


# together
testcutoff=888
with initiate_plot(5, 3, 200) as f:
    linreg1 = linear_model.LinearRegression()
    linreg1.fit(modelX[:-testcutoff], states_rel[:-testcutoff])
    print('linear regression score',linreg1.score(modelX[-testcutoff:], states_rel[-testcutoff:]))
    pred1  = linreg1.predict(modelX[-testcutoff:])
    linreg2 = linear_model.LinearRegression()
    linreg2.fit(modelX[:-testcutoff], belief_rel[:-testcutoff])
    print('linear regression score',linreg2.score(modelX[-testcutoff:], belief_rel[-testcutoff:]))
    pred2  = linreg2.predict(modelX[-testcutoff:])

    ax=f.add_subplot(121)
    plt.scatter(pred1[:,1],states_rel[-testcutoff:,1],s=1, alpha=0.7, label='rel state')
    ax=f.add_subplot(122)
    plt.scatter(pred1[:,0],states_rel[-testcutoff:,0],s=1, alpha=0.7, label='rel state')

    ax=f.add_subplot(121)
    plt.scatter(pred2[:,1],belief_rel[-testcutoff:,1],s=1, alpha=0.7, label='rel belief')
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff rel pos x')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    ax=f.add_subplot(122)
    plt.scatter(pred2[:,0],belief_rel[-testcutoff:,0],s=1, alpha=0.7, label='rel belief')
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff rel pos y')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    quickleg(ax)


# later half of the trial
firsthalftrialidx=np.zeros_like(trial_idx)
for itrial in trials:
    s=np.min(np.argwhere(trial_idx==itrial))
    e=s+len(trial_idx[trial_idx==itrial])//2
    firsthalftrialidx[s:e]+=1
latterhalftrialidx=1-firsthalftrialidx
latterhalftrialidx, firsthalftrialidx = latterhalftrialidx.astype('bool'), firsthalftrialidx.astype('bool')
with initiate_plot(5, 5, 200) as f:
    linreg2 = linear_model.LinearRegression()
    linreg2.fit(modelX, belief_rel)
    print('linear regression score',linreg2.score(modelX, belief_rel))
    pred2  = linreg2.predict(modelX)

    ax=f.add_subplot(121)
    plt.scatter(pred2[latterhalftrialidx,1],belief_rel[latterhalftrialidx,1],s=1, alpha=0.3, label='later half rel belief')
    ax=f.add_subplot(122)
    plt.scatter(pred2[latterhalftrialidx,0],belief_rel[latterhalftrialidx,0],s=1, alpha=0.3, label='later half rel belief')

    ax=f.add_subplot(121)
    plt.scatter(pred2[firsthalftrialidx,1],belief_rel[firsthalftrialidx,1],s=1, alpha=0.3, label='first half rel belief')
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff rel pos x')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    ax=f.add_subplot(122)
    plt.scatter(pred2[firsthalftrialidx,0],belief_rel[firsthalftrialidx,0],s=1, alpha=0.3, label='first half rel belief')
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff rel pos y')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    quickleg(ax)



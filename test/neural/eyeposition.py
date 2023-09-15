
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
from scipy.io import loadmat
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



config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
resdir=config['Datafolder']['data']
resdir = Path(resdir)




datafiles=list((resdir/"neural").glob("*.mat"))
ind=1
# dat = loadmat(datafiles[ind])
# dat = loadmat(resdir/'neuraltest/m53s31.mat') # this is a example session with good recording, more neurons
dat = loadmat(resdir/'neural/m53s36.mat') # this is a pert example
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
len(trials)

itrial=0

itrial+=1
# verify, relative position overhead
with initiate_plot(2,2,200) as fig:
    ax=fig.add_subplot(111)
    plt.plot(X['x_monk'][trial_idx==trials[itrial]], X['y_monk'][trial_idx==trials[itrial]],label='monkey path')
    plt.plot(X['x_fly_rel'][trial_idx==trials[itrial]], X['y_fly_rel'][trial_idx==trials[itrial]]-32.5,label='relative firefly path')
    plt.scatter(exp_data.behav.continuous.x_fly[trials[itrial]], exp_data.behav.continuous.y_fly[trials[itrial]],label='target')
    plt.axis('equal')
    quickspine(ax)
    ax.set_xlabel('world x, cm')
    ax.set_ylabel('world y, cm')
    quickleg(ax)
    # quicksave('example overhead mk position relative itrial{}'.format(itrial))

# verify, world 2 screen, firefly position
with initiate_plot(2,2,200) as fig:
    ax=fig.add_subplot(111)
    xx,yy=world2screen(X['x_fly_rel'][trial_idx==trials[itrial]], X['y_fly_rel'][trial_idx==trials[itrial]])
    plt.plot(xx+3, yy, label='firefly position,shifted for vis')
    plt.plot(X['x_fly_screen'][trial_idx==trials[itrial]], X['z_fly_screen'][trial_idx==trials[itrial]], label='reconstructed firefly on screen')
    plt.axis('equal')
    quickspine(ax)
    ax.set_xlabel('screen x, cm')
    ax.set_ylabel('screen z, cm')
    quickleg(ax)
    # quicksave('example screen mk position relative itrial{}'.format(itrial))

# verity, screen 2 world, firefly position
with initiate_plot(2,2,200) as fig:
    ax=fig.add_subplot(111)
    xx,yy=screen2world(X['x_fly_screen'][trial_idx==trials[itrial]], X['z_fly_screen'][trial_idx==trials[itrial]])
    plt.plot(xx+9, yy, label='reconstructed firefly position,shifted for vis')
    plt.plot(X['x_fly_rel'][trial_idx==trials[itrial]], X['y_fly_rel'][trial_idx==trials[itrial]], label='recorded firefly in world')
    plt.axis('equal')
    quickspine(ax)
    ax.set_xlabel('world x, cm')
    ax.set_ylabel('world y, cm')
    quickleg(ax)
    # quicksave('example reconstructed world mk position relative itrial{}'.format(itrial))

itrial+=1
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

# check correlation to make sure sign is right
plt.scatter((X['x_fly_screen']), (X['x_eye_screen']))
plt.xlim(-2222,2222)
plt.xlabel('x fly')
plt.ylabel('x eye')
# plt.ylim(-100, 100)
plt.axis('equal')
plt.show()


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




gaze_r,gaze_x,gaze_y=get_gaze_location(X['eye_vert'],X['eye_hori'],np.cumsum(X['ang_vel']),X['x_monk'],X['y_monk'])

def normalizematrix(data,low=5,high=95):
    themin=np.percentile(data[~np.isnan(data)],low)
    themax=np.percentile(data[~np.isnan(data)],high)
    res= (data - themin) / (themax- themin)
    res[np.isnan(data)]=np.nan
    res[data>themax]=np.nan
    res[data>themin]=np.nan
    return res


plt.scatter(gaze_x,gaze_y,s=0.5)
plt.xlabel('gaze_x');plt.ylabel('gaze_y')

plt.scatter(gaze_x,gaze_r,s=0.5)
plt.xlabel('gaze_x');plt.ylabel('gaze_r')

plt.scatter(gaze_r,gaze_y,s=0.5)
plt.xlabel('gaze_r');plt.ylabel('gaze_y')


plt.scatter(X['x_fly_rel'],gaze_y,s=0.1)
plt.xlabel('x_fly_relative');plt.ylabel('gaze_x')

plt.scatter(X['y_fly_rel'],gaze_x,s=0.1)
plt.xlabel('y_fly_relative');plt.ylabel('gaze_y')
plt.axis('equal')


plt.scatter(X['x_fly_rel'],X['y_fly_rel'],s=0.5)


plt.scatter(gaze_x,X['x_fly_rel'],s=0.5)
plt.scatter(gaze_y,X['y_fly_rel'],s=0.5)











# fitting check  ------------------

# load belief and neural data -------------------
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

yes=0
total=0
for itrial in range(len(trials)):
    if len(X['x_fly_rel'][trial_idx==trials[itrial]])<1:
        continue
    d=(X['x_fly_rel'][trial_idx==trials[itrial]][-1]**2 + X['y_fly_rel'][trial_idx==trials[itrial]][-1]**2)**0.5
    if d < 65: yes+=1
    total+=1
print('session success rate is ', yes/total)


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
# check, reconstruct relative belief path vs state path
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


# relative polar coord
def cordxy2pol(xr,yr,heading):
    # return distance, angle
    angle=np.arctan(xr,yr)*180/pi
    angle_rel=angle-heading
    distance=(xr**2 +yr**2)**0.5
    return distance, angle_rel

# # testing the relative heading angle
# x, y=(beliefs[trial_idx==trials[itrial]][:,0],beliefs[trial_idx==trials[itrial]][:,1])
# plt.plot(x, y,label='reconstructed relative belief firefly path', color='blue')
# plt.scatter(exp_data.behav.continuous.x_fly[trials[itrial]], exp_data.behav.continuous.y_fly[trials[itrial]],label='target')

# xr, yr=world2mk(beliefs[trial_idx==trials[itrial]][:,0],beliefs[trial_idx==trials[itrial]][:,1],X['ang_vel'][trial_idx==trials[itrial]],exp_data.behav.continuous.x_fly[trials[itrial]], exp_data.behav.continuous.y_fly[trials[itrial]])
# plt.plot(xr, yr-32.5,label='reconstructed relative belief firefly path', color='red')

# heading=beliefs[trial_idx==trials[itrial]][:,2]
# angle=np.arctan(xr,yr)*180/pi
# angle_rel=angle-heading
# plt.plot(heading)
# plt.plot(angle)
# plt.plot(angle_rel)


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


# world centric states
linreg = linear_model.LinearRegression()
linreg.fit(modelX, pos_xy)
print('linear regression score',linreg.score(modelX, pos_xy))
pred  = linreg.predict(modelX)
every=5
with initiate_plot(6, 3, 200) as f:
    ax=f.add_subplot(121)
    # plt.scatter(pred[::every,0],pos_xy[::every,0],s=1, alpha=0.3)
    thispred,thistrue=pred[::every,0],pos_xy[::every,0]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('world centeric x')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    quickspine(ax)
    ax=f.add_subplot(122)
    # plt.scatter(pred[::every,1],pos_xy[::every,1],s=1, alpha=0.3)
    thispred,thistrue=pred[::every,1],pos_xy[::every,1]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('world centeric y')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    quickspine(ax)
    plt.tight_layout()

# world centric belief
linreg = linear_model.LinearRegression()
linreg.fit(modelX, belief_xy)
print('linear regression score',linreg.score(modelX, belief_xy))
pred  = linreg.predict(modelX)
every=5
with initiate_plot(6, 3, 200) as f:
    ax=f.add_subplot(121)
    thispred,thistrue=pred[::every,0],belief_xy[::every,0]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('belief worldcentric x')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    quickspine(ax)
    ax=f.add_subplot(122)
    thispred,thistrue=pred[::every,1],belief_xy[::every,1]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('belief worldcentric y')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    quickspine(ax)
    plt.tight_layout()


# mk centric, states
linreg = linear_model.LinearRegression()
linreg.fit(modelX, states_rel)
print('linear regression score',linreg.score(modelX, states_rel))
pred  = linreg.predict(modelX)
every=5
with initiate_plot(6, 3, 200) as f:
    ax=f.add_subplot(121)
    # plt.scatter(pred[::every,1],states_rel[::every,1],s=1, alpha=0.3)
    thispred,thistrue=pred[::every,0],states_rel[::every,0]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('egocentric x')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')

    ax=f.add_subplot(122)
    # plt.scatter(pred[::every,0],states_rel[::every,0],s=1, alpha=0.3)
    thispred,thistrue=pred[::every,1],states_rel[::every,1]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('egocentric y')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    plt.tight_layout()


# mk centric, belief
linreg = linear_model.LinearRegression()
linreg.fit(modelX, belief_rel)
print('linear regression score',linreg.score(modelX, belief_rel))
pred  = linreg.predict(modelX)
every=5
with initiate_plot(6, 3, 200) as f:
    ax=f.add_subplot(121)
    thispred,thistrue=pred[::every,0],belief_rel[::every,0]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('belief egocentric x')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    quickspine(ax)
    ax=f.add_subplot(122)
    thispred,thistrue=pred[::every,1],belief_rel[::every,1]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('belief egocentric y')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    quickspine(ax)
    plt.tight_layout()

# mk centric, polar, belief
states_rel=np.hstack(states_rel).T
belief_heading = beliefs[:,2][non_nan]
belief_dist, belief_heading_rel=cordxy2pol(belief_rel[:,0],belief_rel[:,1],belief_heading)
belief_polar_rel=np.vstack([belief_dist, belief_heading_rel]).T

linreg = linear_model.LinearRegression()
linreg.fit(modelX, belief_polar_rel)
print('linear regression score',linreg.score(modelX, belief_polar_rel))
pred  = linreg.predict(modelX)
every=5
with initiate_plot(6, 3, 200) as f:
    ax=f.add_subplot(121)
    thispred,thistrue=pred[::every,0],belief_polar_rel[::every,0]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [cm]')
    plt.ylabel('true [cm]')
    plt.title('belief relative distance')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    quickspine(ax)
    ax=f.add_subplot(122)
    thispred,thistrue=pred[::every,1],belief_polar_rel[::every,1]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [degree]')
    plt.ylabel('true [degree]')
    plt.title('belief relative angle')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    quickspine(ax)
    plt.tight_layout()


# r--> belief heading (world centric heading)
belief_heading = beliefs[:,2][non_nan]

linreg = linear_model.LinearRegression()
linreg.fit(modelX, belief_heading)
print('linear regression score',linreg.score(modelX, belief_heading))
pred  = linreg.predict(modelX)
every=5
with initiate_plot(3, 3, 200) as f:
    ax=f.add_subplot(111)
    thispred,thistrue=pred[::every],belief_heading[::every]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [degree]')
    plt.ylabel('true [degree]')
    plt.title('belief heading direction')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    quickspine(ax)


# r --> time in each trial
trialtime=[]
fulltime=np.arange(0,7,0.1)
for itrial in range(len(trials)):
    thistime=len(X['x_monk'][trial_idx==trials[itrial]])
    trialtime.append(fulltime[:thistime])
trialtime=np.hstack(trialtime).T
len(trialtime)

linreg = linear_model.LinearRegression()
linreg.fit(modelX, trialtime)
print('linear regression score',linreg.score(modelX, trialtime))
pred  = linreg.predict(modelX)
every=5
with initiate_plot(3, 3, 200) as f:
    ax=f.add_subplot(111)
    thispred,thistrue=pred[::every],trialtime[::every]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [degree]')
    plt.ylabel('true [degree]')
    plt.title('time in a trial')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    quickspine(ax)


# r --> uncertainty diag in each trial
trialtime=[]
fulltime=np.arange(0,7,0.1)
for itrial in range(len(trials)):
    thistime=len(X['x_monk'][trial_idx==trials[itrial]])
    trialtime.append(fulltime[:thistime])
trialtime=np.hstack(trialtime).T
len(trialtime)

linreg = linear_model.LinearRegression()
linreg.fit(modelX, trialtime)
print('linear regression score',linreg.score(modelX, trialtime))
pred  = linreg.predict(modelX)
every=5
with initiate_plot(3, 3, 200) as f:
    ax=f.add_subplot(111)
    thispred,thistrue=pred[::every],trialtime[::every]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred [degree]')
    plt.ylabel('true [degree]')
    plt.title('time in a trial')
    vmin,vmax=limplot(ax)
    ax.plot([vmin,vmax],[vmin,vmax],'k')
    quickspine(ax)


































# together
testcutoff=888
linreg1 = linear_model.LinearRegression()
linreg1.fit(modelX[:-testcutoff], states_rel[:-testcutoff])
print('linear regression score',linreg1.score(modelX[-testcutoff:], states_rel[-testcutoff:]))
pred1  = linreg1.predict(modelX[-testcutoff:])
linreg2 = linear_model.LinearRegression()
linreg2.fit(modelX[:-testcutoff], belief_rel[:-testcutoff])
print('linear regression score',linreg2.score(modelX[-testcutoff:], belief_rel[-testcutoff:]))
pred2  = linreg2.predict(modelX[-testcutoff:])

with initiate_plot(5, 3, 200) as f:

    ax=f.add_subplot(121)
    # plt.scatter(pred1[:,1],states_rel[-testcutoff:,1],s=1, alpha=0.7, label='rel state')
    thispred,thistrue=pred1[:,1],states_rel[-testcutoff:,1]
    predvstrue1(thispred,thistrue,ax)

    ax=f.add_subplot(122)
    # plt.scatter(pred1[:,0],states_rel[-testcutoff:,0],s=1, alpha=0.7, label='rel state')
    thispred,thistrue=pred1[:,0],states_rel[-testcutoff:,0]
    predvstrue1(thispred,thistrue,ax)

    ax=f.add_subplot(121)
    # plt.scatter(pred2[:,1],belief_rel[-testcutoff:,1],s=1, alpha=0.7, label='rel belief')
    thispred,thistrue=pred2[:,1],belief_rel[-testcutoff:,1]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff rel pos x')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    ax=f.add_subplot(122)
    # plt.scatter(pred2[:,0],belief_rel[-testcutoff:,0],s=1, alpha=0.7, label='rel belief')
    thispred,thistrue=pred2[:,0],belief_rel[-testcutoff:,0]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff rel pos y')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    quickleg(ax)

print('really similar, cannot tell apart')


# later half of the trial
firsthalftrialidx=np.zeros_like(trial_idx)
for itrial in trials:
    s=np.min(np.argwhere(trial_idx==itrial))
    e=s+len(trial_idx[trial_idx==itrial])//2
    firsthalftrialidx[s:e]+=1
latterhalftrialidx=1-firsthalftrialidx
latterhalftrialidx, firsthalftrialidx = latterhalftrialidx.astype('bool'), firsthalftrialidx.astype('bool')
linreg2 = linear_model.LinearRegression()
linreg2.fit(modelX, belief_rel)
print('linear regression score',linreg2.score(modelX, belief_rel))
pred2  = linreg2.predict(modelX)

with initiate_plot(9, 4, 200) as f:
    ax=f.add_subplot(121)
    # plt.scatter(pred2[latterhalftrialidx,1],belief_rel[latterhalftrialidx,1],s=1, alpha=0.3, label='later half rel belief')
    thispred,thistrue=pred2[latterhalftrialidx,1],belief_rel[latterhalftrialidx,1]
    predvstrue1(thispred,thistrue,ax)
    # plt.scatter(pred2[firsthalftrialidx,1],belief_rel[firsthalftrialidx,1],s=1, alpha=0.3, label='first half rel belief')
    thispred,thistrue=pred2[firsthalftrialidx,1],belief_rel[firsthalftrialidx,1]
    predvstrue1(thispred,thistrue,ax)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff rel pos x \n blue: later half \n orange: first half')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)

    ax=f.add_subplot(122)
    # plt.scatter(pred2[latterhalftrialidx,0],belief_rel[latterhalftrialidx,0],s=1, alpha=0.3, label='later half rel belief')
    thispred,thistrue=pred2[latterhalftrialidx,0],belief_rel[latterhalftrialidx,0]
    predvstrue1(thispred,thistrue,ax)
    # plt.scatter(pred2[firsthalftrialidx,0],belief_rel[firsthalftrialidx,0],s=1, alpha=0.3, label='first half rel belief')
    thispred,thistrue=pred2[firsthalftrialidx,0],belief_rel[firsthalftrialidx,0]
    predvstrue1(thispred,thistrue,ax)

    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff rel pos y \n blue: later half \n orange: first half')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    # quickleg(ax)

# seperate
with initiate_plot(5, 5, 200) as f:
    linreg2 = linear_model.LinearRegression()
    linreg2.fit(modelX, belief_rel)
    print('linear regression score',linreg2.score(modelX, belief_rel))
    pred2  = linreg2.predict(modelX)

    ax1=f.add_subplot(221)
    plt.scatter(pred2[latterhalftrialidx,1],belief_rel[latterhalftrialidx,1],s=1, alpha=0.3, label='later half rel belief')
    quickspine(ax1)
    ax1.plot(ax1.get_ylim(),ax1.get_ylim(),'k')
    quickleg(ax1)

    ax2=f.add_subplot(222)
    plt.scatter(pred2[latterhalftrialidx,0],belief_rel[latterhalftrialidx,0],s=1, alpha=0.3, label='later half rel belief')
    quickspine(ax2)
    ax2.plot(ax2.get_ylim(),ax2.get_ylim(),'k')


    ax=f.add_subplot(223, sharex =ax1)
    plt.scatter(pred2[firsthalftrialidx,1],belief_rel[firsthalftrialidx,1],s=1, alpha=0.3, label='first half rel belief', color='orange')
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff rel pos x')
    ax.axis('equal')
    ax.plot(ax1.get_ylim(),ax1.get_ylim(),'k')
    quickspine(ax)
    ax=f.add_subplot(224, sharex =ax2)
    plt.scatter(pred2[firsthalftrialidx,0],belief_rel[firsthalftrialidx,0],s=1, alpha=0.3, label='first half rel belief',color='orange')
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('ff rel pos y')
    ax.axis('equal')
    ax.plot(ax2.get_ylim(),ax2.get_ylim(),'k')
    quickspine(ax)
    quickleg(ax)
    plt.tight_layout()







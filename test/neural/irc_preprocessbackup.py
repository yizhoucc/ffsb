
import sys
sys.path.append('..')
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
#import matplotlib.pyplot as plt
# import statsmodels.api as sm
from scipy.interpolate import interp1d


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
from copy import deepcopy
from scipy.io import loadmat
print('start loading...')
dat = loadmat('Z:/neuraltest/m53s31.mat')
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
                        lfp_beta=None, lfp_alpha=None, extract_lfp_phase=True)

exp_data.set_filters('all', True)

# rebin to 0.1 sec
ts = exp_data.rebin_time_stamps(0.1)
# ts=None
# select the stat/stop trial
# t_targ = dict_to_vec(exp_data.behav.events.t_targ)+0.3
t_targ=dict_to_vec(exp_data.behav.events.t_targ)
# t_move = dict_to_vec(exp_data.behav.events.t_move)
# t_start = np.min(np.vstack((t_targ)),axis=0) - pre_trial_dur
t_start=t_targ
t_stop = dict_to_vec(exp_data.behav.events.t_stop)

# concatenate a cuple of variables with the 0.2 binning
var_names = 'rad_vel','ang_vel','x_monk','y_monk'#,'t_move'
y,X,trial_idx = exp_data.concatenate_inputs(*var_names,t_start=t_start,t_stop=t_stop, time_stamps=ts)


def xy2pol(*args, rotation=True): 
    # return distance and angle. default rotated left 90 degree for the task
    x=args[0][0]; y=args[0][1]
    d = np.sqrt(x**2 + y**2)
    a = np.arctan2(y, x)+pi/2 if rotation else  np.arctan2(y, x)
    return d, a


def state_step2(px, py, heading, v, w,a, pro_gainv=1, pro_gainw=1,dt=0.006, userad=False): 
    if not userad:
        w=w/180*pi

    # overall, x'=Ax+Bu+noise. here, noise=0

    # use current v and w to update x y and heading 
    # (x'=Ax) part

    if v<=0:
        pass
    elif w==0:
        px = px + v*dt * np.cos(heading)
        py = py + v*dt * np.sin(heading)
    else:
        px = px-np.sin(heading)*(v/w-(v*np.cos(w*dt)/w))+np.cos(heading)*((v*np.sin(w*dt)/w))
        py = py+np.cos(heading)*(v/w-(v*np.cos(w*dt)/w))+np.sin(heading)*((v*np.sin(w*dt)/w))
    heading = heading + w*dt
    heading=np.clip(heading,-pi,pi)


    # apply the new control to state
    # (Bu) part
    v = pro_gainv *a[0]
    w = pro_gainw *a[1] 
    return px, py, heading, v, w

# appl for all trials --------------------------------------

alltrials=list(set(trial_idx))
ind=alltrials[np.random.randint(0,len(alltrials))]


states, actions, tasks = [], [], []
for ind in alltrials:

    ind=alltrials[np.random.randint(0,len(alltrials))]

    task=np.array([exp_data.behav.continuous.y_fly[ind]+32.5,-exp_data.behav.continuous.x_fly[ind]])/500
    tasks.append(task.astype('float32'))

    # get the actions
    action=np.array([X['rad_vel'][np.argwhere(trial_idx==ind)]/200,X['ang_vel'][np.argwhere(trial_idx==ind)]/180*pi*-1])[:,:,0].T
    actions.append(torch.tensor(action.astype('float32')))

    # run the actions to get states
    # px, py, heading, v, w = -32.5,0,0,0,0
    px, py, heading, v, w = 0,0,0,0,0
    log=[]
    for a in action:
        px, py, heading, v, w=state_step2(px, py, heading, v, w, a,pro_gainv=0.4, dt=0.1, userad=True)
        log.append([px, py, heading, v, w])
    log=np.array(log)
    states.append(torch.tensor(log.astype('float32')))
    
    print(len(action),len(log),len(trial_idx[trial_idx==ind]), len(X['y_monk'][trial_idx==ind]))






if __name__ == '__main__':
    # pack data
    import pickle
    with open('/data/neuraltest/1208pack', 'wb+') as handle:
        pickle.dump((states, actions, np.array(tasks)), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load and test
    datapath="Z:/neuraltest/1208pack"
    with open(datapath,'rb') as f:
        states, actions, tasks = pickle.load(f)

    ind=np.random.randint(0,len(tasks))
    state=states[ind]
    plt.plot(state[:,0], state[:,1])
    plt.axis('equal')
    task=tasks[ind]
    plt.scatter(task[0],task[1], label='goal')
    plt.legend()



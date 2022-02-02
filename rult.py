import numpy as np
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from contextlib import contextmanager
from numba import njit
from astropy.convolution import convolve
import warnings
import pandas as pd
from copy import deepcopy

import sys; sys.path.append('../model')
import Config as config

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def deepcopy_pd(df, is_df=False):
    func = pd.DataFrame if is_df else pd.Series
    return func(deepcopy(df.to_dict()))


def param_prior_range():
    param = {'target_theta': [55, 125], 'target_dist': [100, 400],
             'perturb_vpeak': [-200, 200], 'perturb_wpeak': [-120, 120],
             'perturb_start_time_ori': [0, 1], 'SAMPLING_RATE': 1 / 833}
    
    param['target_x'] = param['target_dist'][1] * np.cos(np.deg2rad(param['target_theta']))[::-1]
    param['target_y'] = np.array(param['target_dist']) * np.array([np.sin(np.deg2rad(param['target_theta'][0])), 1])
    return param


def min_max_scaling(data, d_range=None):
    d_range = [data.min(), data.max()] if d_range is None else d_range
    
    return (data - d_range[0]) / (d_range[1] - d_range[0])


def min_max_scale_df(df):
    scaled_data = []
    for variable, data in df.iteritems():
        scaled_data.append(min_max_scaling(data, d_range=param_prior_range()[variable]))
        
    return pd.concat(scaled_data, axis=1)


def cartesian_prod(*args):
    return list(np.stack(np.meshgrid(*args), axis=-1).reshape(-1, len(args)))


def exp_sampling(mean, v_range, reflect_x=False):  # for example: mean = 0.95, v_range = [0.9, 1.1]
    if v_range[0] == v_range[1]:
        return v_range[0]
    
    mean -= v_range[0]
    lambd = 1 / mean
    
    v = v_range[1] + 1
    while v > v_range[1]:
        v = torch.zeros(1).exponential_(lambd) + v_range[0]
        
    if reflect_x:
        v = -v
        v += sum(v_range)
    return v


def get_relative_r_ang(px, py, heading_angle, target_x, target_y):
    heading_angle = np.deg2rad(heading_angle)
    distance_vector = np.vstack([px - target_x, py - target_y])
    relative_r = np.linalg.norm(distance_vector, axis=0)
    
    relative_ang = heading_angle - np.arctan2(distance_vector[1],
                                              distance_vector[0])
    # make the relative angle range [-pi, pi]
    relative_ang = np.remainder(relative_ang, 2 * np.pi)
    relative_ang[relative_ang >= np.pi] -= 2 * np.pi
    return relative_r, relative_ang

        
@contextmanager
def initiate_plot(dimx=24, dimy=9, dpi=100, fontweight='normal'):
    plt.rcParams['figure.figsize'] = (dimx, dimy)
    plt.rcParams['font.weight'] = fontweight
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["font.family"] = 'Arial'
    fig = plt.figure(dpi=dpi)
    yield fig
    plt.show()
    
    
def set_violin_plot(bp, facecolor, edgecolor, linewidth=1, alpha=1, ls='-', hatch=r''):
    plt.setp(bp['bodies'], facecolor=facecolor, edgecolor=edgecolor, 
             linewidth=linewidth, alpha=alpha ,ls=ls, hatch=hatch)
    plt.setp(bp['cmins'], facecolor=facecolor, edgecolor=edgecolor, 
             linewidth=linewidth, alpha=alpha)
    plt.setp(bp['cmaxes'], facecolor=facecolor, edgecolor=edgecolor, 
             linewidth=linewidth, alpha=alpha)
    plt.setp(bp['cmedians'], facecolor='k', edgecolor='k', 
             linewidth=linewidth, alpha=alpha)
    plt.setp(bp['cbars'], facecolor='None', edgecolor='None', 
             linewidth=linewidth, alpha=alpha)
    
    
def set_box_plot(bp, facecolor, edgecolor, linewidth=1, alpha=1, ls='-'):
    plt.setp(bp['boxes'], facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, ls=ls)
    plt.setp(bp['whiskers'], color='k', linewidth=linewidth, alpha=alpha ,ls=ls)
    plt.setp(bp['caps'], color='k', linewidth=linewidth, alpha=alpha ,ls=ls)
    plt.setp(bp['medians'], color='k', linewidth=linewidth, alpha=alpha ,ls=ls)
    
    
def filter_fliers(data, whis=1.5, return_idx=False):
    filtered_data = []; fliers_ides = []
    for value in data:
        Q1, Q2, Q3 = np.percentile(value, [25, 50, 75])
        lb = Q1 - whis * (Q3 - Q1); ub = Q3 + whis * (Q3 - Q1)
        filtered_data.append(value[(value > lb) & (value < ub)])
        fliers_ides.append(np.where((value > lb) & (value < ub))[0])
    if return_idx:
        return filtered_data, fliers_ides
    else:
        return filtered_data
    
    
def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10**(-precision), precision)


def my_floor(a, precision=0):
    return np.round(a - 0.5 * 10**(-precision), precision)


def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def match_similar_trials(df, reference, variables=['target_x', 'target_y'], is_scale=False, is_sort=False, 
                         EPISODE_SIZE=2000, replace=False):
    df.reset_index(drop=True, inplace=True)
    reference.reset_index(drop=True, inplace=True)
    df_trials = df.loc[:, variables].copy()
    reference_trials = reference.loc[:, variables].copy()
    if is_scale:
        df_trials, reference_trials = map(min_max_scale_df, [df_trials, reference_trials])
    
    closest_df_indices = []
    for _, reference_trial in reference_trials.iterrows():
        distance = np.linalg.norm(df_trials - reference_trial, axis=1)
        closest_df_trial = df_trials.iloc[distance.argmin()]
        closest_df_indices.append(closest_df_trial.name)
        if not replace:
            df_trials.drop(closest_df_trial.name, inplace=True)
        
    matched_df = df.loc[closest_df_indices]
    matched_df.reset_index(drop=True, inplace=True)
    
    if is_sort:
        if is_scale:
            matched_df_trials = min_max_scale_df(matched_df.loc[:, variables].copy())
        else:
            matched_df_trials = matched_df.loc[:, variables].copy()
        chosen_indices = np.linalg.norm(matched_df_trials - reference_trials, axis=1).argsort()[:EPISODE_SIZE]
        return matched_df.iloc[chosen_indices], reference.iloc[chosen_indices]
    else:
        return matched_df


def config_colors():
    colors = {'LSTM_c': 'olive', 'EKF_c': 'darkorange', 'monkV_c': 'indianred', 'monkB_c': 'blue',
              'sensory_c': '#29AbE2', 'belief_c': '#C1272D', 'motor_c': '#FF00FF',
              'reward_c': 'C0', 'unreward_c': 'salmon', 
              'gain_colors': ['k', 'C2', 'C3', 'C5', 'C9'], 'pos_perturb_c': 'peru', 'neg_perturb_c': 'darkgreen'}
    return colors


def convert_2d_response(x, y, z, xmin, xmax, ymin, ymax, num_bins=20, kernel_size=3, isconvolve=True):
    @njit
    def compute(*args):
        x_bins = np.linspace(xmin - 1, xmax + 1, num_bins + 1)
        y_bins = np.linspace(ymin - 1, ymax + 1, num_bins + 1)

        x_indices = np.digitize(x, x_bins) - 1
        y_indices = np.digitize(y, y_bins) - 1

        data = np.zeros((num_bins, num_bins))
        count = data.copy()
        for z_idx, z_value in enumerate(z):
            data[y_indices[z_idx], x_indices[z_idx]] += z_value
            count[y_indices[z_idx], x_indices[z_idx]] += 1

        data /= count
        return x_bins, y_bins, data
    
    x_bins, y_bins, data = compute(x, y, z, xmin, xmax, ymin, ymax, num_bins)
    xx, yy = np.meshgrid(x_bins, y_bins)
    
    if isconvolve:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kernel = np.ones((kernel_size, kernel_size))
            data = convolve(data, kernel, boundary='extend')
    return xx, yy, data


def simulate_no_generalization(df_manip, df_normal, subject):
    arg = config.ConfigCore()
    assert subject in ['monkey', 'agent'], 'Subject should be "monkey" or "agent".'
    if subject == 'agent':
        LINEAR_SCALE = arg.LINEAR_SCALE
        ang_func = np.rad2deg
        DT = arg.DT
    else:
        LINEAR_SCALE = 1
        ang_func = lambda x: x
        DT = param_prior_range()['SAMPLING_RATE']
    
    sim_vs = []; sim_ws = []; sim_heads = []
    sim_xs = []; sim_ys = []
    sim_rs = []; sim_thetas = []
    relative_rs = []; relative_angs = []
    sim_action_v = []; sim_action_w = []
    sim_perturb_start_idx = []
    for (_, trial_manip), (_, trial_normal) in zip(df_manip.iterrows(), df_normal.iterrows()):
        sim_v = trial_normal.action_v * trial_manip.gain_v * LINEAR_SCALE
        sim_w = trial_normal.action_w * ang_func(trial_manip.gain_w)
        
        perturb_start_idx = np.nan
        if 'perturb_vpeak' in trial_manip.index and trial_manip.perturb_vpeak != 0:  # a perturbation trial
            if subject == 'agent':
                perturb_start_idx = trial_manip.perturb_start_time
            else:
                perturb_start_idx = round(trial_manip.perturb_start_time / DT)
            
            trial_size = max(perturb_start_idx + trial_manip.perturb_v_gauss.size, sim_v.size)
            if trial_size > sim_v.size:
                sim_v = np.hstack([sim_v, np.zeros(trial_size - sim_v.size)])
                sim_w = np.hstack([sim_w, np.zeros(trial_size - sim_w.size)])
            
            sim_v[perturb_start_idx:perturb_start_idx + trial_manip.perturb_v_gauss.size] += trial_manip.perturb_v_gauss
            sim_w[perturb_start_idx:perturb_start_idx + trial_manip.perturb_w_gauss.size] += trial_manip.perturb_w_gauss
            
        if subject == 'agent':
            sim_v = np.hstack([0, sim_v])[:-1]
            sim_w = np.hstack([0, sim_w])[:-1]
            
        sim_head = np.cumsum(sim_w) * DT + 90
        if subject == 'agent':
            sim_head = np.hstack([90, sim_head])[:-1]
            
        sim_x = np.cumsum(sim_v * np.cos(np.deg2rad(sim_head))) * DT
        sim_y = np.cumsum(sim_v * np.sin(np.deg2rad(sim_head))) * DT
        if subject == 'agent':
            sim_x = np.hstack([0, sim_x])[:-1]
            sim_y = np.hstack([0, sim_y])[:-1]
            
        
        sim_r, sim_theta = cart2pol(sim_x, sim_y)
        relative_r, relative_ang = get_relative_r_ang(sim_x, sim_y, sim_head, 
                                                      trial_manip.target_x, 
                                                      trial_manip.target_y)
        sim_vs.append(sim_v); sim_ws.append(sim_w)
        sim_heads.append(sim_head)
        sim_xs.append(sim_x); sim_ys.append(sim_y)
        sim_rs.append(sim_r)
        sim_thetas.append(np.rad2deg(sim_theta))
        relative_rs.append(relative_r)
        relative_angs.append(np.rad2deg(relative_ang))
        sim_action_v.append(trial_normal.action_v)
        sim_action_w.append(trial_normal.action_w)
        sim_perturb_start_idx.append(perturb_start_idx)
        
    return df_manip.assign(sim_pos_v=sim_vs, sim_pos_w=sim_ws, sim_head_dir=sim_heads,
                           sim_pos_x=sim_xs, sim_pos_y=sim_ys, 
                           sim_pos_r=sim_rs, sim_pos_theta=sim_thetas,
                           sim_pos_r_end=[r[-1] for r in sim_rs],
                           sim_pos_theta_end=[theta[-1] for theta in sim_thetas],
                           sim_relative_radius=relative_rs, sim_relative_angle=relative_angs,
                           sim_relative_radius_end=[r[-1] for r in relative_rs],
                           sim_relative_angle_end=[ang[-1] for ang in relative_angs],
                           sim_action_v=sim_action_v, sim_action_w=sim_action_w,
                           sim_perturb_start_idx=sim_perturb_start_idx)


def get_neural_response(agent, df, v_multiplier=1, model_name='actor', layer='rnn1', use_obs=False):
    arg = config.ConfigCore()
    model = agent.actor if model_name == 'actor' else agent.critic
    v_key = ['pos_v', 'pos_w']
    o_key = ['obs_v', 'obs_w']
    a_key = ['action_v', 'action_w']
    tar_key = ['target_x', 'target_y']
    responses = []

    df = pd.DataFrame([df]) if isinstance(df, pd.Series) else df
    for _, trial in df.iterrows():
        if use_obs:
            v_ = np.vstack([np.zeros((1, len(o_key))), 
                            np.vstack(trial[o_key]).T[:-1]]).reshape(-1, 1, len(o_key))
        else:
            v_ = np.vstack(trial[v_key]).T.reshape(-1, 1, len(v_key)) * v_multiplier
            v_[..., 0] /= arg.LINEAR_SCALE
            v_[..., 1] = np.deg2rad(v_[..., 1])

        if agent.get_self_action:
            a_ = np.vstack([np.zeros((1, len(a_key))), np.vstack(trial[a_key]).T[:-1]]).reshape(-1, 1, len(a_key))
        else:
            a_ = v_
            
        states = torch.tensor(np.concatenate([v_, a_], axis=2), dtype=torch.float32)
        
        with torch.no_grad():
            response, _ = model.rnn1(states)
            if layer != 'rnn1':
                response = model.l1(response)
                if layer != 'l1':
                    tar_ = trial[tar_key].values.astype(np.float32) / arg.LINEAR_SCALE
                    tar_ = torch.ones(v_.shape) * tar_
                    if model_name == 'actor':
                        response = F.relu(model.l2(torch.cat([response, tar_], dim=2)))
                    else:
                        _a_ = torch.tensor(np.vstack(trial[a_key]).T.reshape(-1, 1, len(a_key)), dtype=torch.float32)
                        response = F.relu(model.l2(torch.cat([response, tar_, _a_], dim=2)))
                    if layer != 'l2':
                        response = F.relu(model.l3(response))
            responses.append(response)
        
    return torch.cat(responses).squeeze(1).numpy()


def my_tickformatter(value, pos):
    if abs(value > 0) and abs(value < 1):
        value = str(value).replace('0.', '.')
    elif value == 0:
        value = 0
    elif int(value) == value:
        value = int(value)
    return value


def get_radial_error(dfs, is_sim=False):
    if not isinstance(dfs, list):
        dfs = [dfs]
    
    errors = []
    pos_r_key = 'pos_r_end'
    rel_r_key = 'relative_radius_end'
      
    if is_sim:
        pos_r_key = 'sim_' + pos_r_key
        rel_r_key = 'sim_' + rel_r_key
        
    for df in dfs:
        undershoot = df[pos_r_key] < df.target_r
        error = df[rel_r_key].copy()
        error[undershoot] = - error[undershoot]
        errors.append(error.values)
        
    return errors
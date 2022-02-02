import pandas as pd
import numpy as np
import torch
from scipy.signal import medfilt
from scipy.stats import norm
import neo
from pathlib import Path
from rult import *
from monkey_functions import down_sampling, dfdownsample


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def bypass(input):
    return input

class MonkeyDataExtractor():
    def __init__(self, folder_path):
        self.monkey_class = 'NYU' if 'NYU' in str(data_path) else 'BCM'
        self.folder_path = folder_path
        self.smr_full_file_path = sorted(self.folder_path.glob('*.smr'))
        self.log_full_file_path = [file.parent / (file.stem + '.log') 
                                   for file in self.smr_full_file_path]
        if self.monkey_class == 'NYU':
            self.marker_memo = {'file_start': 1, 'trial_start': 2, 'trial_end': 3,
                            'juice' :4, 'perturb_start': 8}
        else:
            self.marker_memo = {'file_start': 1, 'trial_start': 2, 'trial_end': 3,
                            'juice' :4, 'perturb_start': 8, 'perturb_start2': 5}
            
        self.y_offset = 32.5
    
    def __call__(self, downsample_fun=bypass, saving_fun=bypass, returndata=False):
        if self.monkey_class == 'NYU':
            self.nyu_extract_smr()
            self.nyu_extract_log()
            self.nyu_segment()
        else:
            self.bcm_extract_smr()
            self.bcm_extract_log()
            self.bcm_segment()
            downsample_fun(self.monkey_trajectory)
            saving_fun(self.monkey_trajectory)
        if returndata:
            return self.monkey_trajectory
            
    def nyu_extract_smr(self):
        channel_signal_all = []
        marker_all = []
        
        for idx, file_name in enumerate(self.smr_full_file_path):
            seg_reader = neo.io.Spike2IO(filename=file_name).read_segment()
            
            if idx == 0: # only get sampling rate once
                self.SAMPLING_RATE = seg_reader.analogsignals[0].sampling_rate.item()
             
            # Sometimes the length across channels varies a bit
            analog_length = min([i.size for i in seg_reader.analogsignals])
            channel_signal = np.ones((analog_length, seg_reader.size['analogsignals'] + 1))
            
            channel_names = []
            for ch_idx, ch_data in enumerate(seg_reader.analogsignals):
                channel_signal[:, ch_idx] = ch_data.as_array()[:analog_length].T
                channel_names.append(ch_data.annotations['channel_names'][0])
            
            # Add a time channel
            channel_signal[:, -1] = seg_reader.analogsignals[0].times[:analog_length]
            channel_names.append('Time') 
            
            channel_signal_all.append(pd.DataFrame(channel_signal, columns=channel_names))
            
            marker_channel_idx = [idx for idx, value 
                                    in enumerate(seg_reader.events)
                                    if value.name == 'marker'][0]
            marker_key, marker_time = (
                seg_reader.events[marker_channel_idx].get_labels().astype('int'),
                seg_reader.events[marker_channel_idx].as_array())
            marker = {'key': marker_key, 'time': marker_time}
            marker_all.append(marker)
            
            self.channel_signal_all = channel_signal_all
            self.marker_all = marker_all
            
    def nyu_extract_log(self):
        log_data_all = []
        
        for file_name in self.log_full_file_path:
            with open(file_name, 'r', encoding='UTF-8') as content:
                log_content = content.readlines()
            
            floor_density = []
            perturb_vpeak = []; perturb_wpeak = []; perturb_start_time_ori = []
            full_on = []; target_x = []; target_y = []
            for line in log_content:
                if 'Joy Stick Max Velocity' in line:
                    gain_v = float(line.split(': ')[1])
                    
                if 'Joy Stick Max Angular Velocity' in line:
                    gain_w = float(line.split(': ')[1])
                    
                if 'Perturb Max Velocity' in line:
                    perturb_vpeakmax = float(line.split(': ')[1])
                    
                if 'Perturb Max Angular Velocity' in line:
                    perturb_wpeakmax = float(line.split(': ')[1])
                    
                if 'Perturbation Sigma' in line:
                    perturb_sigma = float(line.split(': ')[1])
                    
                if 'Perturbation Duration' in line:
                    perturb_dur = float(line.split(': ')[1])
                    
                if 'Floor Density' in line:
                    content_temp = float(line.split(': ')[1])
                    floor_density.append(content_temp)
                    
                if 'Perturbation Linear Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_vpeak.append(content_temp)
                    
                if 'Perturbation Angular Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_wpeak.append(- content_temp)
                    
                if 'Perturbation Delay Time' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_start_time_ori.append(content_temp / 1000)  # ms to s
                    
                if 'Firefly Full On' in line:
                    content_temp = bool(int(line.split(': ')[1]))
                    full_on.append(content_temp)
                
                if 'Position x/y(cm)' in line:
                    content_temp_x, content_temp_y = line.split(': ')[1].split(' ')
                    target_x.append(float(content_temp_x))
                    # Monkey data's y positions are reversed.
                    target_y.append(- float(content_temp_y) + self.y_offset)
                
            log_data_all.append({'gain_v': gain_v, 'gain_w': gain_w,
                                 'perturb_vpeakmax': perturb_vpeakmax, 'perturb_wpeakmax': perturb_wpeakmax,
                                 'perturb_sigma': perturb_sigma, 'perturb_dur': perturb_dur,
                                 'floor_density': floor_density,
                                 'perturb_vpeak': perturb_vpeak, 'perturb_wpeak': perturb_wpeak,
                                 'perturb_start_time_ori': perturb_start_time_ori,
                                  'full_on': full_on, 'target_x': target_x, 'target_y': target_y})
            
            self.log_data_all = log_data_all
            
    def nyu_segment(self, lazy_threshold=4000, skip_threshold=400, skip_r_threshold=30, 
                    crazy_threshold=200,
                    medfilt_kernel=5, v_threshold=1, reward_boundary=65, 
                    perturb_corr_threshold=50):
        # lazy_threshold (data points): Trial is too long.
        # skip_threshold (data points): Trial is too short.
        # skip_r_threshold (cm): Monkey did not move a lot.
        # crazy_threshold (cm): Monkey stopped too far.
        # v_threshold (cm/s): Velocity threshold for start and end.
        # reward_boundary (cm): Rewarded when stop inside this circular boundary.
        # perturb_corr_threshold (data points): Corrected perturbation start index should not be too biased.
        
        gain_v = []; gain_w = []; perturb_vpeakmax = []; perturb_wpeakmax = []
        perturb_sigma = []; perturb_dur = []; perturb_vpeak = []; perturb_wpeak = []
        perturb_v = []; perturb_w = []; perturb_v_gauss = []; perturb_w_gauss = []
        perturb_start_time = []; perturb_start_time_ori = []
        floor_density = []; pos_x = []; pos_y = []
        head_dir = []; head_dir_end = []; pos_r = []; pos_theta = []
        pos_r_end = []; pos_theta_end = []
        pos_v = []; pos_w = []; target_x = []; target_y = []
        target_r = []; target_theta = []; full_on = []; rewarded = []
        relative_radius = []; relative_angle = []; time = []
        trial_dur = []; action_v = []; action_w = []
        relative_radius_end = []; relative_angle_end = []; category = []

        for session_idx, session_data in enumerate(self.channel_signal_all):
            log_data = self.log_data_all[session_idx]
            marker_data = self.marker_all[session_idx]
            start_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_start']]
            end_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_end']]
            perturb_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['perturb_start']]

            # segment trials
            for trial_idx in range(end_marker_times.size):
                trial_data = session_data[np.logical_and(
                    session_data.Time > start_marker_times[trial_idx],
                    session_data.Time < end_marker_times[trial_idx])].copy()
                
                # Use median filter kernel size as 5 to remove spike noise first.
                trial_data['ForwardV'] = medfilt(trial_data['ForwardV'], medfilt_kernel)
                trial_data['AngularV'] = medfilt(trial_data['AngularV'], medfilt_kernel)
                
                # cut non-moving head and tail
                moving_period = np.where(trial_data['ForwardV'].abs() > v_threshold)[0]
                if moving_period.size > 0:
                    start_idx = moving_period[0]
                    end_idx = moving_period[-1] + 2
                else:
                    start_idx = 0
                    end_idx = None
                  
                # store trial data
                trial_data = trial_data.iloc[start_idx : end_idx]
                trial_data['AngularV'] = - trial_data['AngularV']
                trial_data['MonkeyYa'] = np.cumsum(trial_data['AngularV']) / self.SAMPLING_RATE + 90
                trial_data['MonkeyX'] = np.cumsum(trial_data['ForwardV']
                                            * np.cos(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                trial_data['MonkeyY'] = np.cumsum(trial_data['ForwardV']
                                            * np.sin(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                
                gain_v.append(log_data['gain_v'])
                gain_w.append(log_data['gain_w'])
                perturb_vpeakmax.append(log_data['perturb_vpeakmax'])
                perturb_wpeakmax.append(log_data['perturb_wpeakmax'])
                perturb_vpeak.append(log_data['perturb_vpeak'][trial_idx])
                perturb_wpeak.append(log_data['perturb_wpeak'][trial_idx])
                perturb_start_time_ori.append(log_data['perturb_start_time_ori'][trial_idx])
                perturb_sigma.append(log_data['perturb_sigma'])
                perturb_dur.append(log_data['perturb_dur'])
                pos_x.append(trial_data['MonkeyX'].values)
                pos_y.append(trial_data['MonkeyY'].values)
                head_dir.append(trial_data['MonkeyYa'].values)
                head_dir_end.append(trial_data['MonkeyYa'].values[-1])
                floor_density.append(log_data['floor_density'][trial_idx])
                full_on.append(log_data['full_on'][trial_idx])

                rho, phi = cart2pol(pos_x[-1], pos_y[-1])
                pos_r.append(rho)
                pos_theta.append(np.rad2deg(phi))
                pos_r_end.append(rho[-1])
                pos_theta_end.append(np.rad2deg(phi[-1]))
                
                
                # determine if it is a perturbation trial
                perturb_start_time_temp = perturb_marker_times[(np.logical_and(
                                            perturb_marker_times > start_marker_times[trial_idx],
                                            perturb_marker_times < end_marker_times[trial_idx]))]
                if bool(perturb_start_time_temp.size):
                    assert perturb_start_time_temp.size == 1
                    pos_v.append(trial_data['ForwardV'].values)
                    pos_w.append(trial_data['AngularV'].values)
                    
                    # construct perturbation curves
                    perturb_xaxis = np.linspace(0, perturb_dur[-1], round(self.SAMPLING_RATE))
                    perturb_temp = norm.pdf(perturb_xaxis, loc=perturb_dur[-1] / 2, scale=perturb_sigma[-1])
                    perturb_temp /= perturb_temp.max()
                    perturb_v_temp = perturb_temp * perturb_vpeak[-1]
                    perturb_w_temp = perturb_temp * perturb_wpeak[-1]
                    perturb_v_gauss.append(perturb_v_temp)
                    perturb_w_gauss.append(perturb_w_temp)
                    
                    # use obvious angular perturbation curve as a template
                    if abs(perturb_wpeak[-1]) / perturb_wpeakmax[-1] > 0.1:
                        perturb_template = perturb_w_temp
                        original_vel = pos_w[-1]
                    else:
                        perturb_template = perturb_v_temp
                        original_vel = pos_v[-1]
                        
                    # use the template to do cross-correlation to find perturbation start time
                    perturb_start_idx_mark = int((perturb_start_time_temp
                                                - trial_data['Time'].values[0]) * self.SAMPLING_RATE)
                    perturb_start_idx_mark = np.clip(perturb_start_idx_mark, 0, None)
                    perturb_peak_idx = np.correlate(original_vel, perturb_template, mode='same').argsort()[::-1]
                    perturb_start_idx_corr = perturb_peak_idx - perturb_dur[-1] / 2 * self.SAMPLING_RATE
                    mask = (perturb_start_idx_corr > 0) \
                           & (perturb_start_idx_corr > perturb_start_idx_mark) \
                           & (perturb_start_idx_corr - perturb_start_idx_mark < perturb_corr_threshold)
                    
                    if mask.sum() == 0 or original_vel.size < perturb_template.size:
                        perturb_start_idx = np.clip(perturb_start_idx_mark, None, pos_v[-1].size - 1)
                    else:
                        perturb_start_idx = int(perturb_start_idx_corr[mask][0])
                    perturb_start_time.append(perturb_start_idx / self.SAMPLING_RATE)
                    
                    # get pure actions
                    perturb_v_full = np.zeros_like(pos_v[-1])
                    perturb_v_full[perturb_start_idx:perturb_start_idx + perturb_v_temp.size] = \
                                                            perturb_v_temp[:perturb_v_full.size - perturb_start_idx]
                    perturb_w_full = np.zeros_like(pos_w[-1])
                    perturb_w_full[perturb_start_idx:perturb_start_idx + perturb_w_temp.size] = \
                                                            perturb_w_temp[:perturb_w_full.size - perturb_start_idx]
                    
                    perturb_v.append(perturb_v_full); perturb_w.append(perturb_w_full)
                    action_v.append((pos_v[-1] - perturb_v_full).clip(-gain_v[-1], gain_v[-1]) / gain_v[-1])
                    action_w.append((pos_w[-1] - perturb_w_full).clip(-gain_w[-1], gain_w[-1]) / gain_w[-1])
                else:
                    pos_v.append(trial_data['ForwardV'].values.clip(-gain_v[-1], gain_v[-1]))
                    pos_w.append(trial_data['AngularV'].values.clip(-gain_w[-1], gain_w[-1]))
                    perturb_v_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_w_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_start_time.append(np.nan)
                    perturb_v.append(np.zeros_like(pos_v[-1])); perturb_w.append(np.zeros_like(pos_w[-1]))
                    action_v.append(pos_v[-1] / gain_v[-1])
                    action_w.append(pos_w[-1] / gain_w[-1])
                
                trial_data['Time'] -= trial_data['Time'].iloc[0]
                time.append(trial_data['Time'].values)
                trial_dur.append(trial_data['Time'].values[-1])
                target_x.append(log_data['target_x'][trial_idx])
                target_y.append(log_data['target_y'][trial_idx])
                tar_rho, tar_phi = cart2pol(target_x[-1], target_y[-1])
                target_r.append(tar_rho)
                target_theta.append(np.rad2deg(tar_phi))
                
                relative_r, relative_ang = get_relative_r_ang(
                                pos_x[-1], pos_y[-1], head_dir[-1], target_x[-1], target_y[-1])
                relative_radius.append(relative_r)
                relative_angle.append(np.rad2deg(relative_ang))
                relative_radius_end.append(relative_r[-1])
                relative_angle_end.append(np.rad2deg(relative_ang[-1]))
                rewarded.append(relative_r[-1] < reward_boundary)

                # Categorize trials
                if rewarded[-1]:
                    category.append('normal')
                else:
                    if trial_data['ForwardV'].size < skip_threshold or\
                       pos_r_end[-1] < skip_r_threshold:
                        category.append('skip')
                    elif trial_data['ForwardV'].size > lazy_threshold:
                        category.append('lazy')
                    elif relative_r[-1] > crazy_threshold:
                        category.append('crazy')
                    else:
                        category.append('normal')


        # Construct a dataframe   
        self.monkey_trajectory = pd.DataFrame().assign(gain_v=gain_v, gain_w=gain_w, 
                                 perturb_vpeakmax=perturb_vpeakmax, perturb_wpeakmax=perturb_wpeakmax,
                                 perturb_sigma=perturb_sigma, perturb_dur=perturb_dur,
                                 perturb_vpeak=perturb_vpeak, perturb_wpeak=perturb_wpeak,
                                 perturb_start_time=perturb_start_time,
                                 perturb_start_time_ori=perturb_start_time_ori,
                                 perturb_v_gauss=perturb_v_gauss, perturb_w_gauss=perturb_w_gauss,
                                 perturb_v=perturb_v, perturb_w=perturb_w,
                                 floor_density=floor_density, pos_x=pos_x,
                                 pos_y=pos_y, head_dir=head_dir, head_dir_end=head_dir_end,
                                 pos_r=pos_r, pos_theta=pos_theta, pos_r_end=pos_r_end,
                                 pos_theta_end=pos_theta_end, pos_v=pos_v, pos_w=pos_w, 
                                 target_x=target_x, target_y=target_y, target_r=target_r,
                                 target_theta=target_theta, full_on=full_on, rewarded=rewarded,
                                 relative_radius=relative_radius, relative_angle=relative_angle,
                                 time=time, trial_dur=trial_dur, 
                                 action_v=action_v, action_w=action_w, 
                                 relative_radius_end=relative_radius_end,
                                 relative_angle_end=relative_angle_end, category=category)

    def bcm_extract_smr(self):
        print('starting ext')
        channel_signal_all = []
        marker_all = []
        
        for idx, file_name in enumerate(self.smr_full_file_path):
            seg_reader = neo.io.Spike2IO(filename=str(file_name)).read_segment()
            
            if idx == 0: # only get sampling rate once
                self.SAMPLING_RATE = seg_reader.analogsignals[0].sampling_rate.item()
                
            # Sometimes the length across channels varies a bit
            analog_length = min([i.size for i in seg_reader.analogsignals])
            channel_signal = np.ones((analog_length, seg_reader.size['analogsignals']))
            
            channel_names = []
            # Do not read the last channel as it has a unique shape.
            for ch_idx, ch_data in enumerate(seg_reader.analogsignals[:-1]):
                channel_signal[:, ch_idx] = ch_data.as_array()[:analog_length].T
                channel_names.append(ch_data.annotations['channel_names'][0])
            
            # Add a time channel
            channel_signal[:, -1] = seg_reader.analogsignals[0].times[:analog_length]
            channel_names.append('Time') 
            
            channel_signal_all.append(pd.DataFrame(channel_signal,columns=channel_names))
            
            marker_channel_idx = [idx for idx, value 
                                    in enumerate(seg_reader.events)
                                    if value.name == 'marker'][0]
            marker_key, marker_time = (
                seg_reader.events[marker_channel_idx].get_labels().astype('int'),
                seg_reader.events[marker_channel_idx].as_array())
            marker = {'key': marker_key, 'time': marker_time}
            marker_all.append(marker)
            
            self.channel_signal_all = channel_signal_all
            self.marker_all = marker_all
            
    def bcm_extract_log(self):
        log_data_all = []
        
        for file_name in self.log_full_file_path:
            with open(file_name, 'r', encoding='UTF-8') as content:
                log_content = content.readlines()
                
            floor_density = []
            perturb_vpeak = []; perturb_wpeak = []
            perturb_start_time_ori = []
            full_on = []
            for line in log_content:
                if 'Joy Stick Max Velocity' in line:
                    gain_v = float(line.split(': ')[1])
                    
                if 'Joy Stick Max Angular Velocity' in line:
                    gain_w = float(line.split(': ')[1])
                    
                if 'Perturb Max Velocity' in line:
                    perturb_vpeakmax = float(line.split(': ')[1])
                    
                if 'Perturb Max Angular Velocity' in line:
                    perturb_wpeakmax = float(line.split(': ')[1])
                    
                if 'Perturbation Sigma' in line:
                    perturb_sigma = float(line.split(': ')[1])
                    
                if 'Perturbation Duration' in line:
                    perturb_dur = float(line.split(': ')[1])
                    
                if 'Floor Density' in line:
                    content_temp = float(line.split(': ')[1])
                    floor_density.append(content_temp)
                    
                if 'Perturbation Linear Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_vpeak.append(content_temp)
                    
                if 'Perturbation Angular Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_wpeak.append(- content_temp)
                    
                if 'Perturbation Delay Time' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_start_time_ori.append(content_temp / 1000)
                    
                if 'Firefly Full ON' in line:
                    content_temp = bool(int(line.split(': ')[1]))
                    full_on.append(content_temp)
            
            if len(full_on) == 0: # Quigley's perturbation sessions
                full_on = [False] * len(floor_density)
            
            log_data_all.append({'gain_v': gain_v, 'gain_w': gain_w, 
                                 'perturb_vpeakmax': perturb_vpeakmax, 'perturb_wpeakmax': perturb_wpeakmax,
                                 'perturb_sigma': perturb_sigma, 'perturb_dur': perturb_dur,
                                 'floor_density': floor_density, 
                                 'perturb_vpeak': perturb_vpeak, 'perturb_wpeak': perturb_wpeak,
                                 'perturb_start_time_ori': perturb_start_time_ori,
                                 'full_on': full_on})
            self.log_data_all = log_data_all
            
    def bcm_segment(self, lazy_threshold=4000, skip_threshold=400, skip_r_threshold=30, 
                    crazy_threshold=200,
                    medfilt_kernel=5, v_threshold=1, reward_boundary=65,
                    target_r_range=[100, 400], target_theta_range=[55, 125], 
                    target_tolerance=1, perturb_corr_threshold=100):
        print('starting segmenting')
        # lazy_threshold (time): Trial is too long.

        # skip_threshold (time): Trial is too short.
        # and
        # skip_r_threshold (cm): Monkey did not move a lot.

        # crazy_threshold (cm): Monkey stopped too far, very wrong.

        # medfilt_kernel (data points): Remove spikes from raw data.
        # v_threshold (cm/s): Threshold for end point.
        # reward_boundary (cm): Rewarded when stop inside this circular boundary.
        # target_r_range (cm): Radius of target distribution.
        # target_theta_range (deg): Angle of target distribution.
        # target_tolerance (cm or deg): Max tolerance for targets out of distribution.
        # perturb_corr_threshold (data points): Corrected perturbation start index should not be too biased.

        gain_v = []; gain_w = []; perturb_vpeakmax = []; perturb_wpeakmax = []
        perturb_sigma = []; perturb_dur = []; perturb_vpeak = []; perturb_wpeak = []
        perturb_v = []; perturb_w = []; perturb_v_gauss = []; perturb_w_gauss = []
        perturb_start_time = []; perturb_start_time_ori = []
        floor_density = []; pos_x = []; pos_y = []
        head_dir = []; head_dir_end = []; pos_r = []; pos_theta = []; 
        pos_r_end = []; pos_theta_end = []
        pos_v = []; pos_w = []; target_x = []; target_y = []
        target_r = []; target_theta = []; full_on = []; rewarded = []
        relative_radius = []; relative_angle = []; time = []; 
        trial_dur = []; action_v = []; action_w = []; 
        relative_radius_end = []; relative_angle_end = []; category = []

        for session_idx, session_data in enumerate(self.channel_signal_all):
            log_data = self.log_data_all[session_idx]
            marker_data = self.marker_all[session_idx]
            start_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_start']]
            end_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_end']]
            perturb_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['perturb_start']]
            if perturb_marker_times.size == 0:
                perturb_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['perturb_start2']]

            # segment trials
            for trial_idx in range(end_marker_times.size):
                trial_data = session_data[np.logical_and(
                    session_data.Time > start_marker_times[trial_idx],
                    session_data.Time < end_marker_times[trial_idx])].copy()

                # Use median filter kernel size as 5 to remove spike noise first.
                trial_data['ForwardV'] = medfilt(trial_data['ForwardV'], medfilt_kernel)
                trial_data['AngularV'] = medfilt(trial_data['AngularV'], medfilt_kernel)
                
                # cut non-moving head and tail
                moving_period = np.where(trial_data['ForwardV'].abs() > v_threshold)[0]
                if moving_period.size > 0:
                    start_idx = moving_period[0]
                    end_idx = moving_period[-1] + 2
                else:
                    start_idx = 0
                    end_idx = None
                    
                # store trial data
                trial_data = trial_data.iloc[start_idx : end_idx]
                trial_data['AngularV'] = - trial_data['AngularV']
                trial_data['MonkeyYa'] = np.cumsum(trial_data['AngularV']) / self.SAMPLING_RATE + 90
                trial_data['MonkeyX'] = np.cumsum(trial_data['ForwardV']
                                            * np.cos(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                trial_data['MonkeyY'] = np.cumsum(trial_data['ForwardV']
                                            * np.sin(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                
                gain_v.append(log_data['gain_v'])
                gain_w.append(log_data['gain_w'])
                perturb_vpeakmax.append(log_data['perturb_vpeakmax'])
                perturb_wpeakmax.append(log_data['perturb_wpeakmax'])
                perturb_vpeak.append(log_data['perturb_vpeak'][trial_idx])
                perturb_wpeak.append(log_data['perturb_wpeak'][trial_idx])
                perturb_start_time_ori.append(log_data['perturb_start_time_ori'][trial_idx])
                perturb_sigma.append(log_data['perturb_sigma'])
                perturb_dur.append(log_data['perturb_dur'])
                pos_x.append(trial_data['MonkeyX'].values)
                pos_y.append(trial_data['MonkeyY'].values)
                head_dir.append(trial_data['MonkeyYa'].values)
                head_dir_end.append(trial_data['MonkeyYa'].values[-1])
                floor_density.append(log_data['floor_density'][trial_idx])
                full_on.append(log_data['full_on'][trial_idx])
                
                rho, phi = cart2pol(pos_x[-1], pos_y[-1])
                pos_r.append(rho)
                pos_theta.append(np.rad2deg(phi))
                pos_r_end.append(rho[-1])
                pos_theta_end.append(np.rad2deg(phi[-1]))
                
                
                # determine if it is a perturbation trial
                perturb_start_time_temp = perturb_marker_times[(np.logical_and(
                                            perturb_marker_times > start_marker_times[trial_idx],
                                            perturb_marker_times < end_marker_times[trial_idx]))]
                if bool(perturb_start_time_temp.size):
                    assert perturb_start_time_temp.size == 1
                    pos_v.append(trial_data['ForwardV'].values)
                    pos_w.append(trial_data['AngularV'].values)
                
                    # construct perturbation curves
                    perturb_xaxis = np.linspace(0, perturb_dur[-1], round(self.SAMPLING_RATE))
                    perturb_temp = norm.pdf(perturb_xaxis, loc=perturb_dur[-1] / 2, scale=perturb_sigma[-1])
                    perturb_temp /= perturb_temp.max()
                    perturb_v_temp = perturb_temp * perturb_vpeak[-1]
                    perturb_w_temp = perturb_temp * perturb_wpeak[-1]
                    perturb_v_gauss.append(perturb_v_temp)
                    perturb_w_gauss.append(perturb_w_temp)
                    
                    # use the more obvious perturbation curve as a template
                    corrcoef_v = np.correlate(pos_v[-1] - pos_v[-1].mean(), 
                                              perturb_v_temp - perturb_v_temp.mean(),
                                              mode='same') / (pos_v[-1].std() * perturb_v_temp.std())
                    corrcoef_w = np.correlate(pos_w[-1] - pos_w[-1].mean(), 
                                              perturb_w_temp - perturb_w_temp.mean(),
                                              mode='same') / (pos_w[-1].std() * perturb_w_temp.std())
                    if corrcoef_v.max() > corrcoef_w.max():
                        perturb_template = perturb_v_temp
                        original_vel = pos_v[-1]
                    else:
                        perturb_template = perturb_w_temp
                        original_vel = pos_w[-1]
                        
                    # use the template to do cross-correlation to find perturbation start time
                    perturb_start_idx_mark = int((perturb_start_time_temp
                                                - trial_data['Time'].values[0]) * self.SAMPLING_RATE)
                    perturb_start_idx_mark = np.clip(perturb_start_idx_mark, 0, None)
                    perturb_peak_idx = np.correlate(original_vel, perturb_template, mode='same').argsort()[::-1]
                    perturb_start_idx_corr = perturb_peak_idx - perturb_dur[-1] / 2 * self.SAMPLING_RATE
                    mask = (perturb_start_idx_corr > 0) \
                           & (perturb_start_idx_corr > perturb_start_idx_mark) \
                           & (perturb_start_idx_corr - perturb_start_idx_mark < perturb_corr_threshold)
                    
                    if mask.sum() == 0 or original_vel.size < perturb_template.size:
                        perturb_start_idx = np.clip(perturb_start_idx_mark, None, pos_v[-1].size - 1)
                    else:
                        perturb_start_idx = int(perturb_start_idx_corr[mask][0])
                    perturb_start_time.append(perturb_start_idx / self.SAMPLING_RATE)
                    
                    # get pure actions
                    perturb_v_full = np.zeros_like(pos_v[-1])
                    perturb_v_full[perturb_start_idx:perturb_start_idx + perturb_v_temp.size] = \
                                                            perturb_v_temp[:perturb_v_full.size - perturb_start_idx]
                    perturb_w_full = np.zeros_like(pos_w[-1])
                    perturb_w_full[perturb_start_idx:perturb_start_idx + perturb_w_temp.size] = \
                                                            perturb_w_temp[:perturb_w_full.size - perturb_start_idx]
                    
                    perturb_v.append(perturb_v_full); perturb_w.append(perturb_w_full)
                    action_v.append((pos_v[-1] - perturb_v_full).clip(-gain_v[-1], gain_v[-1]) / gain_v[-1])
                    action_w.append((pos_w[-1] - perturb_w_full).clip(-gain_w[-1], gain_w[-1]) / gain_w[-1])
                else:
                    pos_v.append(trial_data['ForwardV'].values.clip(-gain_v[-1], gain_v[-1]))
                    pos_w.append(trial_data['AngularV'].values.clip(-gain_w[-1], gain_w[-1]))
                    perturb_v_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_w_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_start_time.append(np.nan)
                    perturb_v.append(np.zeros_like(pos_v[-1])); perturb_w.append(np.zeros_like(pos_w[-1]))
                    action_v.append(pos_v[-1] / gain_v[-1])
                    action_w.append(pos_w[-1] / gain_w[-1])
                

                trial_data['Time'] -= trial_data['Time'].iloc[0]
                time.append(trial_data['Time'].values)
                trial_dur.append(trial_data['Time'].values[-1])
                
                
                # target position is analog in BCM data, I bin target channels
                # and find the mode of bins
                targetx_bins = np.arange(my_floor(trial_data['FireflyX'].min(), 1),
                                         my_ceil(trial_data['FireflyX'].max(), 1), 0.1)
                targetx_idxes = np.digitize(trial_data['FireflyX'], targetx_bins)
                targetx_hist, _ = np.histogram(trial_data['FireflyX'], targetx_bins)
                try:
                    tar_x = trial_data['FireflyX'][
                                    targetx_idxes == targetx_hist.argmax()+1].mean()
                except: # when start_idx == end_idx, they are bad trials that not matter
                    tar_x = trial_data['FireflyX'].mean()

                targety_bins = np.arange(my_floor(trial_data['FireflyY'].min(), 1),
                                         my_ceil(trial_data['FireflyY'].max(), 1), 0.1)
                targety_idxes = np.digitize(trial_data['FireflyY'], targety_bins)
                targety_hist, _ = np.histogram(trial_data['FireflyY'], targety_bins)
                try:
                    tar_y = trial_data['FireflyY'][
                                    targety_idxes == targety_hist.argmax()+1].mean()
                except:
                    tar_y = trial_data['FireflyY'].mean()

                target_x.append(tar_x)
                target_y.append(- tar_y + self.y_offset)
                tar_rho, tar_phi = cart2pol(target_x[-1], target_y[-1])
                target_r.append(tar_rho)
                target_theta.append(np.rad2deg(tar_phi))

                relative_r, relative_ang = get_relative_r_ang(
                                pos_x[-1], pos_y[-1], head_dir[-1], target_x[-1], target_y[-1])
                relative_radius.append(relative_r)
                relative_angle.append(np.rad2deg(relative_ang))
                relative_radius_end.append(relative_r[-1])
                relative_angle_end.append(np.rad2deg(relative_ang[-1]))
                rewarded.append(relative_r[-1] < reward_boundary)

                #juice_time = marker_data['time'][marker_data['key'] == marker_memo['juice']]
                #j_marker = np.where(np.logical_and(juice_time > start_marker_times[trial_idx],
                #               juice_time < end_marker_times[trial_idx]))[0]

                # Categorize trials
                # Note that few targets in BCM data are out of the distribution
                # for unknown reason, I just label and ignore them.
                if target_r[-1] < target_r_range[0] - target_tolerance or\
                   target_r[-1] > target_r_range[1] + target_tolerance or\
                   target_theta[-1] < target_theta_range[0] - target_tolerance or\
                   target_theta[-1] > target_theta_range[1] + target_tolerance:
                    category.append('wrong_target')
                else:
                    if rewarded[-1]:
                        category.append('normal')
                    else:
                        if trial_data['ForwardV'].size < skip_threshold and\
                           pos_r_end[-1] < skip_r_threshold:
                            category.append('skip')
                        elif trial_data['ForwardV'].size > lazy_threshold:
                            category.append('lazy')
                        elif relative_r[-1] > crazy_threshold:
                            category.append('crazy')
                        else:
                            category.append('normal')

        # Construct a dataframe   
        self.monkey_trajectory = pd.DataFrame().assign(gain_v=gain_v, gain_w=gain_w, 
                                 perturb_vpeakmax=perturb_vpeakmax, perturb_wpeakmax=perturb_wpeakmax,
                                 perturb_sigma=perturb_sigma, perturb_dur=perturb_dur,
                                 perturb_vpeak=perturb_vpeak, perturb_wpeak=perturb_wpeak,
                                 perturb_start_time=perturb_start_time,
                                 perturb_start_time_ori=perturb_start_time_ori,
                                 perturb_v_gauss=perturb_v_gauss, perturb_w_gauss=perturb_w_gauss,
                                 perturb_v=perturb_v, perturb_w=perturb_w,
                                 floor_density=floor_density, pos_x=pos_x,
                                 pos_y=pos_y, head_dir=head_dir, head_dir_end=head_dir_end,
                                 pos_r=pos_r, pos_theta=pos_theta, pos_r_end=pos_r_end,
                                 pos_theta_end=pos_theta_end, pos_v=pos_v, pos_w=pos_w, 
                                 target_x=target_x, target_y=target_y, target_r=target_r,
                                 target_theta=target_theta, full_on=full_on, rewarded=rewarded,
                                 relative_radius=relative_radius, relative_angle=relative_angle,
                                 time=time, trial_dur=trial_dur, 
                                 action_v=action_v, action_w=action_w, 
                                 relative_radius_end=relative_radius_end,
                                 relative_angle_end=relative_angle_end, category=category)


# ruiyi example
# data_path=Path.cwd().parents[1]/'mkdata/bcm/bruno'
# ext=MonkeyDataExtractor(folder_path=data_path)
# trajectory=ext()
# trajectory.to_pickle(data_path/'test.pkl')


# my testing
from pathlib import Path
folderpath=Path("D:\mkdata\\bruno_pert")
sessions=[x for x in folderpath.iterdir() if x.is_dir()]


# convert and downsample, saving
for eachsession in sessions:
    ext=MonkeyDataExtractor(folder_path=eachsession)
    trajectory=ext()
    # trajectory.to_pickle(eachsession.parent/(str(eachsession)+'_full'))
    dfdownsample(trajectory,eachsession.parent/(str(eachsession)+'_ds'))
    print('file saved')

data_path=sessions[0]
ext=MonkeyDataExtractor(folder_path=data_path)
trajectory=ext()
# trajectory.to_pickle(data_path/'test.pkl')


# rewarded curve
data=list(trajectory.rewarded)
data=[1 if each else 0 for each in data]
x = [i for i in range(len(data))]
y = data
plt.plot(x, smooth(y,20), lw=2,label='reward')


# reward/trial curve
rewardrate=[r/d for r,d in zip(list(trajectory.rewarded),list(trajectory.trial_dur))]
x = [i for i in range(len(data))]
y = rewardrate
plt.plot(x, smooth(y,20), lw=2,label='reward rate')



# cat curve
data=list(trajectory.category)
vocab={'skip':0,'normal':1,"crazy":2,'lazy':3,'wrong_target':4}
data=[vocab[each] for each in data]
x = [i for i in range(len(data))]
y = data
# plt.plot(y)
plt.plot(x, smooth(y,20), lw=2,label='trial type')
plt.xlabel('trial number')
plt.ylabel('reward, reward rate and trial type')
plt.legend()
print(vocab)


x = [i for i in range(len(trajectory))]
y = trajectory.perturb_wpeak*trajectory.perturb_vpeak
plt.plot(x, smooth(y,20), lw=2,label='reward rate')




 
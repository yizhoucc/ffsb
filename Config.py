# this file is for the collection of parameters

# this is for log variance noise parameter
import numpy as np
from numpy import pi
import datetime
import os
import datetime
import torch
from torch.nn import LSTM
import numpy as np
from torch.optim import Adam
from pathlib import Path


class Config:
    def __init__(self):
        # --------------copy in use, new naming---------------------------
        self.dt = 0.1 
        self.episode_len = 100
        self.reward_amount = 100  
        self.terminal_vel = 0.05  
        self.goal_distance_range=   [0.1,1]
        self.gains_range =          [0.1,1.,pi/2-0.6,pi/2+0.6]
        self.std_range =            [0.01,1,0.01,1]
        self.goal_radius_range =    [0.129,0.131]
        # action cost
        self.mag_action_cost_range= [1e-3,1]
        self.dev_action_cost_range= [1e-3,1]
        # acc control, vt+1 = a*vt + b*ut
        self.tau_range=             [1e-2,1] #*5 to normalize
        self.init_uncertainty_range=[1e-8,1]
        self.cost_scale=1
        self.presist_phi=False
        self.agent_knows_phi=False
        self.dev_v_cost_range=[0,1]
        self.dev_w_cost_range=[0,1]
        # --------------previous in use, old version---------------------------
        self.WORLD_SIZE=1.0
        self.DELTA_T = self.dt
        self.EPISODE_LEN = self.episode_len
        self.REWARD = self.reward_amount  # for max reward
        self.TERMINAL_VEL = self.terminal_vel




class ConfigCore():
    def __init__(self):
        # network
        self.device = 'cpu'
        self.SEED_NUMBER = 0
        self.FC_SIZE = 256
        
        # RL
        self.GAMMA = 0.97
        self.TAU = 0.005
        self.POLICY_FREQ = 2
        self.policy_noise = 0.05
        self.policy_noise_clip = 0.1
        
        # optimzer
        self.optimzer = Adam
        self.lr = 3e-4
        self.eps = 1.5e-4
        self.decayed_lr = 5e-5
        
        # environment
        self.STATE_DIM = 5
        self.ACTION_DIM = 2
        self.POS_DIM = 3
        self.OBS_DIM = 2
        self.TARGET_DIM = 2
        self.TERMINAL_ACTION = 0.1
        self.DT = 0.1 # s
        self.EPISODE_TIME = 3 # s
        self.EPISODE_LEN = int(self.EPISODE_TIME / self.DT)
        self.REWARD_SCALE = 100
        self.LINEAR_SCALE = 400 # cm/unit
        self.goal_radius_range = np.array([65, 65]) / self.LINEAR_SCALE
        self.initial_radius_range = np.array([100, 400]) / self.LINEAR_SCALE
        self.relative_angle_range = np.deg2rad([-40, 40])
        self.process_gain_default = torch.tensor([200 / self.LINEAR_SCALE, torch.deg2rad(torch.tensor(90.))])
        
        # others
        self.filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_path = Path('..') / 'model'
        
        # For model-free belief
        self.BATCH_SIZE = 16
        self.MEMORY_SIZE = int(1e5)
        self.RNN = LSTM
        self.RNN_SIZE = 128
        self.BN_SIZE = 13
        
        # For model-based belief
        self.EKF_STATE_DIM = 15
        self.EKF_BATCH_SIZE = 1024
        self.EKF_MEMORY_SIZE = int(1e6)
    
    @property
    def data_path(self):
        return self.data_path_
    
    @data_path.setter
    def data_path(self, value):
        self.data_path_ = value / 'trained_agent'
        
    def save(self):
        torch.save(self.__dict__, self.data_path / f'{self.filename}_arg.pkl')
        
    def load(self, filename):
        self.__dict__ = torch.load(self.data_path / f'{filename}_arg.pkl')
        self.filename = filename

        
class ConfigNormal(ConfigCore):
    def __init__(self):
        super().__init__()
        self.task = 'normal'
        self.GAMMA = 0.95
        self.process_gain_range = None
        self.pro_noise_range = [0, 0.1]
        self.obs_noise_range = [0, 0.1]
        self.perturbation_velocity_range = None
        self.perturbation_duration = None
        self.perturbation_std = None
        self.perturbation_start_t_range = None
    
        
class ConfigNoise(ConfigCore):
    def __init__(self):
        super().__init__()
        self.task = 'noise'
        self.process_gain_range = None
        self.pro_noise_range = [0.3, 0.3] # proportional to process gain
        self.obs_noise_range = [0, 1]
        self.perturbation_velocity_range = None
        self.perturbation_duration = None
        self.perturbation_std = None
        self.perturbation_start_t_range = None
        
        
class ConfigNoiseControl(ConfigNoise):
    def __init__(self):
        super().__init__()
        self.task = 'noise_control'
        self.obs_noise_range = [0, 0]
        
        
class ConfigGain(ConfigCore):
    def __init__(self):
        super().__init__()
        self.task = 'gain'
        self.process_gain_range = [0.9, 1.1]
        self.pro_noise_range = None
        self.obs_noise_range = None
        self.perturbation_velocity_range = None
        self.perturbation_duration = None
        self.perturbation_std = None
        self.perturbation_start_t_range = None
        
        
class ConfigPerturb(ConfigCore):
    def __init__(self):
        super().__init__()
        self.task = 'perturbation'
        self.process_gain_range = None
        self.pro_noise_range = None
        self.obs_noise_range = None
        self.perturbation_velocity_range = np.hstack([np.array([-200, 200]) / self.LINEAR_SCALE, 
                                                      np.deg2rad([-120, 120])])
        self.perturbation_duration = 10 # steps
        self.perturbation_std = 2 # steps
        self.perturbation_start_t_range = [0, 10] # steps
        
        
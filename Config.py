# this file is for the collection of parameters

# this is for log variance noise parameter
import numpy as np
from numpy import pi
import datetime
import os


class Config:
    def __init__(self):
        # --------------copy in use, new naming---------------------------
        self.dt = 0.1 
        self.episode_len = 100
        self.reward_amount = 100  
        self.terminal_vel = 0.05  
        self.goal_distance_range=   [0.3,1]
        self.gains_range =          [0.25,1.,pi/4,pi/1]
        self.std_range =            [0.05,0.05*5,pi/80,pi/80*5]
        self.goal_radius_range =    [0.07, 0.2] 
        # action cost
        self.mag_action_cost_range= [1e-3,1]
        self.dev_action_cost_range= [1e-3,1]
        # acc control, vt+1 = a*vt + b*ut
        self.tau_range=             [1e-2,1] #*5 to normalize
        self.init_uncertainty_range=[1e-8,1]
        # --------------previous in use, old version---------------------------
        self.WORLD_SIZE=1.0
        self.DELTA_T = self.dt
        self.EPISODE_LEN = self.episode_len
        self.REWARD = self.reward_amount  # for max reward
        self.TERMINAL_VEL = self.terminal_vel



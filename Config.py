# this file is for the collection of parameters

# this is for log variance noise parameter
import numpy as np
from numpy import pi
import datetime
import os


class Config:
    def __init__(self):

        # --------------in use---------------------------
        self.WORLD_SIZE=1.0
        self.DELTA_T = 0.1 #0.01  # time to perform one action
        self.EPISODE_LEN = 70
        self.REWARD = 20  # for max reward
        self.TERMINAL_VEL = 0.3  # terminal velocity? # norm(action) that you believe as a signal to stop 0.1.
        self.goal_distance_range=   [0.1,1]
        self.gains_range =          [0.25,1.,pi/4,pi/1]
        self.std_range =            [0.05,0.05*5,pi/80,pi/80*5]
        self.goal_radius_range =    [0.1, 0.3] 
        # action cost
        self.mag_action_cost_range= [0.001,0.05]
        self.dev_action_cost_range= [0.001,0.5]
        # acc control, vt+1 = a*vt + b*ut
        self.tau_range=             [0.1,5]



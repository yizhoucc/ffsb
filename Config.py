# this file is for the collection of parameters

# this is for log variance noise parameter
import numpy as np
from numpy import pi
import datetime
import os


class Config:
    def __init__(self):

        # -------------not using--------------------------

        self.SEED_NUMBER = 0
        self.verbose=True
        self.WORLD_SIZE = 1.0  # 2.5 # size of the world?
        self.ACTION_DIM = 2 # velocity and direction?
        self.STATE_DIM = 29 #4  # 20 
        self.TOT_T = 2000000000  # total number of time steps for this code
        self.BATCH_SIZE = 64  # for replay memory (default:64)
        self.NUM_EPOCHS = 2 # for replay memory
        self.DISCOUNT_FACTOR = 0.99
        self.BOX_STEP_SIZE = 5e-1 # not used
        self.STD_STEP_SIZE = 2e-5  # 1e-4 action space noise (default: 2e-3)
        self.filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.data_path = os.getcwd()+'/'
        self.GOAL_RADIUS_STEP_SIZE = 1e-5



        # --------------in use---------------------------
        self.WORLD_SIZE=1.0
        self.DELTA_T = 0.1 #0.01  # time to perform one action
        self.EPISODE_LEN = 70
        self.REWARD = 10  # for max reward
        self.TERMINAL_VEL = 0.3  # terminal velocity? # norm(action) that you believe as a signal to stop 0.1.

        self.gains_range =          [0.25,1.,pi/4,pi/1]
        self.std_range =            [0.05,0.05*5,pi/80,pi/80*5]
        self.goal_radius_range =    [0.1, 0.3] 
        # action cost
        self.mag_action_cost_range= [0.001,0.05]
        self.dev_action_cost_range= [0.01,0.5]
        # acc control, vt+1 = a*vt + b*ut
        self.tau_range=             [0.1,5]
        




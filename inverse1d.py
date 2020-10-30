
import os
import warnings
warnings.filterwarnings('ignore')
from copy import copy
import time
import random
seed=3
random.seed(seed)
import torch
torch.manual_seed(seed)
import numpy as np
np.random.seed(int(seed))
from numpy import pi
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# -----------invser functions-------------
from InverseFuncs import trajectory, getLoss, reset_theta, theta_range,reset_theta_log, single_theta_inverse

# ---------loading env and agent----------
from stable_baselines import TD3
from FireflyEnv import ffac_1d
from Config import Config
arg = Config()

DISCOUNT_FACTOR = 0.99
arg.NUM_SAMPLES=2
arg.NUM_EP = 1000
arg.NUM_IT = 2 # number of iteration for gradient descent
arg.NUM_thetas = 1
arg.ADAM_LR = 0.05
arg.LR_STEP = 50
arg.LR_STOP = 0.02
arg.lr_gamma = 0.95
arg.PI_STD=1
arg.goal_radius_range=[0.1,0.3]
arg.TERMINAL_VEL = 0.025
arg.goal_radius_range=[0.15,0.3]
arg.std_range = [0.02,0.1,0.02,0.1]
arg.TERMINAL_VEL = 0.025  # terminal velocity? # norm(action) that you believe as a signal to stop 0.1.
arg.DELTA_T=0.2
arg.EPISODE_LEN=35

number_updates=100

# agent convert to torch model
import policy_torch
baselines_mlp_model =TD3.load('trained_agent/1d_easy1000000_9_25_5_14.zip')
agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[512,512],n_inputs=15,n_actions=1)

# loading enviorment, same as training
env=ffac_1d.FireflyTrue1d(arg)

true_theta=torch.Tensor([[0.3556],
        [0.0773],
        [0.5131],
        [0.2299],
        [0.1696],
        [4.5492],
        [0.0262],
        [0.0455]])

for i in range(10):
    filename=(str(time.localtime().tm_mday)+'_'+str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min))
    single_theta_inverse(arg, env, agent, filename, 
                    number_updates=number_updates,
                    true_theta=true_theta, 
                    phi=true_theta,
                    init_theta=None,
                    states=None, 
                    actions=None, 
                    trajectory_data=None,
                    use_H=False,
                    tasks=None,
                    is1d=True,                    
                    )
print('done')



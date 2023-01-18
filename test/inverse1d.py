
import os
import warnings
warnings.filterwarnings('ignore')
from copy import copy
import time
import random
seed=7
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

from FireflyEnv import ffac_1d
from Config import Config
arg = Config()

DISCOUNT_FACTOR = 0.99
arg.NUM_SAMPLES=30
arg.NUM_EP = 20
arg.NUM_IT = 1 # number of iteration for gradient descent
arg.NUM_thetas = 1
arg.ADAM_LR = 0.1
arg.LR_STEP = 20
arg.LR_STOP = 0.01
arg.lr_gamma = 0.95
arg.PI_STD=1
arg.goal_radius_range=[0.1,0.3]
arg.TERMINAL_VEL = 0.025
arg.goal_radius_range=[0.15,0.3]
arg.std_range = [0.02,0.1,0.02,0.1]
arg.TERMINAL_VEL = 0.025  # terminal velocity? # norm(action) that you believe as a signal to stop 0.1.
arg.DELTA_T=0.2
arg.EPISODE_LEN=35

number_updates=10000

# agent convert to torch model
# from stable_baselines import TD3
# import policy_torch
# baselines_mlp_model =TD3.load('trained_agent/1d_easy1000000_9_25_5_14.zip')
# agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[512,512],n_inputs=15,n_actions=1)

# load torch model
import TD3_torch
agent =TD3_torch.TD3.load('trained_agent/1d_1000000_9_16_22_20.zip')
agent=agent.actor.mu.cpu()

# loading enviorment, same as training
env=ffac_1d.FireflyTrue1d_cpu(arg)

# init_theta=torch.Tensor([[5.3127e-01],
#         [3.4570e-01],
#         [1.0000e-04],
#         [1.8348e-01],
#         [4.5674e+00],
#         [8.2376e-02],
#         [1.4962e-01]]).cuda()

# true_theta=torch.Tensor(
#         [[0.5],
#         [0.05],
#         [0.05],
#         [0.15],
#         [1],
#         [0.03],
#         [0.03]]).cuda()

for i in range(1):
    filename=(str(time.localtime().tm_mday)+'_'+str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min))
    single_theta_inverse(arg, env, agent, filename, 
                    number_updates=number_updates,
                    true_theta=None, 
                    phi=None,
                    init_theta=None,
                    states=None, 
                    actions=None, 
                    trajectory_data=None,
                    use_H=False,
                    tasks=None,
                    is1d=True,
                    gpu=False
                    # fixed_param_ind=[5,6],
                    # assign_true_param=[5,6],
                #     task=[torch.tensor([0.7]).cuda()],                    
                    )
print('done')



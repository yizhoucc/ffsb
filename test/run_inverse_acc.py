
import os
import warnings
warnings.filterwarnings('ignore')
from copy import copy
import time
import random
seed=time.time().as_integer_ratio()[0]
seed=1
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
from FireflyEnv import firefly_acc
from Config import Config
arg = Config()

DISCOUNT_FACTOR = 0.99
arg.NUM_SAMPLES=2
arg.NUM_EP = 200
arg.NUM_IT = 2 # number of iteration for gradient descent
arg.NUM_thetas = 1
arg.ADAM_LR = 0.1
arg.LR_STEP = 2
arg.LR_STOP = 0.003
arg.lr_gamma = 0.95
arg.PI_STD=1
arg.goal_radius_range=[0.1,0.3]
arg.TERMINAL_VEL = 0.025
number_updates=100


# agent convert to torch model
import policy_torch
baselines_mlp_model = TD3.load('trained_agent//acc_retrain_1000000_2_18_21_4.zip')
agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[128,128],n_inputs=30)

# loading enviorment, same as training
env=firefly_acc.FireflyAcc(arg)
# ---seting the env for inverse----
# TODO, move it to a function of env
env.agent_knows_phi=False


for i in range(10):
    filename=("test_acc_EP" + str(arg.NUM_EP) + "updates" + str(number_updates)+"lr"+str(arg.ADAM_LR)+'step'+str(arg.LR_STEP)
            +str(time.localtime().tm_mday)+'_'+str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min))
    single_theta_inverse(arg, env, agent, filename, 
                    number_updates=number_updates,
                    true_theta=None, phi=None,init_theta=None,
                    states=None, actions=None, tasks=None)




print('done')


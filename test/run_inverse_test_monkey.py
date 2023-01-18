
import os
import warnings
warnings.filterwarnings('ignore')
from copy import copy
import time
import random
seed=time.time().as_integer_ratio()[0]
seed=2
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
from stable_baselines import DDPG,TD3
from FireflyEnv import firefly_action_cost
from Config import Config
arg = Config()

DISCOUNT_FACTOR = 0.99
arg.NUM_SAMPLES=2
arg.NUM_EP = 100
arg.NUM_IT = 2 # number of iteration for gradient descent
arg.NUM_thetas = 1
arg.ADAM_LR = 0.1
arg.LR_STEP = 2
arg.LR_STOP = 0.002
arg.lr_gamma = 0.95
arg.PI_STD=1
arg.goal_radius_range=[0.1,0.3]
number_updates=500

# agent convert to torch model
import policy_torch
baselines_mlp_model = TD3.load('trained_agent//TD_action_cost_700000_8_19_21_56.zip')
agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[128,128],n_inputs=31)

# loading enviorment, same as training
env=firefly_action_cost.FireflyActionCost(arg)
# ---seting the env for inverse----
# TODO, move it to a function of env
env.agent_knows_phi=False



#------------------------new function part----------------




filename=("EP" + str(arg.NUM_EP) + "updates" + str(number_updates)+"lr"+str(arg.ADAM_LR)+'step'+str(arg.LR_STEP)
        +str(time.localtime().tm_mday)+'_'+str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min))
# single_theta_inverse(arg, env, agent, filename, 
#                 number_updates=number_updates,
#                 true_theta=None, phi=None,init_theta=None,
#                 states=None, actions=None, tasks=None)



for i in range(10):
    filename=("monkey_EP" + str(arg.NUM_EP) + "updates" + str(number_updates)+"lr"+str(arg.ADAM_LR)+'step'+str(arg.LR_STEP)
            +str(time.localtime().tm_mday)+'_'+str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min))

    single_theta_inverse(arg, env, agent, filename, 
                    number_updates=number_updates,
                    true_theta=None, phi=None,init_theta=None,
                    states=None, actions=None, tasks=None)


print('done')






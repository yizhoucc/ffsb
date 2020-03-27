#import torch
#import torch.nn as nn
#from torch.autograd import grad

import os



import warnings
warnings.filterwarnings('ignore')

from inverse_model import Dynamic, Inverse
# agent
from stable_baselines import DDPG
# env
from FireflyEnv import ffenv
from Config import Config
from Inverse_Config import Inverse_Config
import torch

from InverseFuncs import trajectory, getLoss, reset_theta, theta_range
# from single_inverse_part_theta import single_inverse
from single_inverse import single_inverse


from DDPGv2Agent import Agent
from FireflyEnv import Model # firefly_task.py
from Inverse_Config import Inverse_Config

# read configuration parameters
arg = Inverse_Config()
# fix random seed
import random
random.seed(arg.SEED_NUMBER)
import torch
torch.manual_seed(arg.SEED_NUMBER)
if torch.cuda.is_available():
    torch.cuda.manual_seed(arg.SEED_NUMBER)
import numpy as np
from numpy import pi
np.random.seed(arg.SEED_NUMBER)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# if gpu is to be used
CUDA = False
device = "cpu"

#CUDA = torch.cuda.is_available()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#import multiprocessing
#num_cores = multiprocessing.cpu_count()

DISCOUNT_FACTOR = 0.99
arg.gains_range = [0.8, 1.2, pi/5, 3*pi/10]
arg.std_range = [1e-2, 2, 1e-2, 2]
arg.WORLD_SIZE = 1.0
arg.goal_radius_range = [0.2* arg.WORLD_SIZE, 0.5* arg.WORLD_SIZE]
arg.DELTA_T = 0.1
arg.EPISODE_TIME = 1  # # maximum length of time for one episode. if monkey can't firefly within this time period, new firefly comes
arg.EPISODE_LEN = int(arg.EPISODE_TIME / arg.DELTA_T)
arg.NUM_SAMPLES=2

arg.NUM_EP = 500
arg.NUM_IT = 50 # number of iteration for gradient descent
arg.NUM_thetas = 15
filename='test'


# agent 
import policy_torch
baselines_mlp_model = DDPG.load("DDPG_theta")
agent = policy_torch.copy_mlp_weights(baselines_mlp_model)


# env
#  = Model(arg) # build an environment
env=ffenv.FireflyEnv(arg)

env.max_goal_radius = arg.goal_radius_range[1] # use the largest world size for goal radius
env.box = arg.WORLD_SIZE



true_theta_log = []
true_loss_log = []
true_loss_act_log = []
true_loss_obs_log = []
final_theta_log = []
stderr_log = []
result_log = []

for num_thetas in range(1):

    # true theta
    true_theta = reset_theta(arg.gains_range, arg.std_range, arg.goal_radius_range)
    # true theta for DDPG_theta
    true_theta=torch.tensor([1.0537, 0.7328, 0.7053, 1.2038, 0.9661, 0.8689, 0.2930, 1.9330, 0.2000])
    true_theta_log.append(true_theta.data.clone())
    x_traj, obs_traj, a_traj, _ = trajectory(agent, true_theta, env, arg, arg.gains_range, arg.std_range,
                                             arg.goal_radius_range, arg.NUM_EP)  # generate true trajectory
    print('got trajectory!')
    true_loss, true_loss_act, true_loss_obs = getLoss(agent, x_traj, a_traj, true_theta, env, arg.gains_range, arg.std_range, arg.PI_STD,
                        arg.NUM_SAMPLES)  # this is the lower bound of loss?
    print('got loss!') # this loss is completely due to obs noise  
    #true_loss_log.append(true_loss)
    #true_loss_act_log.append(true_loss_act)
    #true_loss_obs_log.append(true_loss_obs)

    print("true loss:{}".format(true_loss))
    print("true act loss:{}".format(true_loss_act))
    print("true obs loss:{}".format(true_loss_obs))

    print("true_theta:{}".format(true_theta))


    result = single_inverse(true_theta, arg, env, agent, x_traj, a_traj, filename, num_thetas)
    #result = single_inverse(true_theta, arg, env, agent, x_traj, a_traj, filename, num_thetas)

    result_log.append(result)


    torch.save(result_log, '../firefly-inverse-data/data/' + filename + "EP" + str(arg.NUM_EP) + str(
        np.around(arg.PI_STD, decimals=2))+"sample"+str(arg.NUM_SAMPLES) +"IT"+ str(arg.NUM_IT) + '_LR_parttheta_result.pkl')


print('done')
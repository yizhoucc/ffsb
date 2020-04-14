
import os
import warnings
warnings.filterwarnings('ignore')
from stable_baselines import DDPG
from FireflyEnv import ffenv_sigmoid
from Config import Config
from Inverse_Config import Inverse_Config
import torch
from InverseFuncs import *
from single_inverse import single_inverse
from DDPGv2Agent import Agent
from Inverse_Config import Inverse_Config
arg = Inverse_Config()
import time
import random
random.seed(arg.SEED_NUMBER)
import torch
torch.manual_seed(time.time().as_integer_ratio()[0])
import numpy as np
from numpy import pi
np.random.seed(arg.SEED_NUMBER)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DISCOUNT_FACTOR = 0.99
arg.gains_range = [0.8, 1.2, pi/5, 3*pi/10]
# arg.gains_range = [np.log(0.8), np.log(1.2), np.log(pi/5), np.log(3*pi/10)]
# arg.gains_range=[inverse_sigmoid(torch.Tensor([0.001])).item(),inverse_sigmoid(torch.Tensor([0.999])).item(),inverse_sigmoid(torch.Tensor([0.001])).item(),inverse_sigmoid(torch.Tensor([0.999])).item()]
arg.std_range = [1e-2, 0.3, 1e-2, 0.2]
# arg.std_range = [inverse_sigmoid(torch.Tensor([0.001])).item(),inverse_sigmoid(torch.Tensor([0.999])).item(),inverse_sigmoid(torch.Tensor([0.001])).item(),inverse_sigmoid(torch.Tensor([0.999])).item()]
arg.WORLD_SIZE = 1.0
arg.goal_radius_range=[0.2,0.5]
# arg.goal_radius_range = [inverse_sigmoid(torch.Tensor([0.001])).item(),inverse_sigmoid(torch.Tensor([0.999])).item()]
arg.DELTA_T = 0.1
arg.EPISODE_TIME = 1  # # maximum length of time for one episode. if monkey can't firefly within this time period, new firefly comes
arg.EPISODE_LEN = int(arg.EPISODE_TIME / arg.DELTA_T)
arg.NUM_SAMPLES=2
arg.NUM_EP = 100
arg.NUM_IT = 200 # number of iteration for gradient descent
arg.NUM_thetas = 1
arg.ADAM_LR = 0.2
arg.LR_STEP = 2
arg.LR_STOP = 50
arg.lr_gamma = 0.95

# agent 
import policy_torch
baselines_mlp_model = DDPG.load("DDPG_theta")
agent = policy_torch.copy_mlp_weights(baselines_mlp_model)
# agent=baselines_mlp_model
env=ffenv_sigmoid.FireflyEnv(arg)
env.max_goal_radius = (arg.goal_radius_range[1]) # use the largest world size for goal radius
env.box = arg.WORLD_SIZE
env.reset_theta=False

true_theta_log = []
true_loss_log = []
true_loss_act_log = []
true_loss_obs_log = []
final_theta_log = []
stderr_log = []
result_log = []

filename="test sigmoid"

for num_thetas in range(arg.NUM_thetas):

    # true theta
    true_theta = reset_theta_sig(arg.gains_range, arg.std_range, arg.goal_radius_range)
    print('true theta: ',true_theta)
    true_theta_log.append(true_theta.data.clone())
    x_traj, obs_traj, a_traj, _ = trajectory(agent, true_theta, env, arg, arg.gains_range, arg.std_range,
                                             arg.goal_radius_range, arg.NUM_EP)  # generate true trajectory


    xlens=[]
    for x in x_traj:
        xlens.append(len(x))
    # print('check len')

    theta=reset_theta_sig()
    theta=true_theta.clone()

    result = single_inverse(true_theta, arg, env, agent, x_traj,obs_traj, a_traj, filename, num_thetas,theta=theta)

    savename=('../firefly-inverse-data/data/' + filename + str(num_thetas) + "EP" + str(arg.NUM_EP) + str(
        np.around(arg.PI_STD, decimals=2))+"sample"+str(arg.NUM_SAMPLES) +"IT"+ str(arg.NUM_IT) + '_LR_parttheta_result.pkl')
    torch.save(result, savename)


print('done')
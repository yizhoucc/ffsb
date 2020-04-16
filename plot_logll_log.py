
import os
import warnings
warnings.filterwarnings('ignore')
from stable_baselines import DDPG
from FireflyEnv import ffenv_lognoise
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
import math
np.random.seed(arg.SEED_NUMBER)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DISCOUNT_FACTOR = 0.99
arg.gains_range = [0.8, 1.2, pi/5, 3*pi/10]
arg.std_range = [np.log(1e-3), np.log(0.3), np.log(1e-3), np.log(0.2)]
arg.WORLD_SIZE = 1.0
arg.goal_radius_range=[0.2,0.5]
arg.DELTA_T = 0.1
arg.EPISODE_TIME = 1  # # maximum length of time for one episode. if monkey can't firefly within this time period, new firefly comes
arg.EPISODE_LEN = int(arg.EPISODE_TIME / arg.DELTA_T)
arg.NUM_SAMPLES=2
arg.NUM_EP = 200
arg.NUM_IT = 200 # number of iteration for gradient descent
arg.NUM_thetas = 1
arg.ADAM_LR = 0.2
arg.LR_STEP = 2
arg.LR_STOP = 50
arg.lr_gamma = 0.95
x_length=11
y_length=11
x_var=1 # a 0 - 8 value that index a paramter in theta
y_var=5
vartick=0.2

# agent 

baselines_mlp_model = DDPG.load("DDPG_theta")
agent = policy_torch.copy_mlp_weights(baselines_mlp_model)
# agent=baselines_mlp_model
env=ffenv_lognoise.FireflyEnv(arg)
env.max_goal_radius = (arg.goal_radius_range[1]) # use the largest world size for goal radius
env.box = arg.WORLD_SIZE
env.reset_theta=False


xpurt = torch.Tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
ypurt = torch.Tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
xpurt[x_var]=vartick
ypurt[y_var]=vartick

# true theta
true_theta = reset_theta(arg.gains_range, arg.std_range, arg.goal_radius_range)
print('true theta: ',true_theta)
true_theta_log.append(true_theta.data.clone())
x_traj, obs_traj, a_traj, _ = trajectory(agent, true_theta, env, arg, arg.gains_range, arg.std_range,
                                            arg.goal_radius_range, arg.NUM_EP)  # generate true trajectory



loss_act_log=np.zeros((x_length,y_length))
purt_log=np.zeros((x_length,y_length))
xticks=[]
yticks=[]
for xi in range(x_length):
    for yi in range(y_length):
    # purt=torch.ones(9)*0.1

        purt=xpurt*(xi-math.floor(x_length/2))+ypurt*(yi-math.floor(y_length/2))
        theta = nn.Parameter(true_theta.data.clone()+purt)

        
        ini_theta = theta.data.clone()
        print('initial theta: ',ini_theta)

        loss, loss_act, loss_obs = getLoss(agent, x_traj, obs_traj,a_traj, theta, env, arg.gains_range, arg.std_range, arg.PI_STD, arg.NUM_SAMPLES)
        loss_act_log[xi,yi]=(loss_act)
        if xi==0:
            yticks.append(purt[y_var].item())
        if yi==0:
            xticks.append(purt[x_var].item())


print('done')

param_name=['pro gain v', 'pro gain w', 'pro noise v', 'pro noise w',
            'obs gain v', 'obs gain w', 'obs noise v' 'obs noise w', 
            'goal radius']


import matplotlib.pyplot as plt
plt.figure(0,figsize=(8,8))
plt.suptitle('Loss surface of {} and {}'.format(param_name[x_var],param_name[y_var]), fontsize=24,y=1)
plt.xticks(xticks)
plt.yticks(yticks)
plt.xlabel(param_name[x_var],fontsize=15)
plt.ylabel(param_name[y_var],fontsize=15)
plt.imshow(loss_act_log,origin='lower',extent=[xticks[0],xticks[-1],yticks[0],yticks[-1]])


# from scipy.interpolate import griddata
# xi = np.linspace(min(xticks),max(xticks),(len(loss_act_log)/3))
# yi = np.linspace(yticks.min(),yticks.max(),(len(loss_act_log)/3))
# zi = griddata((xticks, yticks), loss_act_log, (xi[None,:], yi[:,None]), method='nearest')
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(xticks, yticks, loss_act_log,cmap='viridis', edgecolor='none')

import os
import warnings
warnings.filterwarnings('ignore')
from copy import copy
import time
import random
seed=0
random.seed(seed)
import torch
torch.manual_seed(seed)
import numpy as np
np.random.seed(int(seed))
from numpy import pi
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from InverseFuncs import trajectory, getLoss, reset_theta, theta_range,reset_theta_log, single_theta_inverse
from FireflyEnv import ffacc_real
from monkey_functions import *
from Config import Config
arg = Config()
arg.presist_phi=False
arg.agent_knows_phi=False
arg.goal_distance_range=[0.4,1]
arg.gains_range =[0.05,1.5,pi/4,pi/1]
arg.goal_radius_range=[0.05,0.3]
arg.std_range = [0.08,0.3,pi/80,pi/80*5]
arg.mag_action_cost_range= [0.0001,0.001]
arg.dev_action_cost_range= [0.0001,0.005]
arg.TERMINAL_VEL = 0.2
arg.DELTA_T=0.1
arg.EPISODE_LEN=100
arg.agent_knows_phi=False
DISCOUNT_FACTOR = 0.99
arg.sample=10
arg.batch = 100
# arg.NUM_SAMPLES=1
# arg.NUM_EP = 1
arg.NUM_IT = 1 
arg.NUM_thetas = 1
arg.ADAM_LR = 0.01
arg.LR_STEP = 20
arg.LR_STOP = 0.5
arg.lr_gamma = 0.095
arg.PI_STD=1
arg.presist_phi=False
number_updates=10000
# load torch model
import TD3_torch
agent =TD3_torch.TD3.load('trained_agent/new_1dt_1drealsysvel_1000000_4_6_13_8_3_2_1_16')
agent=agent.actor.mu.cpu()
# loading enviorment, same as training
env=ffacc_real.Firefly_real_vel(arg)

with open("C:/Users/24455/Desktop/DataFrame_normal",'rb') as f:
    df = pickle.load(f)
# df=df[df.isFullOn==False]
# df=df[100:210]
df=df[:-100]
states, actions, tasks=monkey_trajectory(df,new_dt=0.1, goal_radius=65,factor=0.002)

theta=env.reset_task_param()

phi=torch.tensor(   [[0.4],
                    [pi],
                    [0.0001],
                    [0.0001],
                    [0.0001],#
                    [0.0001],# note used, obs noise
                    [0.13],
                    [0.0001],#
                    [0.0001]#
])

theta=torch.tensor( [[0.4],
                    [pi],
                    [0.1],
                    [0.1],
                    [0.1],
                    [0.1],
                    [0.13],
                    [0.001],
                    [0.001]
])

theta_estimation=torch.tensor(
[[0.20835843682289124],
  [1.8204913139343262],
                    [0.1],
                    [0.1],
                    [0.1],
                    [0.1],
 [0.10974789410829544],
                    [0.001],
                    [0.001]]
  )

for i in range(1):
    filename=(str(time.localtime().tm_mday)+'_'+str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min))
    single_theta_inverse(arg, env, agent, filename, 
                    number_updates=number_updates,
                    true_theta=None, 
                    phi=phi,
                    init_theta=theta,
                    trajectory_data=(states, actions, tasks),
                    use_H=False,
                    is1d=False,
                    gpu=False,
                    # fixed_param_ind=[1,2,5,6],
                    # assign_true_param=[1,2,5,6],
                    action_var=0.1
                    )




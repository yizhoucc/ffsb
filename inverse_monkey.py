

import warnings
warnings.filterwarnings('ignore')
from copy import copy
import time
import random
from stable_baselines3 import TD3
seed=0
random.seed(seed)
import torch
torch.manual_seed(seed)
import numpy as np
np.random.seed(int(seed))
from numpy import pi
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from InverseFuncs import *
from FireflyEnv import ffacc_real
from monkey_functions import *
import TD3_torch
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
arg.dev_v_cost_range= [0.1,0.5]
arg.dev_w_cost_range= [0.1,0.5]
arg.TERMINAL_VEL = 0.1
arg.DELTA_T=0.1
arg.EPISODE_LEN=100
arg.agent_knows_phi=False
DISCOUNT_FACTOR = 0.99
arg.NUM_thetas = 1
arg.LR_STEP = 20
arg.lr_gamma = 1
arg.PI_STD=1
arg.presist_phi=False
arg.cost_scale=1

arg.fixed_param_ind=[6]
arg.action_var=0.01
arg.ADAM_LR = 0.001
arg.LR_STOP=0.0001
arg.sample = 11 # samples for 1 mk trial
arg.batch = 10
arg.NUM_IT = 100 # iteration of all data
number_updates=1 # update per batch


# continue from prev inverse
# from InverseFuncs import *
# inverse_data=load_inverse_data('brune_m25_7_40')
# theta_trajectory=inverse_data['theta_estimations']
# true_theta=inverse_data['true_theta']
# theta_estimation=theta_trajectory[-1]
# theta=torch.tensor(theta_estimation)
# phi=torch.tensor(inverse_data['phi'])
env=ffacc_real.FireFlyPaper(arg)
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()

print('loading data')
note='testdcont'
with open("C:/Users/24455/Desktop/bruno_pert_downsample",'rb') as f:
        df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df=df[df.target_r>250]
df=df[df.floor_density==0.005]
# floor density are in [0.0001, 0.0005, 0.001, 0.005]
df=df[:-100]

print('process data')
states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
print('done process data')
# theta=env.reset_task_param()

phi=torch.tensor([[0.5],
        [pi/2],
        [0.001],
        [0.001],
        [0.001],
        [0.001],
        [0.13],
        [0.001],
        [0.001],
        [0.001],
        [0.001],
])

theta=torch.tensor([[0.5],   
        [1.6],   
        [0.5],   
        [0.5],   
        [0.5],   
        [0.5],   
        [0.13],   
        [0.1],   
        [0.1],   
        [0.1],   
        [0.1]])

theta_estimation=torch.tensor(
[[0.8353011012077332],
 [2.23036789894104],
 [0.4379126727581024],
 [0.065259650349617],
 [0.0010000000474974513],
 [0.005447442177683115],
 [0.13],
 [0.5920934677124023],
 [0.6594878435134888],
 [1.00711989402771],
 [0.7560240030288696]]
 )

print('start inverse')
for i in range(1):
    filename=(note+str(time.localtime().tm_mday)+'_'+str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min))
    monkey_inverse(arg, env, agent, filename, 
                    number_updates=number_updates,
                    true_theta=None, 
                    phi=phi,
                    init_theta=theta_estimation,
                    trajectory_data=(states, actions, tasks),
                    use_H=False,
                    is1d=False,
                    gpu=False,
                    # fixed_param_ind=[1,2,5,6],
                    # assign_true_param=[1,2,5,6],
                    action_var=arg.action_var, # how precise we want to predict the action
                    batchsize=arg.batch,
                    fixed_param_ind=arg.fixed_param_ind,
                    )




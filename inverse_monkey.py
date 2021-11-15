

import warnings
warnings.filterwarnings('ignore')
from copy import copy
import time
import random
from stable_baselines3 import SAC,PPO,TD3
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
arg.LR_STOP = 0.5
arg.lr_gamma = 0.5
arg.PI_STD=1
arg.presist_phi=False
arg.cost_scale=1

arg.ADAM_LR = 0.007
arg.LR_STOP=0.001
arg.sample = 40 # samples for 1 mk trial
arg.batch = 200
arg.NUM_IT = 30 # iteration of all data
number_updates=1 # update per batch
# load torch model

# continue from prev inverse
# from InverseFuncs import *
# inverse_data=load_inverse_data('brune_m25_7_40')
# theta_trajectory=inverse_data['theta_estimations']
# true_theta=inverse_data['true_theta']
# theta_estimation=theta_trajectory[-1]
# theta=torch.tensor(theta_estimation)
# phi=torch.tensor(inverse_data['phi'])


# agent_=PPO.load('trained_agent/ppo_60000_9_8_1_21.zip')
# agent = lambda x : agent_.predict(x)

agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()

# agent_ =SAC.load('trained_agent/re_iticosttimes_100000_3_17_0_40_11.zip')
# agent_=agent_.actor.cpu()
# agent = lambda x : agent_.forward(x, deterministic=True)


# loading enviorment, same as training
env=ffacc_real.FireFlyPaper(arg)

# with open("C:/Users/24455/Desktop/DataFrame_normal",'rb') as f:
#     df = pickle.load(f)
# df=df[df.isFullOn==False]
# # df=df[110:130]
# df=df[:-100]
# states, actions, tasks=monkey_trajectory(df,new_dt=0.1, goal_radius=65,factor=0.002)
print('loading data')
with open("C:/Users/24455/Desktop/bruno_pert_downsample",'rb') as f:
        df = pickle.load(f)
        df=df[df.full_on==False]
# df=df[110:130]
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

theta=torch.tensor([[0.3419],
        [1.1250],
        [0.2165],
        [0.2362],
        [0.3720],
        [0.1858],
        [0.0021],
        [0.4856],
        [1.0716],
        [0.0183],
        [0.2193]])

# theta_estimation=torch.tensor(
# [[0.2207319438457489],
#  [1.062300205230713],
#  [0.32934996485710144],
#  [0.1929050236940384],
#  [0.19170257449150085],
#  [0.1894093006849289],
#  [0.16225792467594147],
#  [0.05502069741487503],
#  [0.6376186013221741],
#  [0.7159334421157837]]
#          )
note='brunompert'
print('start inverse')
for i in range(1):
    filename=(note+str(time.localtime().tm_mday)+'_'+str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min))
    monkey_inverse(arg, env, agent, filename, 
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
                    action_var=0.001, # how precise we want to predict the action
                    batchsize=arg.batch,
                    )




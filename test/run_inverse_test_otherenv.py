
import os
import warnings
warnings.filterwarnings('ignore')
from copy import copy
import time
import random
seed=time.time().as_integer_ratio()[0]
seed=6
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
arg.NUM_EP = 200
arg.NUM_IT = 1 # number of iteration for gradient descent
arg.NUM_thetas = 1
arg.ADAM_LR = 0.1
arg.LR_STEP = 2
arg.LR_STOP = 0.003
arg.lr_gamma = 0.95
arg.PI_STD=1
arg.goal_radius_range=[0.1,0.3]
number_updates=1

# agent convert to torch model
import policy_torch
baselines_mlp_model = TD3.load('trained_agent//TD_action_cost_700000_8_19_21_56.zip')
agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[128,128],n_inputs=31)

# loading enviorment, same as training
env=firefly_action_cost.FireflyActionCost(arg)
# ---seting the env for inverse----
# TODO, move it to a function of env
env.agent_knows_phi=False



#----------------manul run part-------------
# true_theta_log = []
# true_loss_log = []
# true_loss_act_log = []
# true_loss_obs_log = []
# final_theta_log = []
# stderr_log = []
# result_log = []

# save_dict={'theta_estimations':[]}
# filename="test_seed"+str(seed)

# # use serval theta to inverse
# for num_thetas in range(arg.NUM_thetas):

#     # make sure phi and true theta stay the same 
#     true_theta = env.reset_task_param()
#     env.presist_phi=True
#     env.reset(phi=true_theta,theta=true_theta) # here we first testing teacher truetheta=phi case
#     theta=env.reset_task_param()
#     phi=env.reset_task_param()

#     save_dict['true_theta']=true_theta.data.clone().tolist()
#     save_dict['phi']=true_theta.data.clone().tolist()
#     save_dict['inital_theta']=theta.data.clone().tolist()


#     for num_update in range(number_updates):
#         states, actions, tasks = trajectory(
#             agent, phi, true_theta, env, arg.NUM_EP)
            
#         result = single_inverse(true_theta, phi, arg, env, agent, states, actions, tasks, filename, num_thetas, initial_theta=theta)
#         save_dict['theta_estimations'].append(result.tolist())
#         savename=('inverse_data/' + filename + "EP" + str(arg.NUM_EP) + "updates" + str(number_update)+"sample"+str(arg.NUM_SAMPLES) +"IT"+ str(arg.NUM_IT) + '.pkl')
#         torch.save(save_dict, savename)
#         print(result)

#------------------------new function part----------------
filename=("EP" + str(arg.NUM_EP) + "updates" + str(number_updates)+"lr"+str(arg.ADAM_LR)+'step'+str(arg.LR_STEP)
        +str(time.localtime().tm_mday)+'_'+str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min))
# single_theta_inverse(arg, env, agent, filename, 
#                 number_updates=number_updates,
#                 true_theta=None, phi=None,init_theta=None,
#                 states=None, actions=None, tasks=None)


true_theta=env.reset_task_param()
phi=env.reset_task_param()

for i in range(10):
    filename=("fromtrue_1true_EP" + str(arg.NUM_EP) + "updates" + str(number_updates)+"lr"+str(arg.ADAM_LR)+'step'+str(arg.LR_STEP)
            +str(time.localtime().tm_mday)+'_'+str(time.localtime().tm_hour)+'_'+str(time.localtime().tm_min))
    single_theta_inverse(arg, env, agent, filename, 
                    number_updates=number_updates,
                    true_theta=true_theta, phi=phi,init_theta=true_theta,
                    states=None, actions=None, tasks=None)
print('done')



'''
logic of this script:

the main file,
    load agent
    load env,
    load arg used in training,
        we are using the param ranges here, 
        this is passing down, to avoid estimation of theta go off bound
    load arg used in inverse
        including learning rate, epoch, data size, etc
    
then call the trajectory, to get teacher trajectory

in trajectory
    input:
        env and agent
        how many we want,
        phi, the teacher knows phi
        theta, here theta is same as phi

after having teacher trajectory, 
pass into single inverse

in single inverse,
    input:
        theta, the initialized theta
        phi,
        args, such as epoch, learning rate
    we call get loss, to generate another set of trajectory,
    which using estimation of theta, not true theta (phi here)

    then, we apply the gradient
    finally we have the final estimation of theta.





'''


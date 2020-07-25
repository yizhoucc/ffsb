
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
from InverseFuncs import trajectory, getLoss, reset_theta, theta_range,reset_theta_log, single_inverse


# ---------loading env and agent----------
from stable_baselines import DDPG,TD3
from FireflyEnv import ffenv_new_cord
from Config import Config
arg = Config()

DISCOUNT_FACTOR = 0.99
arg.NUM_SAMPLES=2
arg.NUM_EP = 1000
arg.NUM_IT = 10 # number of iteration for gradient descent
arg.NUM_thetas = 1
arg.ADAM_LR = 0.007
arg.LR_STEP = 2
arg.LR_STOP = 50
arg.lr_gamma = 0.95
arg.PI_STD=1
arg.goal_radius_range=[0.05,0.2]


# agent convert to torch model
import policy_torch
baselines_mlp_model = TD3.load("TD_95gamma_mc_smallgoal_500000_9_24_1_6.zip")
agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[128,128])

# loading enviorment, same as training
env=ffenv_new_cord.FireflyAgentCenter(arg)
# ---seting the env for inverse----
# TODO, move it to a function of env
env.agent_knows_phi=False

true_theta_log = []
true_loss_log = []
true_loss_act_log = []
true_loss_obs_log = []
final_theta_log = []
stderr_log = []
result_log = []
number_update=100
filename="testnew"

# use serval theta to inverse
for num_thetas in range(arg.NUM_thetas):

    # make sure phi and true theta stay the same 
    true_theta = env.reset_task_param()
    env.presist_phi=True
    env.reset(phi=true_theta,theta=true_theta) # here we first testing teacher truetheta=phi case
    true_theta_log.append(true_theta.data.clone())

    theta=env.reset_task_param()
    phi=true_theta.data.clone()
    for num_update in range(number_update):
        states, actions, tasks = trajectory(
            agent, phi, true_theta, env, arg.NUM_EP)
            
        result = single_inverse(true_theta, phi, arg, env, agent, states, actions, tasks, filename, num_thetas, initial_theta=theta)

        # savename=('../firefly-inverse-data/data/' + filename + str(num_thetas) + "EP" + str(arg.NUM_EP) + str(
        #     np.around(arg.PI_STD, decimals=2))+"sample"+str(arg.NUM_SAMPLES) +"IT"+ str(arg.NUM_IT) + '_LR_parttheta_result.pkl')
        # torch.save(result, savename)
        print(result)

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
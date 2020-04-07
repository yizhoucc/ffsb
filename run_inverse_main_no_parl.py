
import os
import warnings
warnings.filterwarnings('ignore')
from stable_baselines import DDPG
from FireflyEnv import ffenv
from Config import Config
from Inverse_Config import Inverse_Config
import torch
from InverseFuncs import trajectory, getLoss, reset_theta, theta_range,reset_theta_log
from single_inverse import single_inverse
from DDPGv2Agent import Agent
from FireflyEnv import Model # firefly_task.py
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
# arg.gains_range = [0.8, 1.2, pi/5, 3*pi/10]
# arg.gains_range = [np.log(0.8), np.log(1.2), np.log(pi/5), np.log(3*pi/10)]
arg.gains_range = [np.log(0.8-0.799), np.log(1.2-0.799), np.log(pi/5-0.628), np.log(3*pi/10-0.628)]
# arg.std_range = [1e-2, 0.3, 1e-2, 0.2]
arg.std_range = [np.log(1e-3), np.log(0.3), np.log(1e-3), np.log(0.2)]
arg.WORLD_SIZE = 1.0
arg.goal_radius_range = [np.log(0.2* arg.WORLD_SIZE), np.log(0.5* arg.WORLD_SIZE)]
arg.DELTA_T = 0.1
arg.EPISODE_TIME = 1  # # maximum length of time for one episode. if monkey can't firefly within this time period, new firefly comes
arg.EPISODE_LEN = int(arg.EPISODE_TIME / arg.DELTA_T)
arg.NUM_SAMPLES=2
arg.NUM_EP = 50
arg.NUM_IT = 200 # number of iteration for gradient descent
arg.NUM_thetas = 10
arg.ADAM_LR = 0.3
arg.LR_STEP = 2
arg.LR_STOP = 50
arg.lr_gamma = 0.95

# agent 
import policy_torch
baselines_mlp_model = DDPG.load("DDPG_theta")
agent = policy_torch.copy_mlp_weights(baselines_mlp_model)
# agent=baselines_mlp_model
env=ffenv.FireflyEnv(arg)
env.max_goal_radius = np.log(arg.goal_radius_range[1]) # use the largest world size for goal radius
env.box = arg.WORLD_SIZE
env.reset_theta=False

true_theta_log = []
true_loss_log = []
true_loss_act_log = []
true_loss_obs_log = []
final_theta_log = []
stderr_log = []
result_log = []

filename="all log mean obs1"

for num_thetas in range(arg.NUM_thetas):

    # true theta
    true_theta = reset_theta_log(arg.gains_range, arg.std_range, arg.goal_radius_range)
    # true theta for DDPG_theta
    # true_theta=torch.tensor([1.0537, 0.7328, 0.7053, 1.2038, 0.9661, 0.8689, 0.2930, 1.5330, 0.3500])
    # true_theta=torch.tensor([1.0537, 0.7328, 0.1, 0.1, 0.7, 0.8, 0.1, 0.1, 0.3500])
    # true_theta[2:4]=0.3
    # true_theta[4:6]=1
    # true_theta[6:8]=0.1
    # true_theta[6:8]=0.1
    print('true theta: ',true_theta)
    true_theta_log.append(true_theta.data.clone())
    x_traj, obs_traj, a_traj, _ = trajectory(agent, true_theta, env, arg, arg.gains_range, arg.std_range,
                                             arg.goal_radius_range, arg.NUM_EP)  # generate true trajectory

    '''
        # print('got trajectory!')
        # true_loss, true_loss_act, true_loss_obs = getLoss(agent, x_traj, a_traj, true_theta, env, arg.gains_range, arg.std_range, arg.PI_STD,
        #                     arg.NUM_SAMPLES)  # this is the lower bound of loss?
        # print('got loss!') # this loss is completely due to obs noise  
        # #true_loss_log.append(true_loss)
        # #true_loss_act_log.append(true_loss_act)
        # #true_loss_obs_log.append(true_loss_obs)

        # print("true loss:{}".format(true_loss))
        # print("true act loss:{}".format(true_loss_act))
        # print("true obs loss:{}".format(true_loss_obs))

        # print("true_theta:{}".format(true_theta))

    '''

    xlens=[]
    for x in x_traj:
        xlens.append(len(x))
    print('check len')

    result = single_inverse(true_theta, arg, env, agent, x_traj,obs_traj, a_traj, filename, num_thetas)

    savename=('../firefly-inverse-data/data/' + filename + str(num_thetas) + "EP" + str(arg.NUM_EP) + str(
        np.around(arg.PI_STD, decimals=2))+"sample"+str(arg.NUM_SAMPLES) +"IT"+ str(arg.NUM_IT) + '_LR_parttheta_result.pkl')
    torch.save(result, savename)


print('done')
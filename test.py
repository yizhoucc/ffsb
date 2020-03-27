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


def reset_theta(gains_range, std_range, goal_radius_range, Pro_Noise = None, Obs_Noise = None):
    pro_gains = torch.zeros(2)
    obs_gains = torch.zeros(2)

    pro_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  # [proc_gain_vel]
    pro_gains[1] = torch.zeros(1).uniform_(gains_range[2], gains_range[3])  # [proc_gain_ang]
    obs_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  # [obs_gain_vel]
    obs_gains[1] = torch.zeros(1).uniform_(gains_range[2], gains_range[3])  # [obs_gain_ang]
    goal_radius = torch.zeros(1).uniform_(goal_radius_range[0], goal_radius_range[1])

    if Pro_Noise is None:
       pro_noise_stds = torch.zeros(2)
       pro_noise_stds[0] = torch.zeros(1).uniform_(std_range[0], std_range[1])  # [proc_vel_noise]
       pro_noise_stds[1] = torch.zeros(1).uniform_(std_range[2], std_range[3])  # [proc_ang_noise]
    else:
        pro_noise_stds = Pro_Noise


    if Obs_Noise is None:
        obs_noise_stds = torch.zeros(2)
        obs_noise_stds[0] = torch.zeros(1).uniform_(std_range[0], std_range[1])  # [obs_vel_noise]
        obs_noise_stds[1] = torch.zeros(1).uniform_(std_range[2], std_range[3])  # [obs_ang_noise]
    else:
        obs_noise_stds = Obs_Noise

    theta = torch.cat([pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius])
    return theta

inverse_arg=Inverse_Config()
env_arg=Config()
# policy=DDPG.load("DDPG_ff")

# print(os.getcwd())

phi=reset_theta(inverse_arg.gains_range, inverse_arg.std_range, inverse_arg.goal_radius_range)

theta=reset_theta(inverse_arg.gains_range, inverse_arg.std_range, inverse_arg.goal_radius_range)



# testing env
teacher_env=ffenv.FireflyEnv(env_arg)
agent_env=ffenv.FireflyEnv(env_arg)

teacher_env.assign_presist_phi(phi) 
# print(teacher_env.theta)
agent_env.assign_presist_phi(theta) 
# print(agent_env.theta)
# print('state',agent_env.state)


# testing torch agent
import policy_torch
baselines_mlp_model = DDPG.load("DDPG_ff_mlp_32")
policy = policy_torch.copy_mlp_weights(baselines_mlp_model)


# testing dynamics
dynamic=Dynamic(policy, teacher_env,agent_env)

# a,b,c,_,_=dynamic.run_episode()
# print("a,b,c",a,b,c)
# a,b,c,d,e=dynamic.collect_data(3)



# testing inverse model
model=Inverse(dynamic=dynamic,arg=inverse_arg)
while True:

    model.learn(20000)

print("end")


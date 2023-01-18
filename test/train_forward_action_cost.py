
from stable_baselines3.td3.policies import MlpPolicy
from TD3_torch import TD3
from FireflyEnv import firefly_action_cost
from Config import Config
arg=Config()
import numpy as np
from numpy import pi
import time
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from reward_functions import reward_singleff

action_noise = NormalActionNoise(mean=np.zeros(2), sigma=float(0.1) * np.ones(2))
arg.WORLD_SIZE=550
arg.gains_range =[50,200,0.5,2]
arg.goal_radius_range=[50,200]
arg.std_range = [0.5,0.6,0.05,0.06]
arg.mag_action_cost_range= [0.0001,0.00011]
arg.dev_action_cost_range= [0.0001,0.0005]
arg.TERMINAL_VEL = 0.3
arg.DELTA_T=0.2
arg.EPISODE_LEN=100
env=firefly_action_cost.FireflyActionCost(arg)
env.max_distance=0.5


# model=TD3.load('TD3_95gamma_500000_0_17_10_53.zip',
model = TD3(MlpPolicy,
            env, 
            tensorboard_log="./Tensorboard/",
            # policy_kwargs={'optimizer_kwargs':{'weight_decay':0.001}},
            buffer_size=int(1e6),
            batch_size=512,
            verbose=True,
            action_noise=action_noise,

            )

train_time=700000 
env.cost_scale=0.1
for i in range(10): 
    namestr=("trained_agent/TD_action_cost_{}_{}_{}_{}_{}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    )) 
    model.learn(total_timesteps=int(train_time/10),tb_log_name=namestr)
    model.save(namestr)
    env.max_distance=env.max_distance+0.1
    env.cost_scale=env.cost_scale+0.02

env.goal_radius_range=[0.1,0.3]
env.EPISODE_LEN=40
for i in range(10):  
    namestr=("trained_agent/TD_action_cost_sg_{}_{}_{}_{}_{}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))
    model.learn(total_timesteps=int(train_time/10),tb_log_name=namestr)
    model.save(namestr)
    env.cost_scale=env.cost_scale+0.02


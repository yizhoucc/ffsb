import gym
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines import TD3
from FireflyEnv import ffenv_new_cord
from Config import Config
arg=Config()
import numpy as np
from numpy import pi
import time
import torch
from ff_policy.policy_selu import SoftPolicy
import tensorflow as tf
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from reward_functions import reward_singleff

action_noise = NormalActionNoise(mean=np.zeros(2), sigma=float(0.5) * np.ones(2))

arg.goal_radius_range=[0.15,0.3]
env_new_cord=ffenv_new_cord.FireflyAgentCenter(arg)
env_new_cord.max_distance=0.5


# model=TD3.load('TD3_95gamma_500000_0_17_10_53.zip',
model = TD3(MlpPolicy,
            env_new_cord, 
            verbose=1,
            tensorboard_log="./DDPG_tb/",
            action_noise=action_noise,
            buffer_size=int(1e6),
            batch_size=512,
            learning_rate=3e-4, 
            train_freq=100,
            # policy_kwargs={'act_fun':tf.nn.selu,'layers':[256,256,64,32]}
            policy_kwargs={'layers':[128,128]},
            policy_delay=2, 
            learning_starts=1000, 
            gradient_steps=100, 
            random_exploration=0., 
            gamma=0.98, 

            tau=0.005, 
            target_policy_noise=0.1, 
            target_noise_clip=0.1,
            _init_setup_model=True, 
            full_tensorboard_log=False, 
            seed=None, 
            n_cpu_tf_sess=None,            
            )

train_time=500000 

for i in range(10):  
    model.learn(total_timesteps=int(train_time/10))
    model.save("TD_95gamma_mc_{}_{}_{}_{}_{}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))
    env_new_cord.max_distance=env_new_cord.max_distance+0.1

env_new_cord.goal_radius_range=[0.1,0.3]
env_new_cord.EPISODE_LEN=40
for i in range(10):  
    model.learn(total_timesteps=int(train_time/10))
    model.save("TD_95gamma_mc_smallgoal_{}_{}_{}_{}_{}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))


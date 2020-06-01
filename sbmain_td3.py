import gym
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines import TD3, HER
from FireflyEnv import ffenv,ffenv_new_cord,ffenv_original
from Config import Config
arg=Config()
import numpy as np
from numpy import pi
import time
import torch
from ff_policy.policy_selu import SoftPolicy
import tensorflow as tf
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from DDPGv2Agent.rewards import *

arg.std_range = [0.00025,0.00025*5,pi/4000,pi/4000*5]# [vel min, vel max, ang min, ang max]
arg.gains_range=[0.025,0.1,pi/40,pi/10]
arg.DELTA_T = 0.1
arg.EPISODE_TIME = 4
arg.EPISODE_LEN=40
arg.goal_radius_range=[0.05,0.2]



arg.REWARD=10
action_noise = NormalActionNoise(mean=np.zeros(2), sigma=float(0.5) * np.ones(2))
# env=ffenv_original.FireflyEnv(arg)
# env_skip=ffenv.FireflyEnv(arg,kwargs={'let_skip':True})
# env_skip_real_reward=ffenv.FireflyEnv(arg,kwargs={'reward_function':discrete_reward,'let_skip':True})
env_new_cord=ffenv_new_cord.FireflyAgentCenter(arg)


model = TD3(MlpPolicy,
            env_new_cord, 
            verbose=1,
            tensorboard_log="./DDPG_tb/",
            action_noise=action_noise,
            buffer_size=int(1e5),
            batch_size=1024,
            learning_rate=0.0007, 
            train_freq=500,
            # policy_kwargs={'act_fun':tf.nn.selu,'layers':[256,256,64,32]}
            policy_kwargs={'layers':[64,64]},
            policy_delay=2, 
            learning_starts=10000, 
            gradient_steps=200, 
            random_exploration=0., 
            gamma=0.915, 

            tau=0.005, 
            target_policy_noise=0.2, 
            target_noise_clip=0.5,
            _init_setup_model=True, 
            full_tensorboard_log=False, 
            seed=None, 
            n_cpu_tf_sess=None,            
            )
# train_time=30000
# model.learn(total_timesteps=train_time)
# model.learn(total_timesteps=1000000)
# model.set_env(env)
env_new_cord.setup(arg,max_distance=0.2)
model.learn(total_timesteps=10000000)

model.save("TD3_{}".format(time.localtime().tm_mday))



env.close()
import gym
from stable_baselines.ddpg.policies import LnMlpPolicy, MlpPolicy
from stable_baselines import DDPG

from Config import Config

import gym
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


model = DDPG(MlpPolicy,
            
            env_new_cord, verbose=1,
            tensorboard_log="./DDPG_tb/",
            full_tensorboard_log=False,
            action_noise=action_noise,
            batch_size=512,
            buffer_size=int(1e5),

            # gamma=0.99, 
            # memory_policy=None, 
            # eval_env=None, 
            # nb_train_steps=100,
            # nb_rollout_steps=100, 
            # nb_eval_steps=100, 
            # param_noise=None, 
            # normalize_observations=False, 
            # tau=0.001, 
            # param_noise_adaption_interval=50,
            # normalize_returns=False, 
            # enable_popart=False, 
            # observation_range=(-np.inf, np.inf), 
            # critic_l2_reg=0.,
            # return_range=(-np.inf, np.inf), 
            actor_lr=1e-4/3, critic_lr=1e-3/3, 
            # clip_norm=None, 
            reward_scale=2,
            # render=False, 
            # render_eval=False, 
            # memory_limit=None, 
            # random_exploration=0.01, 
            # _init_setup_model=True, 
            # # policy_kwargs={'layers':[256,256,64,32]},
            # seed=None, n_cpu_tf_sess=None
            
            )
train_time=500000

# model=DDPG.load('DDPG_test1000000_3 20 13 59',
# kwargs=
# {
# 'tensorboard_log':True,
# 'gamma':0.95,
# 'batch_size':512,
# 'buffer_size':int(1e5),

# })
# model.set_env(env_new_cord)


for i in range(5):  
    model.learn(total_timesteps=train_time/5)
    model.save("../trained_agent/DDPG_95gamma_{}_{}_{}_{}_{}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))
    env.max_distance=env.max_distance+0.1

env.goal_radius_range=[0.1,0.3]
env.EPISODE_LEN=40


for i in range(5):  
    model.learn(total_timesteps=train_time/5)
    model.save("../trained_agent/DDPG_95gamma_smallgoal_{}_{}_{}_{}_{}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))


# model.learn(total_timesteps=train_time)
# model.save("DDPG_LQC_relu_simple{} {} {} {}".format(train_time,
#     str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#     ))


# env.close()
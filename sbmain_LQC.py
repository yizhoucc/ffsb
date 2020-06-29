import gym
from stable_baselines.ddpg.policies import LnMlpPolicy, MlpPolicy
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DDPG
from FireflyEnv import ffenv
from Config import Config

import gym
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
# from DDPGv2Agent.rewards import *
from reward_functions import reward_singleff

import numpy as np
import time
import torch
from ff_policy.policy_selu import SoftPolicy
import tensorflow as tf
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=float(0.5) * np.ones(2))


# arg.goal_radius_range=[0.05,0.2]



env_new_cord=ffenv_new_cord.FireflyAgentCenter(arg,
    {'reward_function':reward_singleff.state_gaussian_reward,
    'max_distance':0.5
    })


model = DDPG(LnMlpPolicy,
            
            env_new_cord, verbose=1,
            tensorboard_log="./DDPG_tb/",
            full_tensorboard_log=False,
            action_noise=action_noise,

            gamma=0.95, 
            memory_policy=None, 
            eval_env=None, 
            nb_train_steps=50,
            nb_rollout_steps=100, 
            nb_eval_steps=100, 
            param_noise=None, 
            normalize_observations=False, 
            tau=0.001, 
            batch_size=512, 
            param_noise_adaption_interval=50,
            normalize_returns=False, 
            enable_popart=False, 
            observation_range=(-np.inf, np.inf), 
            critic_l2_reg=0.,
            return_range=(-np.inf, np.inf), 
            actor_lr=1e-4, critic_lr=1e-3, 
            clip_norm=None, 
            reward_scale=1,
            render=False, 
            render_eval=False, 
            memory_limit=None, 
            buffer_size=int(1e5),
            random_exploration=0.01, 
            _init_setup_model=True, 
            policy_kwargs={'act_fun':tf.nn.selu,'layers':[256,256,64,32]},
            seed=None, n_cpu_tf_sess=None)
train_time=1000000

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
    model.learn(total_timesteps=train_time/10)
    model.save("DDPG_test_{}_{} {} {} {}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))

env_new_cord=ffenv_new_cord.FireflyAgentCenter(arg,
    {'reward_function':reward_singleff.state_gaussian_reward,
    })
model.set_env(env_new_cord)


for i in range(5):  
    model.learn(total_timesteps=train_time/10)
    model.save("DDPG_test_far_gaol{}_{} {} {} {}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))


# model.learn(total_timesteps=train_time)
# model.save("DDPG_LQC_relu_simple{} {} {} {}".format(train_time,
#     str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#     ))


# env.close()
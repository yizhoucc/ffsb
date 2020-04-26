import gym
from stable_baselines.ddpg.policies import FeedForwardPolicy, MlpPolicy
from stable_baselines import A2C
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DDPG
from FireflyEnv import ffenv
from Config import Config
arg=Config()


import numpy as np
import time
import torch
from ff_policy.policy_selu import SoftPolicy
import tensorflow as tf
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=float(0.5) * np.ones(2))
env=ffenv.FireflyEnv(arg)

model = DDPG(MlpPolicy,
            
            env, verbose=1,tensorboard_log="./DDPG_tb/",full_tensorboard_log=False,action_noise=action_noise,

            gamma=0.99, 
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
            reward_scale=1.2,
            render=False, 
            render_eval=False, 
            memory_limit=None, 
            buffer_size=1e6, 
            # random_exploration=0.01, 
            _init_setup_model=True, 
            policy_kwargs={'act_fun':tf.nn.selu,'layers':[256,256,64,32]},
            seed=None, n_cpu_tf_sess=None)
train_time=1000000

for i in range(10):
    model.learn(total_timesteps=train_time/10)
    # model.learn(total_timesteps=1000000)
    model.save("DDPG_selu_skip_96reward{}_{} {} {} {}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))


# train_time=100000
# model.learn(total_timesteps=train_time)
# model.save("DDPG_selu_nonoise{} {} {} {}".format(train_time,
#     str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#     ))


env.close()
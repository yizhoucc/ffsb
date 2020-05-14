import gym
from stable_baselines.td3.policies import LnMlpPolicy
from stable_baselines import TD3
from FireflyEnv import ffenv
from Config import Config
arg=Config()
import numpy as np
import time
import torch
from ff_policy.policy_selu import SoftPolicy
import tensorflow as tf
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=float(0.3) * np.ones(2))
env=ffenv.FireflyEnv(arg)

model = TD3(LnMlpPolicy,
            env, 
            verbose=1,
            tensorboard_log="./DDPG_tb/",
            action_noise=action_noise,
            buffer_size=1e6,
            batch_size=1024,
            policy_delay=500,
            # learning_starts =1000,
            # target_policy_noise=0.3,
            # target_noise_clip =0.1,
            # policy_kwargs={'act_fun':tf.nn.selu,'layers':[256,256,64,32]}
            )
train_time=300000


model.learn(total_timesteps=train_time)
# model.learn(total_timesteps=1000000)

model.save("TD3_{}".format(train_time))



env.close()
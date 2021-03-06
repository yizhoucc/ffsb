import gym
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines import DDPG
from FireflyEnv import ffenv_new_cord
from Config import Config
arg=Config()
import numpy as np
import time
import torch
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=float(0.2) * np.ones(2))
arg.std_range=[0.0001,0.001,0.0001,0.001] 
env=ffenv_new_cord.FireflyEnv(arg)
model = DDPG(MlpPolicy, env, verbose=1,tensorboard_log="./DDPG_tb/",action_noise=action_noise,

            gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50,
            nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, normalize_observations=False, 
            tau=0.001, batch_size=128, param_noise_adaption_interval=50,
            normalize_returns=False, enable_popart=False, observation_range=(-5., 5.), critic_l2_reg=0.,
            return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
            render=False, render_eval=False, memory_limit=None, buffer_size=int(1e5), random_exploration=0.0,
            _init_setup_model=True, policy_kwargs=None,
            full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)


train_time=1000000

for i in range(10):
    model.learn(total_timesteps=train_time/10)
    # model.learn(total_timesteps=1000000)
    model.save("DDPG_new_cord{}_{} {} {} {}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))


env.close()
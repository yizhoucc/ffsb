import gym
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines import TD3, HER,DDPG
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
from stable_baselines.ddpg.policies import MlpPolicy
# # check env
# from stable_baselines.common.env_checker import check_env
# # env=gym.make('FF-v0')
# env=ffenv.FireflyEnv(arg)
# # env = CustomEnv(arg1, ...)
# # It will check your custom environment and output additional warnings if needed
# check_env(env)

# extra define may need or may not (used to need but now have sb)
# env.model.max_goal_radius = goal_radius_range[1] # use the largest world size for goal radius
# state_dim = env.state_dim
# action_dim = env.action_dim
# MAX_EPISODE = 1000
# std = 0.01
# noise = Noise(action_dim, mean=0., std=std)
# tot_t = 0.
# episode = 0.




# model = A2C.load("a2c_ff")
# model.set_env(env)
# obs = env.reset()

# training
# std = 0.4 # this is for action space noise for exploration
# mu = np.ones(2) * 0 # action dim =2
# n = np.random.randn(2)
# noise=mu+std*n
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
action_noise = NormalActionNoise(mean=np.zeros(2), sigma=float(0.5) * np.ones(2))

arg.REWARD=10
# arg.std_range = [1e-2, 0.1, 1e-2, 0.1]# [vel min, vel max, ang min, ang max]
env_new_cord=ffenv_new_cord.FireflyAgentCenter(arg)

env=ffenv.FireflyEnv(arg)
# model = DDPG(LnMlpPolicy, env, verbose=1,tensorboard_log="./",action_noise=action_noise)
model = DDPG(MlpPolicy, env_new_cord, verbose=1,tensorboard_log="./DDPG_tb/",action_noise=action_noise,

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
            observation_range=(-5., 5.), 
            critic_l2_reg=0.,
            return_range=(-np.inf, np.inf), 
            actor_lr=1e-4, 
            critic_lr=1e-3, 
            clip_norm=None, 
            reward_scale=1.,
            render=False, 
            render_eval=False, 
            memory_limit=None, 
            buffer_size=int(1e5), 
            random_exploration=0.0,
            _init_setup_model=True, 
            policy_kwargs=None,
            full_tensorboard_log=False, 
            seed=None, n_cpu_tf_sess=1)


# env.assign_presist_phi(torch.tensor([1.,2.,3.,2.,1.,2.,3.,1.,1.]))
# env.reset()
# # start=time.time()
# print(env.theta)
model.learn(total_timesteps=1000000)
# env_skip=ffenv.FireflyEnv(arg,kwargs={'let_skip':True})
# model.set_env(env_skip)
# model.learn(total_timesteps=1000000)
model.save("DDPG_test{}_ {} {}".format(
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))
# (tensor([0.9873, 0.7121]), tensor([1.7995, 0.3651]), tensor([0.8017, 0.9060]), tensor([1.8397, 1.9815]), tensor([0.2000]))
# 200 0000000
# print('training',time.time()-start)
# model.save("DDPG_theta")

# eval
# model = DDPG.load("DDPG_ff")
# obs = env.reset()
# # del model # remove to demonstrate saving and loading
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     # if rewards != 0:
#     #     print(rewards)
    # if done:
               # obs = env.reset()
#     print(rewards)
#     # env.render()
#     # env.Brender(b, x, arg.WORLD_SIZE, goal_radius)

# start=time.time()
# for i in range(30000): # no rest, 45 s
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
# print('testing',time.time()-start)

# for i in range(30000): # reset, 105 s
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     obs = env.reset()
# print('testing stoeps',time.time()-start)

env.close()
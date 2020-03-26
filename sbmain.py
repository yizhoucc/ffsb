import gym
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines import A2C
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DDPG
from FireflyEnv import ffenv
from Config import Config
arg=Config()
import numpy as np
import time
import torch
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
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(2), sigma=float(0.5) * np.ones(2))
env=ffenv.FireflyEnv(arg)
# model = DDPG(LnMlpPolicy, env, verbose=1,tensorboard_log="./",action_noise=action_noise)
model = DDPG(MlpPolicy, env, verbose=1,tensorboard_log="./DDPG_tb1/",action_noise=action_noise,

            gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50,
            nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, normalize_observations=False, 
            tau=0.001, batch_size=128, param_noise_adaption_interval=50,
            normalize_returns=False, enable_popart=False, observation_range=(-5., 5.), critic_l2_reg=0.,
            return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
            render=False, render_eval=False, memory_limit=None, buffer_size=50000, random_exploration=0.0,
            _init_setup_model=True, policy_kwargs=None,
            full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)


# env.assign_presist_phi(torch.tensor([1.,2.,3.,2.,1.,2.,3.,1.,1.]))
# env.reset()
# # start=time.time()
model.learn(total_timesteps=200000)
# print('training',time.time()-start)
# model.save("DDPG_ff_mlp_32")

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
import gym
import time
from stable_baselines.ddpg.policies import LnMlpPolicy, MlpPolicy
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DDPG
from FireflyEnv import ffenv_new_cord
from Config import Config
arg=Config()
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from reward_functions import reward_singleff

goal_radius_range=[0.1,0.25]

env=ffenv_new_cord.FireflyAgentCenter(arg,
# {'goal_radius_range':goal_radius_range,
{'reward_function':reward_singleff.belief_reward,
'max_distance':0.5})
model=DDPG.load('DDPG_reward0.97_500000_4 16 20 47',
tensorboard_log="./DDPG_tb/",
            full_tensorboard_log=False,
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
            actor_lr=1e-4/4, critic_lr=1e-3/4, 
            # clip_norm=None, 
            reward_scale=2)


model.set_env(env)
train_time=500000

for i in range(5):  
    model.learn(total_timesteps=train_time/5)
    model.save("DDPG_{}_{}_{}_{}_{}_{}".format(str(goal_radius_range),train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))


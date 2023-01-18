from stable_baselines3.td3.policies import MlpPolicy
from TD3_torch import TD3
from Config import Config
arg=Config()
import numpy as np
from numpy import pi
import time
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from reward_functions import reward_singleff
from FireflyEnv import ffacc_real

action_noise = NormalActionNoise(mean=0., sigma=float(0.3))
arg.init_action_noise=0.5
arg.goal_distance_range=[0.3,1] 
arg.mag_action_cost_range= [0.1,1.]
arg.dev_action_cost_range= [0.1,1.]
arg.dev_v_cost_range= [0.1,1.]
arg.dev_w_cost_range= [0.1,1.]
arg.gains_range =[0.35,0.45,pi/2-0.1,pi/2+0.1]
arg.goal_radius_range=[0.07,0.2]
arg.std_range = [0.01,1,0.01,1]
arg.reward_amount=100
arg.terminal_vel = 0.05
arg.dt=0.1
arg.episode_len=100
arg.training=True
arg.presist_phi=False
arg.agent_knows_phi=True
arg.cost_scale=1
env=ffacc_real.FireFlyReady(arg)
env.no_skip=True
modelname=None
# modelname='re_re_re_re_nocost_0_1_1_1'
note='re' 
from stable_baselines3 import SAC,PPO


if modelname is None:
   #td3
    model = TD3(MlpPolicy,
        env,
        buffer_size=int(1e6),
        batch_size=512,
        learning_rate=5e-4,
        learning_starts= 1000,
        tau= 0.005,
        gamma= 0.99,
        train_freq = 4,
        gradient_steps = -1,
        # n_episodes_rollout = 1,
        action_noise= action_noise,
        optimize_memory_usage = False,
        policy_delay = 2,
        target_policy_noise = 0.2,
        target_noise_clip = 0.5,
        tensorboard_log = None,
        create_eval_env = False,
        policy_kwargs = {'net_arch':[256,256]},
        verbose = 0,
        seed = None,
        device = "cpu",
        )
    train_time=50000
    for i in range(1,11):  
        env.cost_scale=1e-5
        if i==1:
            for j in range(1,5): 
                env.noise_scale=0.2*j
                namestr= ("trained_agent/td3_{}_{}_{}_{}_{}".format(train_time,i,
                str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
                ))
                model.learn(train_time)
                model.save(namestr)
        namestr= ("trained_agent/td3_{}_{}_{}_{}_{}".format(train_time,i,
        str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
        ))
        model.learn(train_time)
        model.save(namestr)
else:
    for i in range(1,11): 
        env.noise_scale=1
        action_noise = NormalActionNoise(mean=0., sigma=float(0.3-0.02*i))
        model = TD3.load('./trained_agent/'+modelname,
            env,
            action_noise=action_noise,
            learning_rate=5e-4
            )
        train_time=50000
        env.cost_scale=(1/20*i)**2
        namestr= ("trained_agent/{}_{}_{}".format(note,modelname,i))
        model.learn(train_time)
        model.save(namestr)

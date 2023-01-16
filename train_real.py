
from stable_baselines3.td3.policies import MlpPolicy
from Config import Config
arg=Config()
import numpy as np
from numpy import pi
import time
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from FireflyEnv import ffacc_real
import torch

arg.init_action_noise=0.5
arg.goal_distance_range=[0.15,1] 
arg.mag_action_cost_range= [0.01,1.]
arg.dev_action_cost_range= [0.01,1.]
arg.dev_v_cost_range= [0.01,1.]
arg.dev_w_cost_range= [0.01,1.]
arg.gains_range =[0.1,1.5,pi/2-0.6,pi/2+0.6]
arg.goal_radius_range=[0.129,0.131]
arg.std_range = [0.01,2,0.01,2]
arg.reward_amount=100
arg.terminal_vel = 0.05
arg.dt=0.1
arg.episode_len=50
arg.training=True
arg.presist_phi=False
arg.agent_knows_phi=True
arg.cost_scale=1
env=ffacc_real.FireFlyPaper(arg)
env.no_skip=True
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))        
modelname=None
modelname='rerere_52000_6_6_18_30_2_20'
note='8' 
from stable_baselines3 import TD3

if modelname is None:
    model = TD3(MlpPolicy,
        env,
        buffer_size=int(1e6),
        batch_size=512,
        learning_rate=7e-4,
        learning_starts= 2000,
        tau= 0.005,
        gamma= 0.96,
        train_freq = 4,
        # gradient_steps = -1,
        # n_episodes_rollout = 1,
        action_noise= action_noise,
        # optimize_memory_usage = False,
        policy_delay = 6,
        # target_policy_noise = 0.2,
        # target_noise_clip = 0.5,
        tensorboard_log = None,
        # create_eval_env = False,
        policy_kwargs = {'net_arch':[64,64],'activation_fn':torch.nn.ReLU},
        verbose = 0,
        seed = 42,
        # device = "cuda",
        )
    train_time=52000
    for i in range(1,200):  
        env.reward_ratio=1
        env.noise_scale=1
        env.cost_scale=min(0.003*i,0.03)
        if i==1:
            for j in range(1,11): 
                # arg.std_range = [0.01,0.1*j,0.01,0.1*j]
                env.terminal_vel=(11-j)*0.05
                env.noise_scale=0.1*j
                env.cost_scale=0.0
                namestr= ("trained_agent/pre_{}_{}_{}_{}_{}_{}".format(note,train_time,j,
                str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
                ))
                model.learn(train_time)
                model.save(namestr)
        namestr= ("trained_agent/{}_{}_{}_{}_{}_{}".format(note,train_time,i,
        str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
        ))
        model.learn(train_time)
        model.save(namestr)
else:
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))  
    model = TD3.load('./trained_agent/'+modelname,
            env,
            train_freq = 4,
            learning_starts= 2000,
            action_noise= action_noise,
            policy_delay = 6,
            buffer_size=int(1e6),
            batch_size=512,
            learning_rate=7e-4,
            tau= 0.005,
            gamma= 0.96,
            tensorboard_log = "./Tensorboard/{}{}".format(note,modelname),
            verbose = 0,
            seed = 42,
            # device = "cuda",
            )
    for i in range(1,200): 
        env.reward_ratio=1
        env.noise_scale=1
        train_time=520000
        env.cost_scale=min(0.002*i,0.1)
        j=9
        env.terminal_vel=(11-j)*0.05
        env.noise_scale=0.1*j

        namestr= ("trained_agent/{}{}_{}".format(note,modelname,i))
        model.learn(train_time)
        model.save(namestr)








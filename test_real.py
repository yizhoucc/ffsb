

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

action_noise = NormalActionNoise(mean=0., sigma=float(0.5))

arg.goal_distance_range=[0.4,1]
# arg.gains_range =[0.1,1,pi/4,pi/1]
# arg.goal_radius_range=[0.1,0.3]
# arg.std_range = [0.01,0.3,pi/80,pi/80*5]

arg.mag_action_cost_range= [0.0001,0.001]
arg.dev_action_cost_range= [0.0001,0.005]
arg.TERMINAL_VEL = 0.1
arg.DELTA_T=0.2
arg.EPISODE_LEN=100
arg.training=True
arg.presist_phi=False
arg.agent_knows_phi=True
env=ffacc_real.Firefly_real_vel(arg)
# modelname=None
modelname='1dt_1drealsysvel_1000000_4_6_13_8_3_2_1'
note='new'
if modelname is None:
    model = TD3(MlpPolicy,
        env,
        tensorboard_log="./Tensorboard/",
        buffer_size=int(1e6),
        batch_size=512,
        device='cpu',
        verbose=True,
        action_noise=action_noise,
        )
    for i in range(10):  
        namestr= ("trained_agent/nonoise_acc_control_retrain_{}_{}_{}_{}_{}".format(train_time,i,
        str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
        ))
        model.learn(total_timesteps=int(train_time/10),tb_log_name=namestr)
        model.save(namestr)
else:
    arg.goal_distance_range=[0.4,1]
    arg.gains_range =[0.05,2,pi/4,pi/0.5]
    arg.goal_radius_range=[0.03,0.3]
    arg.std_range = [0.001,0.3,pi/800,pi/80*5]
    arg.mag_action_cost_range= [0.0001,0.2]
    arg.dev_action_cost_range= [0.0001,0.2]
    env=ffacc_real.Firefly_real_vel(arg)
    model = TD3.load('./trained_agent/'+modelname,
        env,
        tensorboard_log="./Tensorboard/",
        buffer_size=int(1e6),
        batch_size=512,
        device='cpu',
        verbose=True,
        action_noise=action_noise,
        )
    train_time=1000000 
    for i in range(100):  
        namestr= ("trained_agent/{}_{}_{}".format(note,modelname,i))
        model.learn(total_timesteps=int(train_time/10),tb_log_name=namestr)
        model.save(namestr)



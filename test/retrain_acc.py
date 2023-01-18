
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines import TD3
from FireflyEnv import firefly_acc
from Config import Config
arg=Config()
import numpy as np
from numpy import pi
import time
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from reward_functions import reward_singleff
action_noise = NormalActionNoise(mean=np.zeros(2), sigma=float(0.3) * np.ones(2))
arg.goal_radius_range=[0.15,0.3]
arg.TERMINAL_VEL = 0.025  # terminal velocity? # norm(action) that you believe as a signal to stop 0.1.
env=firefly_acc.FireflyAcc(arg)

modelname='trained_agent/acc_retrain_1000000_4_18_11_34.zip'

model=TD3.load(modelname,
            env, 
            verbose=1,
            tensorboard_log="./DDPG_tb/",
            action_noise=action_noise,
            buffer_size=int(1e6),
            batch_size=512,
            learning_rate=1e-4, 
            train_freq=100,
            policy_kwargs={'layers':[128,128]},
            policy_delay=2, 
            learning_starts=1000, 
            gradient_steps=100, 
            random_exploration=0., 
            gamma=0.98, 
            tau=0.005, 
            target_policy_noise=0.1, 
            target_noise_clip=0.1,
            _init_setup_model=True, 
            full_tensorboard_log=False, 
            seed=None, 
            n_cpu_tf_sess=None,            
            )


model.set_env(env)
train_time=1000000

for i in range(5):  
    model.learn(total_timesteps=int(train_time/5))
    model.save("trained_agent/acc_retrain_{}_{}_{}_{}_{}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))


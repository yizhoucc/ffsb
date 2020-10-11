
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines import TD3
from TD3_test import TD3_ff
from FireflyEnv import firefly_accac
from Config import Config
arg=Config()
import numpy as np
from numpy import pi
import time
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from reward_functions import reward_singleff


action_noise = NormalActionNoise(mean=np.zeros(2), sigma=float(0.1) * np.ones(2))

arg.goal_radius_range=[0.15,0.3]
arg.std_range = [0.02,0.3,0.02,0.3]
arg.TERMINAL_VEL = 0.025  # terminal velocity? # norm(action) that you believe as a signal to stop 0.1.
arg.DELTA_T=0.2
arg.EPISODE_LEN=35

env=firefly_accac.FireflyAccAc(arg)
# modelname='trained_agent/TD_acc_control_pretrain_1000000_4_3_23_32.zip'
modelname=None
if modelname is None: # new train
    model = TD3_ff(MlpPolicy,
            env, 
            verbose=1,
            tensorboard_log="./DDPG_tb/",
            action_noise=action_noise,
            buffer_size=int(1e6),
            batch_size=512,
            learning_rate=3e-4, 
            train_freq=100,
            # policy_kwargs={'act_fun':tf.nn.selu,'layers':[256,256,64,32]}
            policy_kwargs={'layers':[512,512]},
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
else: # retrain
    model = TD3_ff.load(modelname,
                    env, 
                    verbose=1,
                    tensorboard_log="./DDPG_tb/",
                    action_noise=action_noise,
                    buffer_size=int(1e6),
                    batch_size=512,
                    learning_rate=3e-4, 
                    train_freq=100,
                    # policy_kwargs={'act_fun':tf.nn.selu,'layers':[256,256,64,32]}
                    # policy_kwargs={'layers':[128,128]},
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

train_time=1000000 
env.cost_scale=0.05
for i in range(10):  
    namestr= ("trained_agent/accac_final_{}_{}_{}_{}_{}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))
    model.learn(total_timesteps=int(train_time/10),tb_log_name=namestr)
    model.save(namestr)



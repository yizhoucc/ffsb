
# from stable_baselines.td3.policies import MlpPolicy
from stable_baselines3.td3.policies import MlpPolicy

# from stable_baselines3 import TD3


# from TD3_test import TD3_ff
from TD3_torch import TD3
from FireflyEnv import ffac_1d
from Config import Config
arg=Config()
import numpy as np
from numpy import pi
import time
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from reward_functions import reward_singleff


action_noise = NormalActionNoise(mean=0., sigma=float(0.2))

arg.goal_radius_range=[0.1,0.2]
arg.std_range = [0.02,0.1,0.02,0.1]
arg.TERMINAL_VEL = 0.02  # terminal velocity? # norm(action) that you believe as a signal to stop 0.1.
arg.DELTA_T=0.2
arg.EPISODE_LEN=35
arg.training=True

env=ffac_1d.FireflyTrue1d_cpu(arg)
# modelname='trained_agent/TD_acc_control_pretrain_1000000_4_3_23_32.zip'
modelname=None
if modelname is None: # new train
    # model = TD3_ff(MlpPolicy,
    #         env, 
    #         verbose=1,
    #         tensorboard_log="./Tensorboard/",
    #         action_noise=action_noise,
    #         buffer_size=int(1e6),
    #         batch_size=512,
    #         learning_rate=3e-4, 
    #         train_freq=100,
    #         # policy_kwargs={'act_fun':tf.nn.selu,'layers':[256,256,64,32]}
    #         policy_kwargs={'layers':[512,512]},
    #         policy_delay=2, 
    #         learning_starts=1000, 
    #         gradient_steps=100, 
    #         gamma=0.98, 
    #         tau=0.005, 
    #         target_policy_noise=0.1, 
    #         target_noise_clip=0.1,
    #         _init_setup_model=True, 
    #         full_tensorboard_log=False, 
    #         seed=None, 
          
    #         )
        model = TD3(MlpPolicy,
            env,
            tensorboard_log="./Tensorboard/",
            policy_kwargs={'optimizer_kwargs':{'weight_decay':0.001}},
            buffer_size=int(1e6),
            batch_size=512,
            device='cuda',
            )
else: # retrain
    model = TD3_ff.load(modelname,
                    env, 
                    verbose=1,
                    tensorboard_log="./Tensorboard/",
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
# env.cost_scale=0.1
for i in range(10):  
    namestr= ("trained_agent/1d_{}_{}_{}_{}_{}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))
    model.learn(total_timesteps=int(train_time/10),tb_log_name=namestr)
    model.save(namestr)



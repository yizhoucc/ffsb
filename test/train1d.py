
# # from stable_baselines.td3.policies import MlpPolicy
# from stable_baselines3.td3.policies import MlpPolicy

# # from stable_baselines3 import TD3


# # from TD3_test import TD3_ff
# from TD3_torch import TD3
# from FireflyEnv import ffac_1d
# from Config import Config
# arg=Config()
# import numpy as np
# from numpy import pi
# import time
# from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# from reward_functions import reward_singleff


# action_noise = NormalActionNoise(mean=0., sigma=float(0.2))

# arg.goal_radius_range=[0.1,0.2]
# arg.std_range = [0.02,0.1,0.02,0.1]
# arg.TERMINAL_VEL = 0.02  # terminal velocity? # norm(action) that you believe as a signal to stop 0.1.
# arg.DELTA_T=0.2
# arg.EPISODE_LEN=35
# arg.training=True

# env=ffac_1d.FireflyTrue1d_cpu(arg)
# # modelname='trained_agent/TD_acc_control_pretrain_1000000_4_3_23_32.zip'
# modelname=None
# if modelname is None: # new train
#     # model = TD3_ff(MlpPolicy,
#     #         env, 
#     #         verbose=1,
#     #         tensorboard_log="./Tensorboard/",
#     #         action_noise=action_noise,
#     #         buffer_size=int(1e6),
#     #         batch_size=512,
#     #         learning_rate=3e-4, 
#     #         train_freq=100,
#     #         # policy_kwargs={'act_fun':tf.nn.selu,'layers':[256,256,64,32]}
#     #         policy_kwargs={'layers':[512,512]},
#     #         policy_delay=2, 
#     #         learning_starts=1000, 
#     #         gradient_steps=100, 
#     #         gamma=0.98, 
#     #         tau=0.005, 
#     #         target_policy_noise=0.1, 
#     #         target_noise_clip=0.1,
#     #         _init_setup_model=True, 
#     #         full_tensorboard_log=False, 
#     #         seed=None, 
          
#     #         )
#         model = TD3(MlpPolicy,
#             env,
#             tensorboard_log="./Tensorboard/",
#             policy_kwargs={'optimizer_kwargs':{'weight_decay':0.001}},
#             buffer_size=int(1e6),
#             batch_size=512,
#             device='cuda',
#             )
# else: # retrain
#     model = TD3_ff.load(modelname,
#                     env, 
#                     verbose=1,
#                     tensorboard_log="./Tensorboard/",
#                     action_noise=action_noise,
#                     buffer_size=int(1e6),
#                     batch_size=512,
#                     learning_rate=3e-4, 
#                     train_freq=100,
#                     # policy_kwargs={'act_fun':tf.nn.selu,'layers':[256,256,64,32]}
#                     # policy_kwargs={'layers':[128,128]},
#                     policy_delay=2, 
#                     learning_starts=1000, 
#                     gradient_steps=100, 
#                     random_exploration=0., 
#                     gamma=0.98, 

#                     tau=0.005, 
#                     target_policy_noise=0.1, 
#                     target_noise_clip=0.1,
#                     _init_setup_model=True, 
#                     full_tensorboard_log=False, 
#                     seed=None, 
#                     n_cpu_tf_sess=None,            
#                     )

# train_time=1000000 
# # env.cost_scale=0.1
# for i in range(10):  
#     namestr= ("trained_agent/1d_{}_{}_{}_{}_{}".format(train_time,i,
#     str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#     ))
#     model.learn(total_timesteps=int(train_time/10),tb_log_name=namestr)
#     model.save(namestr)

#--------------------SAC
from Config import Config
arg=Config()
from numpy import pi
import time
from FireflyEnv import ffacc_real
arg.init_action_noise=0.5
arg.goal_distance_range=[0.2,1]
arg.mag_action_cost_range= [0.1,0.2]
arg.dev_action_cost_range= [0.9,1.]
arg.dev_v_cost_range= [0.1,1.]
arg.dev_w_cost_range= [0.1,1.]
arg.gains_range =[0.35,0.45,pi/2-0.1,pi/2+0.1]
arg.std_range = [0.01,0.07,0.01,0.07]
arg.goal_radius_range=[0.07,0.15]
arg.tau_range=[0.1,0.13]
arg.REWARD=100
arg.TERMINAL_VEL = 0.1
arg.DELTA_T=0.1
arg.EPISODE_LEN=40
arg.training=True
arg.presist_phi=False
arg.agent_knows_phi=True
arg.cost_scale=1
arg.trial_base=True
modelname=None
# modelname="simple1d_100000_0_18_18_13"
note='re' 
from stable_baselines3 import SAC
env=ffacc_real.Simple1d(arg)
if modelname is None:
    model = SAC("MlpPolicy", 
            env,
            # tensorboard_log="./Tensorboard/",
            buffer_size=int(1e6),
            batch_size=1024,
            device='cpu',
            verbose=False,
            train_freq=6,
            target_update_interval=9,
            learning_rate=3e-4,
            gamma=0.99,
    )
    train_time=10000
    env.no_skip=False
    env.cost_scale=0.
    for i in range(9): 
        env.cost_scale=0.01*i
        model.learn(total_timesteps=int(train_time))
        namestr= ("trained_agent/1dmaindev{}_{}_{}_{}_{}".format(train_time,i,
        str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
        ))
        model.save(namestr)
    raise RuntimeError
    env.no_skip=False
    for i in range(20):  
        env.cost_scale=0.05*(i+1)
        namestr= ("trained_agent/simple1d_{}_{}_{}_{}_{}".format(train_time,i,
        str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
        ))
        model.learn(total_timesteps=int(train_time))
        model.save(namestr)
    for i in range(2): 
        namestr= ("trained_agent/simple1d_{}_{}_{}_{}_{}".format(train_time,i,
        str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
        ))
        model.learn(total_timesteps=int(train_time),tb_log_name=namestr,log_interval=100)
        model.save(namestr)
else:
    train_time=100000
    for i in range(20):  
        for i in range(9):  
            env.cost_scale=0.05*(i+1)+0.1
            model = SAC.load('./trained_agent/'+modelname, 
                env,
                # tensorboard_log="./Tensorboard/",
                buffer_size=int(1e6),
                batch_size=512,
                device='cpu',
                verbose=False,
                learning_rate=5e-4,
                gamma=0.99,
                train_freq=6,           
                target_update_interval=9,

        )
            model.learn(total_timesteps=int(train_time))
            namestr= ("trained_agent/simple1d_{}_{}_{}_{}_{}".format(train_time,i,
            str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
            ))
            model.save(namestr)


# # A2C vectorized
# from stable_baselines3.common.vec_env import SubprocVecEnv
# from stable_baselines3 import A2C
# if __name__ == '__main__':
#     def make_env(Env, arg, seed=0):
#         def _init():
#             env = Env(arg,seed=seed)
#             return env
#         return _init
#     env = SubprocVecEnv([make_env(ffacc_real.FireflyTrue1d,arg, i) for i in range(4)])
#     model = A2C("MlpPolicy", 
#             env,
#             # buffer_size=int(1e6),
#             # batch_size=1024,
#             device='cpu',
#             # verbose=True,
#             n_steps=4,
#             # target_update_interval=8,
#             learning_rate=3e-4,
#             gamma=0.99,
#             policy_kwargs={'net_arch':[256,256]} ,
#     )
#     train_time=100000
#     for i in range(9):  
#         env.cost_scale=0.1*(i+1)
#         namestr= ("trained_agent/simple1d_{}_{}_{}_{}_{}".format(train_time,i,
#         str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#         ))
#         model.learn(total_timesteps=int(train_time),tb_log_name=namestr,log_interval=10)
#         model.save(namestr)
#     env.no_skip=False
#     for i in range(2): 
#         namestr= ("trained_agent/simple1d_{}_{}_{}_{}_{}".format(train_time,i,
#         str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#         ))
#         model.learn(total_timesteps=int(train_time),tb_log_name=namestr,log_interval=100)
#         model.save(namestr)

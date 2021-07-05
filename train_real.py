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
# arg.goal_distance_range=[0.01,0.99]
arg.gains_range =[0.35,0.45,pi/2-0.1,pi/2+0.1]
# arg.goal_radius_range=[0.07,0.2]
arg.std_range = [0.01,0.07,0.01,0.07]
# arg.mag_action_cost_range= [0.0001,0.0005]
# arg.dev_action_cost_range= [0.0001,0.0005]
arg.REWARD=100
arg.TERMINAL_VEL = 0.05
arg.DELTA_T=0.1
arg.EPISODE_LEN=40
arg.training=True
arg.presist_phi=False
arg.agent_knows_phi=True
arg.cost_scale=1
env=ffacc_real.FireFlyReady(arg)
env.no_skip=True
modelname=None
# modelname='iticosttimes_100000_2_2_21_23'
note='re' 
from stable_baselines3 import SAC

# # 1d test
# arg.initial_uncertainty_range=[0,1]
# env=ffacc_real.Simple1d(arg)
# env.no_skip=False
# if modelname is None:
#     model = SAC("MlpPolicy", 
#             env,
#             tensorboard_log="./Tensorboard/",
#             buffer_size=int(1e6),
#             batch_size=512,
#             device='cpu',
#             verbose=True,
#             learning_rate=1e-3,
#             gamma=0.999
#     )
#     train_time=300000
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
# else:
#     train_time=300000
#     for i in range(20):  
#         for i in range(9):  
#             env.cost_scale=0.1*i
#             model = SAC.load('./trained_agent/'+modelname, 
#                 env,
#                 tensorboard_log="./Tensorboard/",
#                 buffer_size=int(1e6),
#                 batch_size=512,
#                 device='cpu',
#                 verbose=False,
#                 learning_rate=5e-4,
#                 gamma=0.999
#         )
#             namestr= ("trained_agent/simple1d_{}_{}_{}_{}_{}".format(train_time,i,
#             str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#             ))
#             model.learn(total_timesteps=int(train_time),tb_log_name=namestr,log_interval=10)
#             model.save(namestr)



# 2d training
if modelname is None:
    model = SAC("MlpPolicy", 
            env,
            tensorboard_log="./Tensorboard/",
            buffer_size=int(1e6),
            learning_starts=1000,
            batch_size=1024,
            device='cpu',
            verbose=False,
            learning_rate=3e-4,
            train_freq=6,
            target_update_interval=8,
            gamma=0.95,
            policy_kwargs={'net_arch':[128,256,256]}
    )
    train_time=200000
    for i in range(2):  
        env.no_skip=True
        # env.session_len=100*(i+1)
        env.cost_scale=0
        namestr= ("trained_agent/iti_{}_{}_{}_{}_{}".format(train_time,i,
        str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
        ))
        model.learn(total_timesteps=int(train_time),tb_log_name=namestr,log_interval=100)
        model.save(namestr)
    for i in range(20):  
        # model.gamma=(0.95+i)/(i+1)
        # env.session_len=min(100*(i+3),3000)
        env.no_skip=False
        env.cost_scale=i**2/400
        namestr= ("trained_agent/iticosttimes_{}_{}_{}_{}_{}".format(train_time,i,
        str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
        ))
        model.learn(total_timesteps=int(train_time))
        model.save(namestr)
    # for i in range(10):  
    #     env.no_skip=False
    #     env.cost_scale=1
    #     env.reset()
    #     namestr= ("trained_agent/skipcost_{}_{}_{}_{}_{}".format(train_time,i,
    #     str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    #     ))
    #     model.learn(total_timesteps=int(train_time),tb_log_name=namestr,log_interval=100)
    #     model.save(namestr)
    # for i in range(20):  
    #         namestr= ("trained_agent/{}_{}_{}".format(note,modelname,i))
    #         model.learn(total_timesteps=int(train_time),tb_log_name=namestr)
    #         model.save(namestr)

else:
    train_time=100000
    model = SAC.load('./trained_agent/'+modelname, 
            env,
            tensorboard_log="./Tensorboard/",
            buffer_size=int(1e6),
            batch_size=1024,
            device='cpu',
            verbose=False,
            learning_rate=3e-4,
            train_freq=4,
            target_update_interval=4,
            gamma=0.99,
        )
    for i in range(22):  
        if i <20:
        #     env.no_skip=False
        #     # env.session_len=300
        #     env.cost_scale=2/400
        #     env.reset()
        #     model.learning_rate=7e-4
        #     model.learn(total_timesteps=int(train_time))
        #     namestr= ("trained_agent/{}_{}_{}".format(note,modelname,i))
        #     model.save(namestr)
        # elif i >=2:  
            env.no_skip=False
            env.cost_scale=1/20*i
            model.learning_rate=3e-4
            env.reset()
            model.learn(total_timesteps=int(train_time))
            namestr= ("trained_agent/{}_{}_{}".format(note,modelname,i))
            model.save(namestr)


raise RuntimeError
agent=SAC.load('trained_agent/re_sacappbelief_500000_4_20_5_38_2.zip')
agent=agent.actor.cpu()

if modelname is None:
    model = TD3(MlpPolicy,
        env,
        tensorboard_log="./Tensorboard/",
        buffer_size=int(1e6),
        batch_size=512,
        device='cpu',
        verbose=True,
        # action_noise=action_noise,
        learning_rate=1e-3,
        )
    train_time=500000
    for i in range(10):  
        namestr= ("trained_agent/{}_{}_{}_{}_{}".format(train_time,i,
        str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
        ))
        # namestr="trained_agent/manual"
        model.learn(total_timesteps=int(train_time/10),tb_log_name=namestr)
        model.save(namestr)
        
        # model = TD3.load('./trained_agent/manual',
        # env,
        # tensorboard_log="./Tensorboard/",
        # buffer_size=int(1e6),
        # batch_size=512,
        # device='cpu',
        # verbose=True,
        # action_noise=action_noise,
        # learning_rate=1e-4*5
        # )

        # arg.dev_v_cost_range[1]+=0.1
        # arg.dev_w_cost_range[1]+=0.1
        # env=ffacc_real.FireflyFinal(arg)
        # arg.init_action_noise=arg.init_action_noise/2+0.05
        # action_noise=NormalActionNoise(mean=0., sigma=float(arg.init_action_noise))
else:
    # arg.goal_distance_range=[0.2,1]
    # arg.mag_action_cost_range= [0.0001,0.1]
    # arg.dev_v_cost_range= [0.1,2]
    # arg.dev_w_cost_range= [0.1,2]
    # arg.std_range = [0.01,0.5,0.01,0.5]  

    # # arg.goal_distance_range=[0.1,1]
    # # arg.gains_range =[0.05,1,pi/4,pi]
    # # arg.goal_radius_range=[0.07,0.2]
    # # arg.std_range = [0.001,0.6,pi/800,0.6]
    # # arg.mag_action_cost_range= [0.0001,0.1]
    # # arg.dev_action_cost_range= [0.0001,0.2]
    # arg.goal_distance_range=[0.01,0.99]
    # arg.gains_range =[0.3,0.5,pi/2-0.2,pi/2+0.2]
    # # arg.gains_range =[0.39,0.41,pi/2-0.1,pi/2+0.1]
    # arg.goal_radius_range=[0.1,0.15]
    # arg.mag_action_cost_range= [0.00001,0.05]
    # arg.dev_action_cost_range= [0.0001,0.1]
    model = TD3.load('./trained_agent/'+modelname,
        env,
        tensorboard_log="./Tensorboard/",
        buffer_size=int(1e6),
        batch_size=512,
        device='cpu',
        verbose=True,
        action_noise=action_noise,
        learning_rate=1e-4*5
        )
    train_time=5000000 
    for i in range(100):  
        namestr= ("trained_agent/{}_{}_{}".format(note,modelname,i))
        model.learn(total_timesteps=int(train_time/100),tb_log_name=namestr)
        model.save(namestr)



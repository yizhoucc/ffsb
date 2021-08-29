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
# modelname='initcost'
note='re' 
from stable_baselines3 import SAC,PPO


if modelname is None:
   #td3
    model = TD3(MlpPolicy,
        env,
        buffer_size=int(1e6),
        batch_size=512,
        learning_rate=1e-3,
        learning_starts= 2000,
        tau= 0.005,
        gamma= 0.99,
        # train_freq = 10,
        gradient_steps = -1,
        n_episodes_rollout = 1,
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
    train_time=150000
    for i in range(1,11):  
        env.cost_scale=(1/20)**3
        if i==1:
            for j in range(1,5): 
                env.noise_scale=0.1*j
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
        env.noise_scale=0.5
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

# # ppo
#     model = PPO('MlpPolicy',
#         env,
#         learning_rate=3e-4,
#         n_steps=4,
#         batch_size=512,
#         gamma=0.99,
#         policy_kwargs = {'net_arch':[128,256,256]},
#     )
#     train_time=60000
#     for i in range(10):  
#         env.cost_scale=1/20*i
#         namestr= ("trained_agent/ppo_{}_{}_{}_{}_{}".format(train_time,i,
#         str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#         ))
#         model.learn(train_time)
#         model.save(namestr)

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



# 2d training sac
# if modelname is None:
#     model = SAC("MlpPolicy", 
#             env,
#             tensorboard_log="./Tensorboard/",
#             buffer_size=int(1e6),
#             learning_starts=1000,
#             batch_size=1024,
#             device='cpu',
#             verbose=False,
#             learning_rate=3e-4,
#             train_freq=6,
#             target_update_interval=8,
#             gamma=0.95,
#             policy_kwargs={'net_arch':[128,256,256]}
#     )
#     train_time=200000
#     for i in range(2):  
#         env.no_skip=True
#         # env.session_len=100*(i+1)
#         env.cost_scale=0
#         namestr= ("trained_agent/pre_{}_{}_{}_{}_{}".format(train_time,i,
#         str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#         ))
#         model.learn(total_timesteps=int(train_time),tb_log_name=namestr,log_interval=100)
#         model.save(namestr)
#     for i in range(20):  
#         # model.gamma=(0.95+i)/(i+1)
#         # env.session_len=min(100*(i+3),3000)
#         env.no_skip=False
#         env.cost_scale=i**2/400
#         namestr= ("trained_agent/dev_{}_{}_{}_{}_{}".format(train_time,i,
#         str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#         ))
#         model.learn(total_timesteps=int(train_time))
#         model.save(namestr)
#     for i in range(10):  
#     #     env.no_skip=False
#     #     env.cost_scale=1
#     #     env.reset()
#         namestr= ("trained_agent/final_{}_{}_{}_{}_{}".format(train_time,i,
#         str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#         ))
#         model.learn(total_timesteps=int(train_time),tb_log_name=namestr,log_interval=100)
#         model.save(namestr)
#     # for i in range(20):  
#     #         namestr= ("trained_agent/{}_{}_{}".format(note,modelname,i))
#     #         model.learn(total_timesteps=int(train_time),tb_log_name=namestr)
#     #         model.save(namestr)

# else:
#     train_time=100000
#     model = SAC.load('./trained_agent/'+modelname, 
#             env,
#             tensorboard_log="./Tensorboard/",
#             buffer_size=int(1e6),
#             batch_size=1024,
#             device='cpu',
#             verbose=False,
#             learning_rate=3e-4,
#             train_freq=4,
#             target_update_interval=4,
#             gamma=0.99,
#         )
#     for i in range(22):  
#         if i <20:
#         #     env.no_skip=False
#         #     # env.session_len=300
#         #     env.cost_scale=2/400
#         #     env.reset()
#         #     model.learning_rate=7e-4
#         #     model.learn(total_timesteps=int(train_time))
#         #     namestr= ("trained_agent/{}_{}_{}".format(note,modelname,i))
#         #     model.save(namestr)
#         # elif i >=2:  
#             env.no_skip=False
#             env.cost_scale=1/20*i
#             model.learning_rate=3e-4
#             env.reset()
#             model.learn(total_timesteps=int(train_time))
#             namestr= ("trained_agent/{}_{}_{}".format(note,modelname,i))
#             model.save(namestr)
# raise RuntimeError



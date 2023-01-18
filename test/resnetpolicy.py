
import torch
from torch import nn
from torch.nn import functional as F
import copy
from collections import OrderedDict

class Residual(nn.Module):  #@save
    """
    The Residual block of ResNet policy.
    for each fully connected linear layer, 
    we concat the theta before the next layer, 
    as the residue conenction
    """
    def __init__(self, numin, numout, thetasize=11):
        super().__init__()
        self.linear=nn.Linear(numin, numout)
        self.thetasize=thetasize

    def forward(self, X):
        # Y = F.relu((self.linear(X))) # the fully connecting part
        # Y += X # 
        Y=self.linear(X)
        Y=torch.cat((Y, X[:,-self.thetasize:]), dim=1)
        return F.relu(Y)


def makemodel_resnet(nnodes, inputdim=2, outputdim=2, thetasize=11):
    arch=[]
    prev=inputdim
    for ind,each in enumerate(nnodes):
        arch=arch+[('res'+str(ind), Residual(prev,each,thetasize=thetasize))]
        prev=each+thetasize
        arch=arch+[('relu'+str(ind), nn.ReLU())]
    arch.append(('linearout', nn.Linear(prev,outputdim)))
    arch.append(('tanh',nn.Tanh()))
    model=nn.Sequential(OrderedDict(arch))
    return model

r=makemodel_resnet( [32,32+11],inputdim=21, outputdim=2 )


def makemodel(nnodes, inputdim=2, outputdim=2):
    arch=[]
    prev=inputdim
    for ind,each in enumerate(nnodes):
        arch=arch+[('linear'+str(ind), nn.Linear(prev,each))]
        prev=each
        arch=arch+[('tanh'+str(ind), nn.Tanh())]
    arch.append(('linearout', nn.Linear(prev,outputdim)))
    arch.append(('tanh',nn.Tanh()))
    model=nn.Sequential(OrderedDict(arch))
    return model

# policy=makemodel([11,53], inputdim=21, outputdim=2)



from stable_baselines3.td3.policies import MlpPolicy
# from TD3_torch import TD3
from Config import Config
arg=Config()
import numpy as np
from numpy import pi
import time
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from reward_functions import reward_singleff
from FireflyEnv import ffacc_real
import torch

arg.init_action_noise=0.5
arg.goal_distance_range=[0.1,1] 
arg.mag_action_cost_range= [0.1,1.]
arg.dev_action_cost_range= [0.1,1.]
arg.dev_v_cost_range= [0.1,1.]
arg.dev_w_cost_range= [0.1,1.]
arg.gains_range =[0.1,1.,pi/2-0.6,pi/2+0.6]
arg.goal_radius_range=[0.129,0.131]
arg.std_range = [0.01,1,0.01,1]
arg.reward_amount=100
arg.terminal_vel = 0.05
arg.dt=0.1
arg.episode_len=40
arg.training=True
arg.presist_phi=False
arg.agent_knows_phi=True
arg.cost_scale=1
env=ffacc_real.FireFlyPaper(arg)
env.no_skip=True
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.4 * np.ones(n_actions))        
modelname=None
# modelname='re1re1repaper_3_199_199'
note='re1' 
from stable_baselines3 import TD3

if modelname is None:
    model = TD3(
        MlpPolicy,
        env,
        buffer_size=int(1e6),
        batch_size=512,
        learning_rate=7e-4,
        # learning_starts= 1000,
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
        # policy_kwargs = {'net_arch':[64,64],'activation_fn':torch.nn.ReLU},
        verbose = 0,
        seed = 42,
        # device = "cuda",
        )
    model.actor.mu=copy.deepcopy(r)
    # model.critic
    # model.critic_target
    model.actor_target.mu=copy.deepcopy(r)
    train_time=100000
    for i in range(1,100):  
        env.noise_scale=1
        env.cost_scale=min(0.03*i,0.5)
        if i==1:
            for j in range(1,11): 
                env.noise_scale=0.1*j
                env.cost_scale=0.00
                namestr= ("trained_agent/{}_{}_{}_{}_{}_{}".format(note,train_time,j,
                str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
                ))
                model.learn(train_time)
                # model.save(namestr)
        namestr= ("trained_agent/{}_{}_{}_{}_{}_{}".format(note,train_time,i,
        str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
        ))
        model.learn(train_time)
        model.save(namestr)
else:
    for i in range(1,200): 
        n_actions = env.action_space.shape[-1]
        model = TD3.load('./trained_agent/'+modelname,
            env,
            train_freq = 4,
            learning_starts= 0,
            action_noise= action_noise,
            policy_delay = 6,
            learning_rate=1e-4,
            seed = 1,
            )
        env.noise_scale=1
        train_time=10000
        env.cost_scale=min(0.002*i+0.3,0.5)
        # env.cost_scale=0.3
        # env.cost_scale=0.1
        env.reward_ratio=1
        namestr= ("trained_agent/{}{}_{}".format(note,modelname,i))
        model.learn(train_time)
        model.save(namestr)


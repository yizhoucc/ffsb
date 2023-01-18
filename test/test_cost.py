import numpy as np
from numpy import pi
import time
import torch
import gym
from stable_baselines.ddpg.policies import LnMlpPolicy, MlpPolicy

import matplotlib.pyplot as plt
import scipy.stats as stats
import math

from stable_baselines import DDPG, TD3
from FireflyEnv import ffenv_new_cord
from reward_functions import reward_singleff
from Config import Config
arg=Config()
arg.goal_radius_range=[0.15,0.3]
env=ffenv_new_cord.FireflyAgentCenter(arg)

model=TD3.load('trained_agent\TD_95gamma_mc_500000_0_23_22_8.zip')

model.set_env(env)

# collection some actionsf

actions=[]
env.reset()
done=False
while not done:
    action,_=model.predict(env.decision_info)
    decision_info,_,done,_=env.step(action)
    actions.append(action)

print('mag cost: ',action_cost_magnitude(actions))
print('dev cost: ',action_cost_dev(actions))
print('total cost: ',cost_wrapper(actions, action_cost_param=[0.05,0.5]))

fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot()
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])

ax.set_prop_cycle('color',plt.cm.Spectral(np.linspace(0,1,len(actions))))


# for action in actions:

actions=np.array(actions)
for i in range(len(actions)-1):
    ax.plot(actions[i:i+2,1],actions[i:i+2,0],linewidth=6,alpha=0.5,marker='p',ms=12)


plt.show()






def action_cost_magnitude(actions):
    # sum up the action magnitude
    cost=0.
    for action in actions:
        # print(action,np.linalg.norm(action))
        cost+=np.sum(np.linalg.norm(action)**2)
    return cost


def action_cost_dev(actions):
    cost=0.
    prev_action=np.zeros((2))
    for action in actions:
        # print(action,prev_action,action-prev_action,np.linalg.norm(action-prev_action))
        cost+=(np.linalg.norm(action-prev_action))**2
        prev_action=action
    return cost


def cost_wrapper(actions, action_cost_param=[0.05,0.5]):
    cost=action_cost_param[0]*action_cost_magnitude(actions)+action_cost_param[1]*action_cost_dev(actions)    
    return cost


# testing the obj inheritence
from gym.spaces import Box
import numpy as np
class A(object):
    def __init__(self):
        # super().__init__()
        self.var1=1
        low = -np.inf
        high = np.inf
        self.space=Box(low=low, high=high,shape=(1,self.var1))

class B(A):
    def __init__(self):
        super().__init__()
        self.var2=2

class C(B):
    def __init__(self):
        super().__init__()
        self.var2=0
        self.var1=20

objA=A()
objB=B()
objC=C()
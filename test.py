import os



import warnings
warnings.filterwarnings('ignore')

from inverse_model import Dynamic, Inverse
# agent
from stable_baselines import DDPG
# env
from FireflyEnv import ffenv
from Config import Config
from Inverse_Config import Inverse_Config
import torch
phi=torch.ones(9)


inverse_arg=Inverse_Config()
env_arg=Config()
policy=DDPG.load("DDPG_ff")

print(os.getcwd())





# testing loading
teacher_env=ffenv.FireflyEnv(env_arg)
agent_env=ffenv.FireflyEnv(env_arg)

teacher_env.assign_presist_phi(phi) 
print(teacher_env.theta)
agent_env.assign_presist_phi((phi-0.5)) 
print(agent_env.theta)
print('state',agent_env.state)




# testing dynamics
dynamic=Dynamic(policy, teacher_env,agent_env)

# a,b,c,_,_=dynamic.run_episode()
# print("a,b,c",a,b,c)
a,b,c,d,e=dynamic.collect_data(5)



# testing inverse model
model=Inverse(dynamic=dynamic,arg=inverse_arg)
model.learn(100)
print("end")
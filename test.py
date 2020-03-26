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
# policy=DDPG.load("DDPG_ff")

print(os.getcwd())





# testing env
teacher_env=ffenv.FireflyEnv(env_arg)
agent_env=ffenv.FireflyEnv(env_arg)

teacher_env.assign_presist_phi(phi) 
print(teacher_env.theta)
agent_env.assign_presist_phi((phi-0.5)) 
print(agent_env.theta)
print('state',agent_env.state)


# testing torch agent
import policy_torch
baselines_mlp_model = DDPG.load("DDPG_ff_mlp")
policy = policy_torch.copy_mlp_weights(baselines_mlp_model)


# testing dynamics
dynamic=Dynamic(policy, teacher_env,agent_env)

# a,b,c,_,_=dynamic.run_episode()
# print("a,b,c",a,b,c)
# a,b,c,d,e=dynamic.collect_data(3)



# testing inverse model
model=Inverse(dynamic=dynamic,arg=inverse_arg)
while True:

    model.learn(500)
    print((agent_env.theta))
print("end")
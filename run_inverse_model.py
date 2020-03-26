from inverse_model import Dynamic, Inverse
# agent
from stable_baselines import DDPG
# env
from FireflyEnv import ffenv
# arg
from Inverse_Config import Config 
arg=Config()


# load policy
policy=DDPG.load("DDPG_ff")
# load envs
teacher_env=ffenv.FireflyEnv(arg)
agent_env=ffenv.FireflyEnv(arg)
# assign phi to teacher, and init theta to agent
teacher_env.assign_presist_phi(phi) 
agent_env.assign_presist_phi(init_theta(phi)) 
# init dynamic
dynamic=Dynamic(policy, teacher_env,agent_env)
# init model
model=Inverse(dynamic)


# training
model.learn(num_episode)


# save
model.save("filename")
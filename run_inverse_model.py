from inverse_model import Dynamic, Inverse
import InverseFuncs
# agent
from stable_baselines import DDPG
# env
from FireflyEnv import ffenv
# arg
from Config import Config
from Inverse_Config import Inverse_Config 

# easy change arg here
arg=Inverse_Config()
arg.gains_range = [0.8, 1.2, pi/5, 3*pi/10]
arg.std_range = [1e-2, 0.3, 1e-2, 0.2]
arg.WORLD_SIZE = 1.0
arg.goal_radius_range = [0.2* arg.WORLD_SIZE, 0.5* arg.WORLD_SIZE]
arg.DELTA_T = 0.1
arg.EPISODE_TIME = 1  # # maximum length of time for one episode. if monkey can't firefly within this time period, new firefly comes
arg.EPISODE_LEN = int(arg.EPISODE_TIME / arg.DELTA_T)
arg.NUM_SAMPLES=2
arg.NUM_EP = 50
arg.NUM_IT = 100 # number of iteration for gradient descent
arg.NUM_thetas = 5


# load policy
sbpolicy=DDPG.load("DDPG_theta") # 100k step trained, with std noise.
# convert to torch policy
policy=policy_torch.copy_mlp_weights(sbpolicy)

# load envs
envarg=Config()
teacher_env=ffenv.FireflyEnv(arg)
agent_env=ffenv.FireflyEnv(arg)

# prepare the env
# assign phi to teacher, and init theta to agent
phi=InverseFuncs.reset_theta(arg)
teacher_env.assign_presist_phi(phi) 
agent_env.assign_presist_phi(InverseFuncs.init_theta(phi,arg,purt=None)) 

# pack the policy and env into dynamic
data=Dynamic(policy, teacher_env,agent_env,datasource=simulation)


# the inverse model is constructed given dynamic and algorithm
model=Inverse(data)


# recover the theta by running optimzation
model.learn(num_episode)


# save the parameter and log
model.save("filename")
from stable_baselines import A2C
from stable_baselines.common.policies import ActorCriticPolicy,MlpPolicy, MlpLnLstmPolicy,RecurrentActorCriticPolicy
from FireflyEnv import ffenv_new_cord
from Config import Config
arg=Config()
arg.goal_radius_range=[0.3,0.5]
import numpy as np


env_new_cord=ffenv_new_cord.FireflyAgentCenter(arg)
env_new_cord.max_distance=0.5
model=A2C(MlpLnLstmPolicy,env_new_cord,n_steps=)


model.learn(100000)
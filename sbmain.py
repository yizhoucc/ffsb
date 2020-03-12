import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import A2C
from FireflyEnv import ffenv

env=gym.make('FF-v0')
model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)

obs = env.reset()

env.close()
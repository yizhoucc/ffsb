import gym
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import DDPG
from FireflyEnv import ffenv
from Config import Config
arg=Config()


# # check env
# from stable_baselines.common.env_checker import check_env
# # env=gym.make('FF-v0')
# env=ffenv.FireflyEnv(arg)
# # env = CustomEnv(arg1, ...)
# # It will check your custom environment and output additional warnings if needed
# check_env(env)

# extra define may need or may not (used to need but now have sb)
# env.model.max_goal_radius = goal_radius_range[1] # use the largest world size for goal radius
# state_dim = env.state_dim
# action_dim = env.action_dim
# MAX_EPISODE = 1000
# std = 0.01
# noise = Noise(action_dim, mean=0., std=std)
# tot_t = 0.
# episode = 0.



env=ffenv.FireflyEnv(arg)
model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=5000)

model.save("a2c_ff")
obs = env.reset()
del model # remove to demonstrate saving and loading
model = A2C.load("a2c_ff")

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# env.close()
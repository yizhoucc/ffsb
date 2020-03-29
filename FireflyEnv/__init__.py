
from gym.envs.registration import register
register(
    id ='FF-v0',
    entry_point='FireflyEnv.ffenv:FireflyEnv'
)
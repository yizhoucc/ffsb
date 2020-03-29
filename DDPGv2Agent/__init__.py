
from gym.envs.registration import register

register(
    id ='FireflyTorch-v0',
    entry_point ='FireflyEnv.firefly_gym:FireflyEnv',

)

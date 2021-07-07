

import time
from FireflyEnv import multiffenv
from stable_baselines3 import SAC
env=multiffenv.MultiFF()
model = SAC("MlpPolicy", 
            env,
            buffer_size=int(1e6),
            batch_size=512,
            device='cpu',
            verbose=False,
            train_freq=6,
            target_update_interval=9,
            learning_rate=3e-4,
            gamma=0.95,
    )
train_time=1000
for i in range(9): 
    model.learn(total_timesteps=int(train_time))
    namestr= ("trained_agent/testmulti{}_{}_{}_{}_{}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))
    model.save(namestr)

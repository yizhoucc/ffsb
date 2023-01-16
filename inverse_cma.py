import sys
sys.path.append('..')
import numpy as np
from cmaes import CMA
import copy
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import torch
from stable_baselines3 import TD3
torch.manual_seed(42)
from numpy import pi
from InverseFuncs import *
from FireflyEnv import ffacc_real
from Config import Config
import ray
from pathlib import Path
arg = Config()


print('loading data')
datapath=Path("example_data")
savename=datapath.parent/('preall'+datapath.name)
with open(datapath,'rb') as f:
    df = pickle.load(f)

print('process data')
states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
print('done process data')

# decide if to continue
optimizer=None
log=[]
if savename.is_file():
    print('continue on previous inverse...')
    with open(savename, 'rb') as f:
        log = pickle.load(f)
    optimizer=log[-1][0]

env=ffacc_real.FireFlyPaper(arg)
phi=torch.tensor([[0.5],
            [pi/2],
            [0.001],
            [0.001],
            [0.001],
            [0.001],
            [0.13],
            [0.001],
            [0.001],
            [0.001],
            [0.001],
    ])
agent_=TD3.load('agent')
agent=agent_.actor.mu.cpu()



# use cluster
# ray.init(address='192.168.0.177:6379', _redis_password='5241590000000000', 
# log_to_driver=False,ignore_reinit_error=True,_node_ip_address='192.168.0.119')

# ray.init(address="ray://192.168.0.177:6379", _redis_password='5241590000000000', 
# log_to_driver=False,ignore_reinit_error=True,_node_ip_address='192.168.0.119')

# use localhost
ray.init(log_to_driver=False,ignore_reinit_error=True)

@ray.remote
def getlogll(x):
    with torch.no_grad():
        return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=0.01,num_iteration=1, states=states, samples=5,gpu=False).item()

# fresh start
if not optimizer:
    # init condition, we want to at least cover some dynamic range
    init_theta=torch.tensor([[0.5],   
            [1.6],   
            [0.5],   
            [0.5],   
            [0.5],   
            [0.5],   
            [0.13],   
            [0.5],   
            [0.5],   
            [0.5],   
            [0.5]])
    dim=init_theta.shape[0]
    init_cov=torch.diag(torch.ones(dim))*0.3
    cur_mu=init_theta.view(-1)
    cur_cov=init_cov
    optimizer = CMA(mean=np.array(cur_mu), sigma=0.5,population_size=14)
    optimizer.set_bounds(np.array([
    [0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.129, 0.01, 0.01, 0.01, 0.01],
    [1.,2,1,1,1,1,0.131,1,1,1,1]],dtype='float32').transpose())



for generation in range(len(log),len(log)+99):
    solutions = []
    xs=[]
    for _ in range(optimizer.population_size):
        x = optimizer.ask().astype('float32')
        xs.append(x)
    solutions=ray.get([getlogll.remote(p) for p in xs])
    solutions=[[x,s] for x,s in zip(xs,solutions)]
    optimizer.tell(solutions)
    log.append([copy.deepcopy(optimizer), xs, solutions])
    # plt.imshow(optimizer._C)
    # plt.colorbar()
    # plt.show()
    with open(savename, 'wb') as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("generation: ",generation)
    print(["{0:0.2f}".format(i) for i in optimizer._mean])
    print(["{0:0.2f}".format(i) for i in np.diag(optimizer._C)**0.5])
    if optimizer.should_stop():
        print('stop at {}th generation'.format(str(generation)))



import os
os.chdir(r"C:\Users\24455\iCloudDrive\misc\ffsb")
import numpy as np
from cmaes import CMA
import copy
import matplotlib.pyplot as plt
from numpy.core.defchararray import array
from FireflyEnv.env_utils import is_pos_def
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import torch
import heapq
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt
import time
from stable_baselines3 import TD3
torch.manual_seed(42)
from numpy import pi
from InverseFuncs import *
from monkey_functions import *
from FireflyEnv import ffacc_real
from Config import Config
# from cma_mpi_helper import run
import ray
from pathlib import Path


# use cluster
# ray.init(address='192.168.0.177:6379', _redis_password='5241590000000000', 
# log_to_driver=False,ignore_reinit_error=True,_node_ip_address='192.168.0.119')

# ray.init(address="ray://192.168.0.177:6379", _redis_password='5241590000000000', 
# log_to_driver=False,ignore_reinit_error=True,_node_ip_address='192.168.0.119')

# use localhost
ray.init(log_to_driver=False,ignore_reinit_error=True)

arg = Config()
import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager, ExitStack
@contextmanager
def suppress(out=True, err=False):
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield
            
env=ffacc_real.FireFlyPaper(arg)
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()

# new way
datapath=Path("D:\mkdata\\bruno_pert")
sessions=list(datapath.glob('*ds'))
df=None
for session in sessions:
    # savename=str(session.name)+'_inv'
    with open(session,'rb') as f:
        df_ = pickle.load(f)
    if df is None:
        df=df_
    else:
        df=df.append(df_)


print('loading data')
# note='testdcont'
savename='cmafull_s_pert'
# with open("C:/Users/24455/Desktop/victor_normal_downsample",'rb') as f:
#         df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
# df=df[df.target_r>250]
# df=df[df.floor_density==0.0005]
# floor density are in [0.0001, 0.0005, 0.001, 0.005]
# df=df[500:600]
print('process data')
states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
print('done process data')

# misc
phi=torch.tensor([[0.5],
        [pi/2],
        [0.001],
        [0.001],
        [0.001],
        [0.001],
        # [0.13],
        [0.001],
        [0.001],
        [0.001],
        [0.001],
])
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
init_theta=torch.tensor([0.8901243,  1.6706846,  0.858392,   0.37444177, 0.05250579, 0.08552641, 0.12986924, 0.20578894, 0.7251345,  0.4741538,  0.40967906]).view(-1,1)
dim=init_theta.shape[0]
init_cov=torch.diag(torch.ones(dim))*0.3
cur_mu=init_theta.view(-1)
cur_cov=init_cov



@ray.remote
def getlogll(x):
    with torch.no_grad():
        return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=0.01,num_iteration=1, states=states, samples=5,gpu=False).item()



optimizer = CMA(mean=np.array(cur_mu), sigma=0.5,population_size=14)
optimizer.set_bounds(np.array([
[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.129, 0.01, 0.01, 0.01, 0.01],
[1.,2,1,1,1,1,0.131,1,1,1,1]],dtype='float32').transpose())
log=[]
for generation in range(50):
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
    print(generation,optimizer._mean,'\n',np.diag(optimizer._C)**0.5,)



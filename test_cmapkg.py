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
from cma_mpi_helper import run
import ray
ray.init(log_to_driver=False,ignore_reinit_error=True,include_dashboard=True)


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

print('loading data')
note='testdcont'
with open("C:/Users/24455/Desktop/bruno_normal_downsample",'rb') as f:
        df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df=df[df.target_r>250]
df=df[df.floor_density==0.001]
# floor density are in [0.0001, 0.0005, 0.001, 0.005]
df=df[500:600]
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
        [0.13],
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
dim=init_theta.shape[0]
init_cov=torch.diag(torch.ones(dim))*0.3
cur_mu=init_theta.view(-1)
cur_cov=init_cov



@ray.remote
def getlogll(x):
    with torch.no_grad():
        return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=0.01,num_iteration=1, states=states, samples=5,gpu=False).item()



optimizer = CMA(mean=np.array(cur_mu), sigma=0.5,population_size=16*4)
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
    plt.imshow(optimizer._C)
    plt.colorbar()
    plt.show()
    with open('cmapk56fixgoalrnorm3_', 'wb') as handle:
            pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(generation,optimizer._mean,'\n',np.diag(optimizer._C)**0.5,)






# loading
import pickle
with open('cmapk56fixlossnorm1_', 'rb') as f:
    log = pickle.load(f)

# theta bar
optimizer=log[-1][0]
cov=optimizer._C
theta=torch.tensor(optimizer._mean).view(-1,1)
finaltheta=theta
cov=optimizer._C
finalcov=torch.tensor(cov)
theta_names = [ 'pro gain v',
                'pro gain w',
                'pro noise v',
                'pro noise w',
                'obs noise v',
                'obs noise w',
                'goal radius',
                'action cost v',      
                'action cost w',      
                'init uncertainty x',      
                'init uncertainty y',      
                ]
fig=plt.figure(figsize=[3,2])
ax = fig.add_subplot(111)
# Create bars and choose color
ax.bar([i for i in range(11)], finaltheta,yerr=torch.diag(finalcov)**0.5,color = 'tab:blue')
# title and axis names
ax.set_ylabel('inferred parameter value')
ax.set_xticks([i for i in range(11)])
ax.set_xticklabels(theta_names, rotation=45, ha='right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

theta_mean=np.array([
    0.5,
    1.57,
    0.5,
    0.5,
    0.5,
    0.5,
    0.13,
    0.5,
    0.5,
    0.5,
    0.5    
])
# plt.bar([i for i in range(11)], torch.diag(finalcov)**0.5*1/theta_mean)
fig=plt.figure(figsize=[3,2])
ax = fig.add_subplot(111)
# Create bars and choose color
ax.bar([i for i in range(11)], torch.diag(finalcov)**0.5*1/theta_mean, color = 'tab:blue')
# title and axis names
ax.set_ylabel('inferred parameter uncertainty (std/mean)')
ax.set_xticks([i for i in range(11)])
ax.set_xticklabels(theta_names, rotation=45, ha='right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# density obs
densities=[0.0001, 0.0005, 0.001, 0.005]
filenamecommom='cmapk56fixgoalrnorm'
logs=[None]*4
for i, density in enumerate(densities):
    try:
        with open(filenamecommom+str(i+1)+'_', 'rb') as f:
            log = pickle.load(f)
        logs[i]=log
    except:
        continue
means=[log[-1][0]._mean if log else None for log in logs]
stds=[np.diag(log[-1][0]._C)**0.5 if log else None for log in logs]
xs,ys,errs=[],[],[]
for i in range(4):
    if means[i] is not None and stds[i] is not None and densities[i] is not None:
        x,y,err=densities[i],means[i],stds[i]
        xs.append(x);ys.append(y);errs.append(err)
ys=np.array(ys)
plt.plot(xs,ys[:,2:4])
plt.plot(xs,ys[:,4:6])
plt.plot(xs,ys[:,2:4]/ys[:,4:6])
plt.plot(xs,ys[:,6])

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(xs,ys[:,2]/ys[:,4])
ax.set_xlabel('optical flow density',fontsize=12)
ax.set_ylabel('observation reliable degree',fontsize=12)
ax.set_title('forward observation noise vs optical flow density', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# log ll vs gen
res=[l[2] for l in log]
loglls=[]
for r in res:
    genloglls=[]
    for point in r:
        genloglls.append(point[1])
    loglls.append(genloglls)
gen=[[i]*optimizer.population_size for i in range(optimizer.generation)]
gen=torch.tensor(gen).flatten()
loglls=torch.tensor(loglls).flatten()
plt.scatter(gen,loglls,alpha=0.1)
plt.xlabel('generations')
plt.ylabel('- log likelihood')

# theta trend
means=[l[0]._mean for l in log]
covs=[l[0]._C for l in log]
stds=[np.diag(cov)**0.5 for cov in covs]
means=torch.tensor(means)
stds=torch.tensor(stds)
for i in range(11):
    plt.plot(means[:,i],alpha=0.9)
    plt.fill_between([j for j in range(stds.shape[0])],y1=means[:,i]+stds[:,i],y2=means[:,i]-stds[:,i],alpha=0.2)
plt.xlabel('number updates')
plt.ylabel('parameter value')

# pca space of all samples
res=[l[2] for l in log]
alltheta=[]
loglls=[]
for r in res:
    genloglls=[]
    gentheta=[]
    for point in r:
        gentheta.append(point[0])
        genloglls.append(point[1])
    loglls.append(genloglls)
    alltheta.append(torch.tensor(gentheta))
alltheta=[torch.tensor(a) for a in alltheta]
alltheta=torch.tensor(alltheta)
loglls=torch.tensor(loglls).flatten()


np.set_printoptions(precision=2)
print((optimizer._mean))

optimizer=log[-1][0]
cov=optimizer._C
theta=torch.tensor(optimizer._mean).view(-1,1)
finaltheta=theta
cov=optimizer._C
finalcov=torch.tensor(cov)
H=torch.inverse(torch.tensor(cov))
ev, evector=torch.eig(H,eigenvectors=True)
ev=ev[:,0]
ev,esortinds=ev.sort(descending=True)
evector=evector[esortinds].float()

eval = CMA(mean=np.array(optimizer._mean), sigma=0.5,population_size=16*4)
eval.set_bounds(np.array([[0.01]*11,[1.,2,1,1,1,1,0.2,1,1,1,1]],dtype='float32').transpose())
xs=[]
for _ in range(16*20):
    x = eval.ask().astype('float32')
    xs.append(x)
solutions=ray.get([getlogll.remote(p) for p in xs])

projectedparam=torch.pca_lowrank(torch.tensor(xs),2)
projectedparam=projectedparam[0]
# pc space scatter logll
s = plt.scatter(projectedparam[:,0], projectedparam[:,1], c=solutions,alpha=0.3,edgecolors=None, cmap='jet')
plt.clim(min(solutions), max(solutions)) 
plt.xlabel('projected parameters')
plt.ylabel('projected parameters')
c = plt.colorbar()






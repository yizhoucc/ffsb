# validation of the model using simulation data
# vary 2 param and get logll background

import torch
import numpy as np
from plot_ult import *
import ray
from stable_baselines3 import TD3

# load agent and task 
env=ffacc_real.FireFlyPaper(arg)
env.episode_len=50
env.debug=1
env.terminal_vel=0.2
phi=torch.tensor([[1],
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
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()

# load data
with open('Z:/simulation/inv_0119simulation200','rb') as f:
    log=pickle.load(f)
path=Path('Z:/simulation')
datafile=path/'0119simulation200'
with open(datafile, 'rb') as f:
    states, actions, tasks, groundtruth = pickle.load(f)

# inverse results
finaltheta=log[-1][0]._mean
finalcov=log[-1][0]._C
realfinalcov=np.cov(np.array([l[0] for l in log[-1][2]]).T)

# inverse trend
res=[l[2] for l in log]
allsamples=[]
alltheta=[]
loglls=[]
for r in res:
    alltheta.append(np.mean(np.stack([p[0] for p in r]), axis=0))
    loglls.append(np.mean(np.stack([p[1] for p in r]), axis=0))
alltheta=np.array(alltheta)


# use ray. get likelihood surface
ray.init(log_to_driver=False,ignore_reinit_error=True)
@ray.remote
def getlogll(x):
    x=torch.tensor(x)
    if torch.any(x>2) or torch.any(x<=0):
        return None
    with torch.no_grad():
        return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).view(-1,1), env, action_var=0.001,num_iteration=1, states=states, samples=5,gpu=False).item()


xpixel=9
ypiexel=9
xrange=np.linspace(0.4,1.2,xpixel)
yrange=np.linspace(0.3,0.8,ypiexel)
paramls=[]
for i in xrange:
    for j in yrange:
        finaltheta[0]=i
        finaltheta[1]=j
        paramls.append(finaltheta)


Z=ray.get([getlogll.remote(each.reshape(1,-1).astype('float32')) for each in paramls])
background_data=np.array(Z).reshape(ypiexel,xpixel)

with open('Z:/simulation/bg/4', 'wb+') as f:
    pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)
from notification import notify
notify()


# check
from scipy.ndimage.filters import gaussian_filter




paramls,Z=pickle.load(open('Z:/simulation/bg/4', 'rb'))
# plt.imshow(-background_data, cmap='jet')
X, Y=np.meshgrid(xrange, yrange)
plt.contourf(X, Y, -gaussian_filter(background_data, 0.5), cmap='jet')
plt.colorbar()
plt.scatter(groundtruth[0], groundtruth[1], s=111)
plt.plot(alltheta[:,0],alltheta[:,1])
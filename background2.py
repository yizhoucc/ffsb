# validation of the model using simulation data
# plot the background in pca space
import torch
import numpy as np
from plot_ult import *
import ray
from stable_baselines3 import TD3
# load agent and task --------------------------------------------------------
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


with open('Z:/simulation/inv_0119simulation200','rb') as f:
    log=pickle.load(f)
path=Path('Z:/simulation')
datafile=path/'0119simulation200'
with open(datafile, 'rb') as f:
    states, actions, tasks, groundtruth = pickle.load(f)

# pca space cma theta as scatter --------------------
res=[l[2] for l in log]
allsamples=[]
alltheta=[]
loglls=[]
for r in res:
    for point in r:
        alltheta.append(torch.tensor(point[0]))
        loglls.append(torch.tensor(point[1]))
        allsamples.append([point[1],point[0]])
alltheta=torch.stack(alltheta)
logllsall=torch.tensor(loglls).flatten()
allsamples.sort(key=lambda x: x[0])

projectedparam=torch.pca_lowrank(torch.tensor(alltheta),2)
transition=projectedparam[2] # the V
projectedparamall=projectedparam[0] # the U

allthetamu=torch.mean(alltheta, axis=0)
centeralltheta=alltheta-allthetamu
u,s,v=torch.pca_lowrank(torch.tensor(centeralltheta),2)
# u[0]@torch.diag(s)@v.T+allthetamu
# alltheta[0]
# (alltheta[1]-allthetamu)@np.array(v)@np.linalg.inv(np.diag(s))
# u[1]

mu=torch.mean(projectedparamall,axis=0)
aroundsolution=allsamples[:len(allsamples)//2]
aroundsolution.sort(key=lambda x: x[0])
alltheta=np.vstack([x[1] for x in aroundsolution])
loglls=[x[0] for x in aroundsolution]
pccov=transition.T@torch.tensor(np.cov(alltheta.T)).float()@transition



# slice pca --------------------------------------


res=[l[2] for l in log]
allsamples=[]
alltheta=[]
loglls=[]
for r in res:
    for point in r:
        alltheta.append(torch.tensor(point[0]))
        loglls.append(torch.tensor(point[1]))
        allsamples.append([point[1],point[0]])
alltheta=torch.stack(alltheta)
logllsall=torch.tensor(loglls).flatten()
allsamples.sort(key=lambda x: x[0])
allthetameans=np.array([l[0]._mean for l in log])
alltheta=alltheta[50:]
logllsall=logllsall[50:]
mu=np.mean(np.asfarray(alltheta),0).astype('float32')
score, evectors, evals = pca(np.asfarray(alltheta))
x=score[:,0] # pc1
y=score[:,1] # pc2
z=logllsall


finalcov=log[-1][0]._C
realfinalcov=np.cov(np.array([l[0] for l in log[-1][2]]).T)

finaltheta=log[-1][0]._mean
pcfinaltheta=((finaltheta.reshape(-1)-mu)@evectors).astype('float32')
finalcovpc=(evectors.T@finalcov@evectors).astype('float32')
realfinalcovpc=(evectors.T@realfinalcov@evectors).astype('float32')
pccov=evectors.T@(np.cov(alltheta.T))@evectors
allthetameanspc=((allthetameans-mu)@evectors).astype('float32')

xlow,xhigh=pcfinaltheta[0]-0.3,pcfinaltheta[0]+0.3
ylow,yhigh=pcfinaltheta[1]-0.3,pcfinaltheta[1]+0.3




npixel=11
xrange=np.linspace(-0.1, 0.3,npixel)
yrange=np.linspace(-0.25,0.25,npixel)

# use ray
ray.init(log_to_driver=False,ignore_reinit_error=True)
@ray.remote
def getlogll(x):
    x=torch.tensor(x)
    if torch.any(x>2) or torch.any(x<=0):
        return None
    with torch.no_grad():
        return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).view(-1,1), env, action_var=0.001,num_iteration=1, states=states, samples=5,gpu=False).item()

jobs=[]
for i,u in enumerate(xrange):
    for j,v in enumerate(yrange):
        score=np.array([u,v])
        reconstructed_theta=score@evectors.transpose()[:2,:]+mu
        reconstructed_theta=reconstructed_theta.clip(1e-4,999)
        jobs.append(reconstructed_theta)
        
Z=ray.get([getlogll.remote(each.reshape(1,-1).astype('float32')) for each in jobs])
background_data=np.array(Z).reshape(npixel,npixel)

with open('Z:/simulation/bg/{}'.format(datafile.name), 'wb+') as f:
    pickle.dump((reconstructed_theta,Z), f, protocol=pickle.HIGHEST_PROTOCOL)
from notification import notify
notify()

plt.imshow(background_data)
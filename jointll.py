

print(
'''
the most distinguished parameter that distinghuishs the ASD vs control.

define a midpoint
vary by d1,d2, where d1,d2 is delta theta in the interested direction
calcualte the logll for ASD and healthy seperatly, add together.
final result is a heat map
''')

import matplotlib
from playsound import playsound
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import os
import pandas as pd
import numpy as np
from cmaes import CMA
import copy
import matplotlib.pyplot as plt
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
torch.manual_seed(0)
from numpy import linspace, pi
from InverseFuncs import *
from firefly_task import ffacc_real
from env_config import Config
import ray
from pathlib import Path
arg = Config()
import os
from timeit import default_timer as timer
from plot_ult import *
import pickle


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

# define the midpoint between ASD and healthy 
logls=['Z:/human/fixragroup','Z:/human/clusterpaperhgroup']
monkeynames=['ASD', 'Ctrl' ]

mus,covs,errs=[],[],[]
thetas=[]
for inv in logls:
    finaltheta,finalcov, err=process_inv(inv,ind=60,removegr=False)
    mus.append(np.array(finaltheta).reshape(-1))
    covs.append(finalcov)
    errs.append(err)
    thetas.append(finaltheta)
thetas=torch.tensor(mus)
theta_init=thetas[0]
theta_final=thetas[1]

# load theta distribution
alltag=[]
alltheta=[]
loglls=[]
with open(logls[0],'rb') as f:
    log=pickle.load(f)
    res=[l[2] for l in log[19:99]]
    for r in res:
        for point in r:
            alltheta.append(point[0]) # theta
            loglls.append(point[1])
            alltag.append(0)
with open(logls[1],'rb') as f:
    log=pickle.load(f)
    res=[l[2] for l in log[19:99]]
    for r in res:
        for point in r:
            alltheta.append(point[0]) # theta
            loglls.append(point[1])
            alltag.append(1)

alltheta=np.array(alltheta)
alltag=np.array(alltag)

# compute svm
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(alltheta, alltag)
w = clf.coef_[0] # the normal vector
midpoint=(mus[0]+mus[1])/2
lb=np.array([0,0,0,0,0,0,0.129,0,0,0,0])
hb=np.array([1,2,1,1,1,1,0.131,2,2,1,1])

theta_init=np.min(midpoint-lb)/w[np.argmin(hb-midpoint)]*-w*3+midpoint
theta_final=np.min(midpoint-lb)/w[np.argmin(midpoint-lb)]*w*3+midpoint
theta_init,theta_final=torch.tensor(theta_init).view(-1,1).float(),torch.tensor(theta_final).view(-1,1).float()



# function to eval the logll of the delta theta
num_sample=200
# load the data
datapath=Path("res/hgroup")
with open(datapath, 'rb') as f:
    hstates, hactions, htasks = pickle.load(f)
datapath=Path("res/agroup")
with open(datapath, 'rb') as f:
    astates, aactions, atasks = pickle.load(f)
# select the side tasks
hstates, hactions, htasks=select_tar(hstates, hactions, htasks)
astates, aactions, atasks=select_tar(astates, aactions, atasks)
# randomly select some data
hstates, hactions, htasks=hstates[:num_sample], hactions[:num_sample], htasks[:num_sample].astype('float32')
astates, aactions, atasks=astates[:num_sample], aactions[:num_sample], atasks[:num_sample].astype('float32')

ray.init(log_to_driver=False,ignore_reinit_error=True)
@ray.remote
def getlogll(x):
    atheta,htheta=x[0],x[1]
    atheta.clamp_(1e-3,3)
    htheta.clamp_(1e-3,3)
    with torch.no_grad():
        lla=monkeyloss_(agent, aactions, atasks, phi, atheta, env, action_var=0.001,num_iteration=1, states=astates, samples=5,gpu=False,debug=False)
        llh=monkeyloss_(agent, hactions, htasks, phi, htheta, env, action_var=0.001,num_iteration=1, states=hstates, samples=5,gpu=False,debug=False)
        return -lla-llh


def totalbias(g,p,o):
    return g*(p**2/(p**2+o**2))
def totalgivenop(a,b):
    return (a**2)*(b**2)/((a**2)+(b**2))
def pogiventotal(a, c):
    return ((a**2)*c/((a**2)-c))**0.5


# vary noise and gain -------------------------------
gridreso=11
dx,dy=np.zeros((11)),np.zeros((11))
dx[5]=1
dy[3]=1

auncertainty=mus[0][5]**2*mus[0][3]**2/(mus[0][5]**2+mus[0][3]**2)
huncertainty=(mus[1][5]**2*mus[1][3]**2)/(mus[1][5]**2+mus[1][3]**2)
abias=totalbias(mus[0][1],mus[0][3],mus[0][5])
hbias=totalbias(mus[1][1],mus[1][3],mus[1][5])

c=totalgivenop(mus[0][5],mus[0][3])
pogiventotal(mus[0][5],auncertainty)

X,Y=np.linspace(-0.1,0.7,gridreso),np.linspace(-0.2,0.2,gridreso) # on and pn
paramls=[]
z=np.zeros((gridreso,gridreso))
for i in range(gridreso):
    for j in range(gridreso):
        # vary process noise and obs noise.
        # calcualte the gains while keep bias same
        anewparam_=torch.tensor(-dx*X[i]-dy*Y[j]+midpoint).float().view(-1,1)
        hnewparam_=torch.tensor(dx*X[i]+dy*Y[j]+midpoint).float().view(-1,1)

        anewpnoise=anewparam_[3]
        anewk = anewpnoise**2 / (anewparam_[5]**2+anewpnoise**2)
        anewg=abias/anewk
        anewparam_[1]=anewg


        hnewpnoise=hnewparam_[3]
        hnewk = hnewpnoise**2 / (hnewparam_[5]**2+hnewpnoise**2)
        hnewg=hbias/hnewk
        hnewparam_[1]=hnewg

        atheta=anewparam_.clone()
        htheta=hnewparam_.clone()
        paramls.append([atheta,htheta])
        z[i,j]=(atheta-htheta)[5]

# Z=ray.get([getlogll.remote(each) for each in paramls])
# with open('distinguishparamZtwonoisessmaller2finer19', 'wb+') as f:
#         pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)

Z=ray.get([getlogll.remote(each) for each in paramls])
with open('Z:/human/jointlikelihood/2noise3', 'wb+') as f:
    pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)
from notification import notify
notify()

import matplotlib.pyplot as plt
import numpy as np
import pickle
_,Z=pickle.load(open('Z:/human/jointlikelihood/2noise3', 'rb'))
plt.contourf(np.array(Z).reshape(gridreso,gridreso), oringin='image')


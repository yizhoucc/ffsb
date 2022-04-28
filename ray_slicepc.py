name='bruno'


from monkey_functions import datawash, monkey_data_downsampled
from InverseFuncs import monkeyloss_
import pickle
from pathlib import Path
import torch
import ray
import numpy as np
from plot_ult import *
import numpy as np
import matplotlib
from FireflyEnv import ffacc_real
from Config import Config
import numpy as np
import matplotlib.pyplot as plt
import warnings
from numpy.linalg.linalg import svd
warnings.filterwarnings('ignore')
import pickle
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
torch.manual_seed(42)
from numpy import pi
from InverseFuncs import *
from monkey_functions import *
from pathlib import Path
from plot_ult import *
from stable_baselines3 import TD3
import matplotlib

import os
import pandas as pd
from numpy.lib.npyio import save
os.chdir(r"C:\Users\24455\iCloudDrive\misc\ffsb")
import numpy as np
from cmaes import CMA
import copy
import matplotlib.pyplot as plt
# from numpy.core.defchararray import array
# from FireflyEnv.env_utils import is_pos_def
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
arg = Config()
import os


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
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()



torch.manual_seed(42)
matplotlib.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'





# slice pc -----------------------------------------------------
print('loading data')
datapath=Path("Z:\\{}_pert\".format(name))
with open(datapath/'packed','rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df=df[df.target_r>200]
print('process data')
states, actions, tasks=monkey_data_downsampled(df[110:120],factor=0.0025)
print('done process data')

def getlogll(x):
    with torch.no_grad():
        return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=0.01,num_iteration=1, states=states, samples=5,gpu=False).item()

with open(datapath/'cmafull_packed_{}_pert'.format(name), 'rb') as f:
    slog = pickle.load(f)
log=slog
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
npixel=20

finalcov=log[-1][0]._C
realfinalcov=np.cov(np.array([l[0] for l in log[-1][2]]).T)

finaltheta=log[-1][0]._mean
pcfinaltheta=((finaltheta.reshape(-1)-mu)@evectors).astype('float32')
finalcovpc=(evectors.T@finalcov@evectors).astype('float32')
realfinalcovpc=(evectors.T@realfinalcov@evectors).astype('float32')
pccov=evectors.T@(np.cov(alltheta.T))@evectors
allthetameanspc=((allthetameans-mu)@evectors).astype('float32')
with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(111)
    plot_cov_ellipse(pccov[:2,:2], pcfinaltheta[:2], alpha=0.3,nstd=1,ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    s = ax.scatter(x,y, c=z,alpha=0.5,edgecolors='None', cmap='jet')
    ax.set_xlabel('projected parameters')
    ax.set_ylabel('projected parameters')
    c = fig.colorbar(s, ax=ax)
    ax.locator_params(nbins=3, axis='y')
    ax.locator_params(nbins=3, axis='x')
    # c.clim(min(np.log(loglls)), max(np.log(loglls))) 
    c.set_label('- log likelihood')
    # c.set_ticks([min((loglls)),max((loglls))])
    c.ax.locator_params(nbins=4)
    # ax.set_xlim(xlow,xhigh)
    # ax.set_ylim(ylow,yhigh)
    ax.plot(allthetameanspc[:,0],allthetameanspc[:,1])

xlow,xhigh=pcfinaltheta[0]-0.5,pcfinaltheta[0]+0.5
ylow,yhigh=pcfinaltheta[1]-0.5,pcfinaltheta[1]+0.5
# xlow,xhigh=ax.get_xlim()
xrange=np.linspace(xlow,xhigh,npixel)
# ylow,yhigh=ax.get_ylim()
yrange=np.linspace(ylow,yhigh,npixel)
X,Y=np.meshgrid(xrange,yrange)

background_data=np.zeros((npixel,npixel))
for i,u in enumerate(xrange):
    for j,v in enumerate(yrange):
        score=np.array([u,v])
        reconstructed_theta=score@evectors.transpose()[:2,:]+mu
        reconstructed_theta=reconstructed_theta.clip(1e-4,999)
        background_data[i,j]=getlogll(reconstructed_theta.reshape(1,-1).astype('float32'))

with open('{}slicepc{}'.format(name,str(npixel)), 'wb') as handle:
        pickle.dump((obsvls, p), handle, protocol=pickle.HIGHEST_PROTOCOL)



a=background_data
plt.contourf(a)
plt.contourf(np.log(a))
norma=a
norma=(norma/(np.max(norma)-np.min(norma)))
norma-=np.min(norma)
plt.contourf(norma)

value=0.9
newa = np.where((1 - norma) < value,1,norma+value)
plt.contourf(newa)


with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(111)
    plot_cov_ellipse(pccov[:2,:2], pcfinaltheta[:2], alpha=1,nstd=1,ax=ax, edgecolor=[1,1,1])
    im=ax.contourf(X,Y,-newa,cmap='jet')
    # c = add_colorbar(im)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('projected parameters')
    ax.set_ylabel('projected parameters')
    c = fig.colorbar(im, ax=ax)
    ax.locator_params(nbins=3, axis='y')
    ax.locator_params(nbins=3, axis='x')
    # c.clim(min(np.log(loglls)), max(np.log(loglls))) 
    c.set_label('normalized likelihood')
    # c.set_ticks([min((loglls)),max((loglls))])
    c.ax.locator_params(nbins=4)
    ax.set_xlim(xlow,xhigh)
    ax.set_ylim(ylow,yhigh)
    ax.plot(allthetameanspc[:,0],allthetameanspc[:,1])




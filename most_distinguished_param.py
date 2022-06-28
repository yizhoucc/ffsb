

print(
'''
the most distinguished parameter that distinghuishs the ASD vs control.

define a midpoint
vary by d1,d2, where d1,d2 is delta theta in the interested direction
calcualte the logll for ASD and healthy seperatly, add together.
final result is a heat map
''')

# imports
# ---------------------------------------------------------------------------------
from playsound import playsound
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import os
import pandas as pd
os.chdir(r"C:\Users\24455\iCloudDrive\misc\ffsb")
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
from timeit import default_timer as timer
from plot_ult import ll2array, run_trial, sample_all_tasks, initiate_plot,process_inv, eval_curvature, xy2pol,suppress, quickspine, select_tar


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


# define the midpoint between ASD and healthy ------------------------------------------------


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



# function to eval the logll of the delta theta-----------------------------------------------
gridreso=11
num_sample=200
# load the data
datapath=Path("Z:/human/hgroup")
with open(datapath, 'rb') as f:
    hstates, hactions, htasks = pickle.load(f)
datapath=Path("Z:/human/agroup")
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
    
'''
complexity:
gridreso**2 *(ntrialT * nsample) * 2
'''












# vary noise and cost----------------------------------------------------------------------
dx,dy=np.zeros((11)),np.zeros((11))
dx[5]=1 # noise
dy[8]=1 # cost
X,Y=np.linspace(-0.7,0.7,gridreso), np.linspace(-0.45,0.45,gridreso)
paramls=[]
# Z=np.zeros((gridreso,gridreso))
for i in range(gridreso):
    for j in range(gridreso):
        atheta=torch.tensor(-dx*X[i]-dy*Y[j]+midpoint).float().view(-1,1)
        htheta=torch.tensor(dx*X[i]+dy*Y[j]+midpoint).float().view(-1,1)
        # eval the logll
        # Z[i,j]=atheta,htheta
        paramls.append([atheta,htheta])

# Z=ray.get([getlogll.remote(each) for each in paramls])
# with open('distinguishparamZnoisecost3finer19', 'wb+') as f:
#         pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)


# with open('distinguishparamZnoisecost3finer19', 'rb') as f:
#     paramls,Z= pickle.load(f)
# formatedZ=np.array(Z).reshape(int(len(Z)**0.5),int(len(Z)**0.5)).T
# truedelta=mus[1]-mus[0]
# with initiate_plot(3,3,300) as f:
#     ax=f.add_subplot(111)
#     # im=ax.imshow(formatedZ,origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),aspect='auto',vmin=-79, vmax=-73)
#     im=ax.imshow(formatedZ[1:-1],origin='lower', extent=(X[0],X[-2],Y[2],Y[-2]),aspect='auto',vmin=-79, vmax=-73)    # im=ax.imshow(formatedZ,origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),aspect='auto',vmin=-103, vmax=-73)
#     # im=ax.contourf(formatedZ,origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),vmin=-80, vmax=-73)
#     ax.set_aspect('equal')
#     plt.colorbar(im,label='joint log likelihood') 
#     ax.scatter(0,0,label='zero') # midpoint, 0,0
#     ax.set_xlabel('delta noise')
#     ax.set_ylabel('delta cost')
#     ax.scatter(truedelta[5]/2,truedelta[8]/2,label='inferred delta') # inferred delta
#     # ax.legend()
#     quickspine(ax)





# prediciton over obs -----------------------------------------------------
# vary obs noise, while try to keep simlar uncertainty and same bias
'''
constaints:
1, total uncertainty same. po/(p+o)=c
2, biased degree same. (1-k)*g, 
where k=p/(p+o), p and o are uncertainty variance 
'''
def totalbias(g,p,o):
    return g*(p**2/(p**2+o**2))
def totalgivenop(a,b):
    return (a**2)*(b**2)/((a**2)+(b**2))
def pogiventotal(a, c):
    return ((a**2)*c/((a**2)-c))**0.5


dx,dy=np.zeros((11)),np.zeros((11))
dx[5]=0.3 # noise
# dy[8]=0.45 # 
auncertainty=mus[0][5]**2*mus[0][3]**2/(mus[0][5]**2+mus[0][3]**2)
huncertainty=(mus[1][5]**2*mus[1][3]**2)/(mus[1][5]**2+mus[1][3]**2)
abias=totalbias(mus[0][1],mus[0][3],mus[0][5])
hbias=totalbias(mus[1][1],mus[1][3],mus[1][5])

c=totalgivenop(mus[0][5],mus[0][3])
pogiventotal(mus[0][5],auncertainty)

X,Y=np.linspace(-1,1,gridreso), np.linspace(-1,1,gridreso)
paramls=[]
for i in range(gridreso):
    for j in range(gridreso):
        # calcualte the dy, keep uncertainty same
        anewparam_=torch.tensor(-dx*X[i]+midpoint).float().view(-1,1)
        anewpnoise=pogiventotal(anewparam_[5],auncertainty)
        anewk = anewpnoise**2 / (anewparam_[5]**2+anewpnoise**2)
        anewg=abias/anewk
        anewparam_[3]=anewpnoise
        anewparam_[1]=anewg


        hnewparam_=torch.tensor(dx*X[i]+midpoint).float().view(-1,1)
        hnewpnoise=pogiventotal(hnewparam_[5],huncertainty)
        hnewk = hnewpnoise**2 / (hnewparam_[5]**2+hnewpnoise**2)
        hnewg=hbias/hnewk
        hnewparam_[3]=hnewpnoise
        hnewparam_[1]=hnewg

        atheta=anewparam_.clone()
        htheta=hnewparam_.clone()
        paramls.append([atheta,htheta])


# Z=ray.get([getlogll.remote(each) for each in paramls])
# with open('distinguishparamZobspre', 'wb+') as f:
#         pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)


# # vary obs, while keep uncertainty and kalman gain the same.
# with open('distinguishparamZobspre', 'rb') as f:
#     paramls,Z= pickle.load(f)
# formatedZ=np.array(Z).reshape(int(len(Z)**0.5),int(len(Z)**0.5)).T
# truedelta=mus[1]-mus[0]
# X=np.linspace(-1,1,gridreso)
# with initiate_plot(3,3,300) as f:
#     ax=f.add_subplot(111)
#     # ax.errorbar(X,np.mean(formatedZ,axis=0),yerr=np.array([np.mean(formatedZ,axis=0)-np.percentile(formatedZ,20,axis=0),np.percentile(formatedZ,80,axis=0)-np.mean(formatedZ,axis=0)]))
#     ax.errorbar(X,np.mean(formatedZ,axis=0),yerr=np.std(formatedZ,axis=0))
#     ax.set_xlabel('delta observation')
#     ax.set_ylabel('log likelihood')
#     ax.vlines(truedelta[5],np.min(formatedZ),np.max(formatedZ),'orange',label='inferred delta') # inferred delta
#     ax.vlines(0,np.min(formatedZ),np.max(formatedZ),label='zero') # inferred delta
#     ax.legend()
#     quickspine(ax)




# vary two noise ------------------------------------------------------------------------
'''
        # vary process noise and obs noise.
        # calcualte the gains while keep bias same
'''
dx,dy=np.zeros((11)),np.zeros((11))
dx[5]=1
dy[3]=1

auncertainty=mus[0][5]**2*mus[0][3]**2/(mus[0][5]**2+mus[0][3]**2)
huncertainty=(mus[1][5]**2*mus[1][3]**2)/(mus[1][5]**2+mus[1][3]**2)
abias=totalbias(mus[0][1],mus[0][3],mus[0][5])
hbias=totalbias(mus[1][1],mus[1][3],mus[1][5])

c=totalgivenop(mus[0][5],mus[0][3])
pogiventotal(mus[0][5],auncertainty)

X,Y=np.linspace(-0.5,0.5,gridreso),np.linspace(-0.3,0.3,gridreso) # on and pn
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

# # vary obs and process noise, while keep bias the same (so we vary gains), result in vary uncertainties, smaller range
# with open('distinguishparamZtwonoisessmaller2finer19', 'rb') as f:
#     paramls,Z= pickle.load(f)
# formatedZ=np.array(Z).reshape(int(len(Z)**0.5),int(len(Z)**0.5)).T
# truedelta=mus[1]-mus[0]

# with initiate_plot(3,3,300) as f:
#     ax=f.add_subplot(111)
#     im=ax.imshow(formatedZ,origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),aspect='auto',vmin=-79, vmax=-73)
#     # im=ax.imshow(formatedZ[:-1,:],origin='lower', extent=(X[0],X[-2],Y[0],Y[-1]),aspect='auto',vmin=-103, vmax=-73)
#     # im=ax.contourf(formatedZ,origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),vmin=-103, vmax=-73)
#     ax.set_aspect('equal')
#     plt.colorbar(im,label='joint log likelihood') 
#     ax.scatter(0,0,label='zero') # midpoint, 0,0
#     ax.set_xlabel('delta obs noise')
#     ax.set_ylabel('delta pro noise')
#     ax.scatter(truedelta[5]/2,truedelta[3]/2,label='inferred delta') # inferred delta
#     # ax.legend()
#     quickspine(ax)





# kalman gain vs uncertainty ---------------------------------------------------------------------

def getk(midpoint):
    return midpoint[5]**2/(midpoint[3]**2+midpoint[5]**2)

# ak=getk(mus[0])
# hk=getk(mus[1])
dx,dy=np.zeros((11)),np.zeros((11))
dx[5]=1
dy[3]=1

X,Y=np.linspace(-0.2,0.2,gridreso),np.linspace(-0.7,0.7,gridreso) # k, uncertainty

paramls=[]
for i in range(gridreso):
    for j in range(gridreso):

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


# bias vs uncertainty ---------------------------------------------------------------------
# vary obs noise and gain

'''
to vary the uncertainty, we vary po/(p+o)
the range is 0, 1/2
to vary the bias, we vary (1-k)*g, 
    where k=p/(p+o), thus pg/(p+o)
this is equvalent to vary o and g, while solve for p that fit the wantted uncertainty.
'''

X,Y=np.linspace(-0.7,0.7,gridreso),np.linspace(-1,1,gridreso) # o, gain
dx,dy=np.zeros((11)),np.zeros((11))
dx[5]=1
dy[1]=1

paramls=[]
for i in range(gridreso):
    for j in range(gridreso):
        anewparam_=torch.tensor(-dx*X[i]-dy*Y[j]+midpoint).float().view(-1,1)
        hnewparam_=torch.tensor(dx*X[i]+dy*Y[j]+midpoint).float().view(-1,1)
        # print(anewparam_[1],anewparam_[5])
        # print(hnewparam_[1],hnewparam_[5])
        paramls.append([anewparam_,hnewparam_])

# Z=ray.get([getlogll.remote(each) for each in paramls])
# with open('distinguishparamZgainonoisefiner19', 'wb+') as f:
#         pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)

# # vary gain and noise (vary bias vs uncertainty)
# with open('distinguishparamZgainonoisefiner19', 'rb') as f:
#     paramls,Z= pickle.load(f)
# formatedZ=np.array(Z).reshape(int(len(Z)**0.5),int(len(Z)**0.5)).T
# truedelta=mus[1]-mus[0]
# X,Y=np.linspace(-0.7,0.7,gridreso),np.linspace(-1,1,gridreso) # o, gain
# with initiate_plot(3,3,300) as f:
#     ax=f.add_subplot(111)
#     # im=ax.imshow(formatedZ,origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]), aspect='auto')
#     im=ax.imshow(formatedZ,origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),aspect='auto',vmin=-84, vmax=-73)
#     # im=ax.contourf(formatedZ,origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),vmin=-103, vmax=-73)
#     ax.set_aspect('equal')
#     plt.colorbar(im,label='joint log likelihood') 
#     ax.scatter(0,0,label='delta zero') # midpoint, 0,0
#     ax.set_xlabel('delta obs noise')
#     ax.set_ylabel('delta assumed control gain')
#     ax.scatter(truedelta[5]/2,truedelta[1]/2,label='inferred delta') # inferred delta
#     ax.legend(loc='center left', bbox_to_anchor=(2, 0.5))
#     quickspine(ax)


# with initiate_plot(3,3,300) as f:
#     ax=f.add_subplot(111)
#     # im=ax.imshow(formatedZ[2:-2],origin='lower', extent=(X[0],X[-1],Y[2],Y[-3]), aspect='auto')
#     im=ax.imshow(formatedZ[:,2:],origin='lower', extent=(X[0],X[-1],Y[0],Y[-3]),aspect='auto',vmin=-103, vmax=-73)
#     # im=ax.contourf(formatedZ[2:-2],origin='lower', extent=(X[0],X[-1],Y[2],Y[-3]),vmin=-103, vmax=-73)
#     ax.set_aspect('equal')
#     plt.colorbar(im,label='joint log likelihood') 
#     ax.scatter(0,0,label='delta zero') # midpoint, 0,0
#     ax.set_xlabel('delta uncertainty')
#     ax.set_ylabel('delta prediction bias')
#     ax.scatter(truedelta[5]/2,truedelta[1]/2,label='inferred delta') # inferred delta
#     ax.legend(loc='center left', bbox_to_anchor=(2, 0.5))
#     quickspine(ax)

# ks,ss=transform2bs(paramls)
# plt.tricontourf(ks,ss,formatedZ.flatten(),3)

# plot the bias instead of gain.
# bias=assumed gain * (1-kalman gain)

# bs,ss=transform2bs(paramls)
# with initiate_plot(3,3,300) as f:
#     ax=f.add_subplot(111)
#     im=ax.imshow(formatedZ,origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]), aspect='auto')
#     ax.set_aspect('equal')
#     plt.colorbar(im,label='joint log likelihood') 
#     ax.scatter(0,0,label='delta zero') # midpoint, 0,0
#     ax.set_xlabel('delta obs noise')
#     ax.set_ylabel('delta assumed control gain')
#     ax.scatter(truedelta[5]/2,truedelta[1]/2,label='inferred delta') # inferred delta
#     ax.legend(loc='center left', bbox_to_anchor=(2, 0.5))
#     quickspine(ax)

# plt.tricontourf(bs,ss,formatedZ.flatten(),3)

# transform2bs([thetas.view(2,-1,1)])
# thetas.view(2,-1,1)[0]


# shwo bias matters ---------------------------------------------------------------------

'''
vary gain, and vary some other param. vary cost here.
'''
# vary gain and cost, show bias matters
gridreso=17
X,Y=np.linspace(-0.45,0.45,gridreso),np.linspace(-1,1,gridreso) # cost, gain
dx,dy=np.zeros((11)),np.zeros((11))
dx[8]=1 # cost
dy[1]=1 # gain

paramls=[]
for i in range(gridreso):
    for j in range(gridreso):
        anewparam_=torch.tensor(-dx*X[i]-dy*Y[j]+midpoint).float().view(-1,1)
        hnewparam_=torch.tensor(dx*X[i]+dy*Y[j]+midpoint).float().view(-1,1)
        # print(anewparam_[1],anewparam_[5])
        # print(hnewparam_[1],hnewparam_[5])
        paramls.append([anewparam_,hnewparam_])

# Z=ray.get([getlogll.remote(each) for each in paramls])
# with open('distinguishparamZcostgain2', 'wb+') as f:
#         pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)

with open('distinguishparamZcostgain2', 'rb+') as f:
    paramls,Z= pickle.load(f)
formatedZ=np.array(Z).reshape(int(len(Z)**0.5),int(len(Z)**0.5)).T
truedelta=mus[1]-mus[0]

with initiate_plot(3,3,300) as f:
    ax=f.add_subplot(111)
    # im=ax.imshow(formatedZ[:,2:],origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),aspect='auto')
    im=ax.imshow(formatedZ[:,1:-1],origin='lower', extent=(X[0],X[-1],Y[1],Y[-2]),aspect='auto',vmin=-103, vmax=-73)
    # im=ax.contourf(formatedZ,origin='lower', extent=(X[2],X[-1],Y[0],Y[-1]),vmin=-103, vmax=-73)
    ax.set_aspect('equal')
    plt.colorbar(im,label='joint log likelihood') 
    ax.scatter(0,0,label='zero') # midpoint, 0,0
    ax.set_xlabel('delta w cost')
    ax.set_ylabel('delta assumed control gain')
    ax.scatter(truedelta[5]/2,truedelta[3]/2,label='inferred delta') # inferred delta
    # ax.legend()
    quickspine(ax)



# show v does not matter
# vary v and w cost ----------------------------------------------------------------------
dx,dy=np.zeros((11)),np.zeros((11))
dx[8]=1 
dy[7]=1 
X,Y=np.linspace(-0.45,0.3,gridreso), np.linspace(-0.9,0.9,gridreso)
paramls=[]
# Z=np.zeros((gridreso,gridreso))
for i in range(gridreso):
    for j in range(gridreso):
        atheta=torch.tensor(-dx*X[i]-dy*Y[j]+midpoint).float().view(-1,1)
        htheta=torch.tensor(dx*X[i]+dy*Y[j]+midpoint).float().view(-1,1)
        # eval the logll
        # Z[i,j]=atheta,htheta
        paramls.append([atheta,htheta])

# Z=ray.get([getlogll.remote(each) for each in paramls])
# with open('distinguishparamZvwcost2', 'wb+') as f:
#         pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)
# [p[0][8] for p in paramls]
# [p[1][8] for p in paramls]

# vary v and w cost
with open('distinguishparamZvwcost2', 'rb+') as f:
    paramls,Z= pickle.load(f)
formatedZ=np.array(Z).reshape(int(len(Z)**0.5),int(len(Z)**0.5)).T
truedelta=mus[1]-mus[0]

with initiate_plot(3,3,300) as f:
    ax=f.add_subplot(111)
    # im=ax.imshow(formatedZ[:,2:],origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),aspect='auto')
    im=ax.imshow(formatedZ[:,2:],origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),aspect='equal',vmin=-103, vmax=-73)
    plt.colorbar(im,label='joint log likelihood') 
    ax.scatter(0,0,label='zero') # midpoint, 0,0
    ax.set_xlabel('delta w cost')
    ax.set_ylabel('delta v cost')
    ax.scatter(truedelta[5]/2,truedelta[3]/2,label='inferred delta') # inferred delta
    # ax.legend()
    quickspine(ax)

# vary v and w gain ----------------------------------------------------------------------
dx,dy=np.zeros((11)),np.zeros((11))
dx[1]=1 
dy[0]=1 
X,Y=np.linspace(-1,1,gridreso) , np.linspace(-1,1,gridreso) 
paramls=[]
for i in range(gridreso):
    for j in range(gridreso):
        atheta=torch.tensor(-dx*X[i]-dy*Y[j]+midpoint).float().view(-1,1)
        htheta=torch.tensor(dx*X[i]+dy*Y[j]+midpoint).float().view(-1,1)
        paramls.append([atheta,htheta])
# Z=ray.get([getlogll.remote(each) for each in paramls])

# with open('distinguishparamZvwgain', 'wb+') as f:
#         pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)


with open('distinguishparamZvwgain', 'rb+') as f:
    paramls,Z= pickle.load(f)
formatedZ=np.array(Z).reshape(int(len(Z)**0.5),int(len(Z)**0.5)).T
truedelta=mus[1]-mus[0]

with initiate_plot(3,3,300) as f:
    ax=f.add_subplot(111)
    # im=ax.imshow(formatedZ[:,2:],origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),aspect='auto')
    im=ax.imshow(formatedZ[2:-2,2:],origin='lower', extent=(X[0],X[-1],Y[0],Y[-1]),aspect='equal',vmin=-103, vmax=-73)
    plt.colorbar(im,label='joint log likelihood') 
    ax.scatter(0,0,label='zero') # midpoint, 0,0
    ax.set_xlabel('delta w gain')
    ax.set_ylabel('delta v gain')
    ax.scatter(truedelta[5]/2,truedelta[3]/2,label='inferred delta') # inferred delta
    # ax.legend()
    quickspine(ax)




















# vary v and w cost, eval the curvature   ----------------------------------------------
# ASD-health

# curvature 1, total angle turned---------------------------------------------------
datapath=Path("Z:/human/hgroup")
with open(datapath, 'rb') as f:
    hstates, hactions, htasks = pickle.load(f)
res=[]
for task in htasks:
    d,a=xy2pol(task, rotation=False)
    # if  env.min_angle/2<=a<env.max_angle/2:
    if a<=-pi/5*0.7 or a>=pi/5*0.7:
        res.append(task)
sidetasks=np.array(res)

sidetasks=sidetasks[:100]

@ray.remote
def getcur(x):
    x.clamp_(1e-3,3)
    with torch.no_grad():
        curvature=eval_curvature(agent, env, phi, x,sidetasks,vctrl=0.,wctrl=0.,ntrials=50)
        return  curvature

dx,dy=np.zeros((11)),np.zeros((11))
dx[8]=1 
dy[7]=1 

X,Y=np.linspace(-0.45,1,gridreso), np.linspace(-1.1,0.4,gridreso)
paramls=[]
# Z=np.zeros((gridreso,gridreso))
for i in range(gridreso):
    for j in range(gridreso):
        theta=torch.tensor(-dx*X[i]-dy*Y[j]+midpoint).float().view(-1,1)
        paramls.append(theta)

# Z=ray.get([getcur.remote(each) for each in paramls])

# with open('distinguishparamZvwcostcurvaturesingle', 'wb+') as f:
#         pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)

# vary v and w cost, heatmap of curvature
with open('distinguishparamZvwcostcurvaturesingle', 'rb+') as f:
    paramls,Z= pickle.load(f)
formatedZ=np.array(Z).reshape(int(len(Z)**0.5),int(len(Z)**0.5)).T
X,Y=np.linspace(-0.45,1,gridreso), np.linspace(-1.1,0.4,gridreso)

with initiate_plot(3,3,300) as f:
    ax=f.add_subplot(111)
    im=ax.imshow(formatedZ[:,2:],origin='lower', extent=(X[0]+midpoint[8],X[-1]+midpoint[8],Y[0]+midpoint[7],Y[-1]+midpoint[7]),aspect='auto')
    plt.colorbar(im,label='curvature') 
    ax.scatter(0,0,label='zero') # midpoint, 0,0
    ax.set_xlabel('delta w cost')
    ax.set_ylabel('delta v cost')
    # ax.scatter(truedelta[5]/2,truedelta[3]/2,label='inferred delta') # inferred delta
    # ax.legend()


# using curvature 2, the furthest point -----------------------------------------------------
def eval_curvature2(agent, env, phi, theta, tasks,ntrials=10):
    costs=[]
    states=[]
    with suppress():
        for task in tasks:
            while len(states)<ntrials:
                env.reset(phi=phi, theta=theta, goal_position=task, pro_traj=None,vctrl=0.,wctrl=0. )
                _,_,_,epstates=run_trial(agent,env,given_action=None, given_state=None, action_noise=0.1)
                if len(epstates)>5:
                    states.append(torch.stack(epstates)[:,:,0])
    for s in states:
        end=s[-1][:2]
        rotation=xy2pol(end,rotation=False)[1].item()
        R=np.array([[np.cos(rotation),np.sin(rotation)],[-np.sin(rotation),np.cos(rotation)]])
        rotatedxy=R@np.array(astates[i][:,:2].T)
        epcost=np.max(rotatedxy[1])
        costs.append(epcost)    
    return sum(costs)

@ray.remote
def getcur(x):
    x.clamp_(1e-3,3)
    with torch.no_grad():
        curvature=eval_curvature2(agent, env, phi, theta, [None]*50, ntrials=50)
        return  curvature

dx,dy=np.zeros((11)),np.zeros((11))
dx[8]=1 
dy[7]=1 

X,Y=np.linspace(-0.45,1,gridreso), np.linspace(-1.1,0.4,gridreso)
paramls=[]
# Z=np.zeros((gridreso,gridreso))
for i in range(gridreso):
    for j in range(gridreso):
        theta=torch.tensor(-dx*X[i]-dy*Y[j]+midpoint).float().view(-1,1)
        paramls.append(theta)

# Z=ray.get([getcur.remote(each) for each in paramls])

# with open('distinguishparamZvwcostcurvature2', 'wb+') as f:
#         pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)



# vary v and w cost, heatmap of curvature
with open('distinguishparamZvwcostcurvature2', 'rb+') as f:
    paramls,Z= pickle.load(f)
formatedZ=np.array(Z).reshape(int(len(Z)**0.5),int(len(Z)**0.5)).T
X,Y=np.linspace(-0.45,1,gridreso), np.linspace(-1.1,0.4,gridreso)

with initiate_plot(3,3,300) as f:
    ax=f.add_subplot(111)
    im=ax.imshow(formatedZ[:,2:],origin='lower', extent=(X[0]+midpoint[8],X[-1]+midpoint[8],Y[0]+midpoint[7],Y[-1]+midpoint[7]),aspect='auto')
    plt.colorbar(im,label='curvature') 
    ax.scatter(0,0,label='zero') # midpoint, 0,0
    ax.set_xlabel('delta w cost')
    ax.set_ylabel('delta v cost')
    # ax.scatter(truedelta[5]/2,truedelta[3]/2,label='inferred delta') # inferred delta
    # ax.legend()



def transform2ks(paramls):
    # transform the param list to kalman gain vs total uncertainty 
    ks=[] # kalman gain diffs
    for ta, th in paramls:
        ks.append((getk(ta)-getk(th)).item())
    ss=[] # uncertainty diffs
    for ta, th in paramls:
        ss.append((totalgivenop(ta[3],ta[5])-totalgivenop(th[3],th[5])).item())

    plt.scatter(ks,ss)
    return ks,ss

def transform2bs(paramls):
    # transform the param list to controlgain*(1-kalman gain) vs total uncertainty 
    ks=[] # kalman gain diffs
    for ta, th in paramls:
        ks.append((ta[1]*(1-getk(ta))-th[1]*(1-getk(th))).item())
    ss=[] # uncertainty diffs
    for ta, th in paramls:
        ss.append((totalgivenop(ta[3],ta[5])-totalgivenop(th[3],th[5])).item())

    plt.scatter(ks,ss)
    return ks,ss









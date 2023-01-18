

print(
'''
does group inverse 'covers' the individual inverse population?
if true, than we think the group inverse can represent individuals.
otherwise, we need to think more.



''')

# imports
# -------------------------------------------------------------

from sklearn.metrics import roc_curve, auc

from scipy.stats import multivariate_normal
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import matplotlib
from playsound import playsound
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
import numpy as np
import os
import pandas as pd
from tensorboard import notebook
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
from numpy import linspace, pi
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
from plot_ult import *


# load agent and task --------------------------------------------------------
env=ffacc_real.FireFlyPaper(arg)
env.episode_len=50
env.debug=1
env.terminal_vel=0.2
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

# load the individual subject logs --------------------
numhsub,numasub=25,14
foldername='persub1cont'
logs={'a':'Z:/human/fixragroup','h':'Z:/human/clusterpaperhgroup'}

invres={'a':[],'h':[]}
for isub in range(numhsub):
    dataname="hsub{}".format(str(isub))
    savename=Path("Z:/human/{}".format(foldername))/"invhsub{}".format(str(isub))
    if savename.is_file():
        invres['h'].append(process_inv(savename,ind=31, usingbest=True, removegr=False))
for isub in range(numasub):
    dataname="asub{}".format(str(isub))
    savename=Path("Z:/human/{}".format(foldername))/"invasub{}".format(str(isub))
    if savename.is_file():
        invres['a'].append(process_inv(savename,ind=31, usingbest=True,removegr=False))

invres['agroup']=process_inv('Z:/human/fixragroup', usingbest=True)
invres['hgroup']=process_inv('Z:/human/clusterpaperhgroup', usingbest=True)

      

#--------------------------------------------------------------------------

print(
'''
try to split the two group with svm. 3d plot can rotate

''')

%matplotlib qt
alltheta,alltag=[],[]
allcov=[]
for theta,cov,_ in invres['a']:
    alltheta.append(theta)
    allcov.append(cov)
    alltag.append(1)
for theta,cov,_ in invres['h'][:len(invres['a'])]:
    alltheta.append(theta)
    allcov.append(cov)
    alltag.append(0)
    
alltheta=np.array(torch.cat(alltheta,axis=1)).T
alltag=np.array(alltag)
X, Y=alltheta,alltag
X = X[np.logical_or(Y==0,Y==1)]
Y = Y[np.logical_or(Y==0,Y==1)]


model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)
def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('parameter coef')
    ax=plt.gca()
    quickspine(ax)

f_importances(np.abs(clf.coef_[0]),theta_names_)

axis=list(np.argsort(np.abs(clf.coef_))[0])[::-1][:3]
print('top parameter index',axis)
z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][axis[0]]*x -clf.coef_[0][axis[1]]*y) / clf.coef_[0][axis[2]]
tmp = np.linspace(0,2,30)
x,y = np.meshgrid(tmp,tmp)
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot3D(X[Y==0,axis[0]], X[Y==0,axis[1]], X[Y==0,axis[2]],'ob',label='health control')
ax.plot3D(X[Y==1,axis[0]], X[Y==1,axis[1]], X[Y==1,axis[2]],'sr',label='ASD')
ax.plot_surface(x, y, z(x,y))
ax.set_xlim(0,2); ax.set_xlabel(theta_names_[axis[0]])
ax.set_ylim(0,2); ax.set_ylabel(theta_names_[axis[1]])
ax.set_zlim(-0.1,1.1); ax.set_zlabel(theta_names_[axis[2]])
quickleg(ax)
plt.show()



#--------------------------------------------------------------------------
print(
'''
try to split the two group samples with svm (with samples)

''')
%matplotlib inline
# load data
numhsub,numasub=25,14
foldername='persub1cont'
logs={'a':'Z:/human/fixragroup','h':'Z:/human/clusterpaperhgroup'}

invres={'a':[],'h':[]}
for isub in range(numhsub):
    dataname="hsub{}".format(str(isub))
    savename=Path("Z:/human/{}".format(foldername))/"invhsub{}".format(str(isub))
    if savename.is_file():
        invres['h'].append(process_inv(savename,ind=31, usingbest=True, removegr=True))
for isub in range(numasub):
    dataname="asub{}".format(str(isub))
    savename=Path("Z:/human/{}".format(foldername))/"invasub{}".format(str(isub))
    if savename.is_file():
        invres['a'].append(process_inv(savename,ind=31, usingbest=True,removegr=True))

invres['agroup']=process_inv('Z:/human/fixragroup', usingbest=True)
invres['hgroup']=process_inv('Z:/human/clusterpaperhgroup', usingbest=True)

# we have 3 ways. just mean, mean and variance, mean and covariance

# just mean svm
alltheta,alltag=[],[]
allcov=[]
for _ in invres['h']:
    for theta,cov,_ in invres['a']:
        alltheta.append(theta)
        allcov.append(cov)
        alltag.append(1)
for _ in invres['a']:
    for theta,cov,_ in invres['h']:
        alltheta.append(theta)
        allcov.append(cov)
        alltag.append(0)
    
alltheta=np.array(torch.cat(alltheta,axis=1)).T
alltag=np.array(alltag)
X, Y=alltheta[:,:8],alltag
X = X[np.logical_or(Y==0,Y==1)]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)
f_importances(np.abs(clf.coef_[0]),theta_names)

# adding auc, roc
y_test_pred = clf.decision_function(X) 
test_fpr, test_tpr, te_thresholds = roc_curve(Y, y_test_pred)
plt.grid()
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()



axis=list(np.argsort(np.abs(clf.coef_))[0])[::-1][:2]
axis=[1,0]
print('top parameter index',axis)
y = lambda x: (-clf.intercept_[0]-clf.coef_[0][axis[0]]*x ) / clf.coef_[0][axis[1]]
x=  np.linspace(0,2,30)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.scatter(X[Y==0,axis[0]], X[Y==0,axis[1]],color='b',label='health control',alpha=0.01)
ax.scatter(X[Y==1,axis[0]], X[Y==1,axis[1]],color='r',label='ASD',alpha=0.01)
ax.plot(x, y(x))
ax.set_xlim(0,2); ax.set_xlabel(theta_names_[axis[0]])
ax.set_ylim(0,2); ax.set_ylabel(theta_names_[axis[1]])
quickleg(ax)
plt.show()

print('''
project the individual thetas on to the normal vector.
''')
w=clf.coef_[0]
ticks=X[:,:8].dot(w)
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(ticks[Y==0],density=True,color='b',bins=22,label='health control',alpha=0.6)
ax.hist(ticks[Y==1],density=True,color='r',bins=22,label='ASD',alpha=0.6)
quickleg(ax)
quickspine(ax)
ax.set_xlabel('param value')
ax.set_ylabel('probability')
# quicksave('asd group project svm normal vector no init uncertainty')



# mean and variance svm
numsamples=100
adjustratio=len(invres['h'])/len(invres['a'])
alltag=[]
allsamples=[]
for theta,cov,_ in invres['a']:
    # newcov=torch.pow(torch.abs(cov),0.5)*torch.sign(cov)
    # newcov=cov
    newcov=cov*0.01
    # distribution=MultivariateNormal(theta.view(-1),newcov)
    newcov=torch.eye(10)*torch.diag(cov)
    distribution=MultivariateNormal(theta.view(-1),newcov)
    samples=[]
    while len(samples)<int(numsamples*adjustratio):
        a=distribution.sample()
        if torch.all(a.clamp(0,2)==a):
            samples.append(a)
    allsamples.append(torch.stack(samples))
    alltag+=[1]*int(numsamples*adjustratio)
for theta,cov,_ in invres['h']:
    # newcov=torch.pow(torch.abs(cov),0.5)*torch.sign(cov)
    # newcov=cov
    newcov=cov*0.01
    # distribution=MultivariateNormal(theta.view(-1),newcov)
    newcov=torch.eye(10)*torch.diag(cov)
    distribution=MultivariateNormal(theta.view(-1),newcov)
    samples=[]
    while len(samples)<numsamples:
        a=distribution.sample()
        if torch.all(a.clamp(0,2)==a):
            samples.append(a)
    allsamples.append(torch.stack(samples))
    alltag+=[0]*numsamples

allsamples=np.array(torch.cat(allsamples,axis=0))
alltag=np.array(alltag).astype('int')
X, Y=allsamples,alltag
X = X[np.logical_or(Y==0,Y==1)][:,:8]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)
f_importances(np.abs(clf.coef_[0]),theta_names)

# adding auc, roc
y_test_pred = clf.decision_function(X) 
test_fpr, test_tpr, te_thresholds = roc_curve(Y, y_test_pred)
plt.grid()
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()


axis=list(np.argsort(np.abs(clf.coef_))[0])[::-1][:2]
axis=[1,0]
print('top parameter index',axis)
y = lambda x: (-clf.intercept_[0]-clf.coef_[0][axis[0]]*x ) / clf.coef_[0][axis[1]]
x=  np.linspace(0,2,30)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.scatter(X[Y==0,axis[0]], X[Y==0,axis[1]],color='b',label='health control',alpha=0.01)
ax.scatter(X[Y==1,axis[0]], X[Y==1,axis[1]],color='r',label='ASD',alpha=0.01)
ax.plot(x, y(x))
ax.set_xlim(0,2); ax.set_xlabel(theta_names_[axis[0]])
ax.set_ylim(0,2); ax.set_ylabel(theta_names_[axis[1]])
quickleg(ax)
plt.show()

print('''
project the individual thetas on to the normal vector.
''')
w=clf.coef_[0]
ticks=X[:,:8].dot(w)
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(ticks[Y==0],density=True,color='b',bins=22,label='health control',alpha=0.6)
ax.hist(ticks[Y==1],density=True,color='r',bins=22,label='ASD',alpha=0.6)
ax.set_xlabel('param value')
ax.set_ylabel('probability')
quickspine(ax)
quickleg(ax)
# quicksave('asd group project svm normal vector no init uncertainty')

from matplotlib import cm
norm = matplotlib.colors.Normalize(vmin=0, vmax=len(invres['a']))
colors=[]
for i in range(len(invres['a'])):
    rgba_color = cm.autumn(norm(i),bytes=False) 
    rgb_color=rgba_color[:3]
    colors.append(rgb_color)
numsamplesasd=int(numsamples*adjustratio)
w=clf.coef_[0]
asdticks=X[Y==1,:8].dot(w).reshape(numsamplesasd,-1)
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(asdticks,density=True,bins=22,label='ASD',color=colors,alpha=0.5, histtype='bar',stacked=True)

norm = matplotlib.colors.Normalize(vmin=0, vmax=len(invres['h']))
colors=[]
for i in range(len(invres['h'])):
    rgba_color = cm.winter(norm(i),bytes=False) 
    rgb_color=rgba_color[:3]
    colors.append(rgb_color)
w=clf.coef_[0]
healthyticks=X[Y==0,:8].dot(w).reshape(numsamples,-1)
ax.hist(healthyticks,density=True,bins=22,label='health control',color=colors,alpha=0.5, histtype='bar',stacked=True)


fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(ticks[Y==0],density=True,color='b',bins=22,label='health control',alpha=0.6)
ax.hist(ticks[Y==1],density=True,color='r',bins=22,label='ASD',alpha=0.6)
quickleg(ax)
quickspine(ax)
ax.set_xlabel('param value')
ax.set_ylabel('probability')


# mean and covariance svm
numsamples=100
adjustratio=len(invres['h'])/len(invres['a'])
alltag=[]
allsamples=[]
for theta,cov,_ in invres['a']:
    distribution=MultivariateNormal(theta.view(-1),cov*0.01)
    samples=[]
    while len(samples)<int(numsamples*adjustratio):
        a=distribution.sample()
        if torch.all(a.clamp(0,2)==a):
            samples.append(a)
    allsamples.append(torch.stack(samples))
    alltag+=[1]*int(numsamples*adjustratio)
for theta,cov,_ in invres['h']:
    distribution=MultivariateNormal(theta.view(-1),cov*0.01)
    samples=[]
    while len(samples)<numsamples:
        a=distribution.sample()
        if torch.all(a.clamp(0,2)==a):
            samples.append(a)
    allsamples.append(torch.stack(samples))
    alltag+=[0]*numsamples

allsamples=np.array(torch.cat(allsamples,axis=0))
alltag=np.array(alltag).astype('int')
X, Y=allsamples,alltag
X = X[np.logical_or(Y==0,Y==1)][:,:8]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)
f_importances(np.abs(clf.coef_[0]),theta_names)
# quicksave('svm mean cov coef')



# adding auc, roc
y_test_pred = clf.decision_function(X) 
test_fpr, test_tpr, te_thresholds = roc_curve(Y, y_test_pred)
plt.grid()
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()



axis=list(np.argsort(np.abs(clf.coef_))[0])[::-1][:2]
axis=[1,0]
print('top parameter index',axis)
y = lambda x: (-clf.intercept_[0]-clf.coef_[0][axis[0]]*x ) / clf.coef_[0][axis[1]]
x=  np.linspace(0,2,30)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.scatter(X[Y==0,axis[0]], X[Y==0,axis[1]],color='b',label='health control',alpha=0.01)
ax.scatter(X[Y==1,axis[0]], X[Y==1,axis[1]],color='r',label='ASD',alpha=0.01)
ax.plot(x, y(x))
ax.set_xlim(0,2); ax.set_xlabel(theta_names_[axis[0]])
ax.set_ylim(0,2); ax.set_ylabel(theta_names_[axis[1]])
quickleg(ax)
plt.show()

print('''
project the individual thetas on to the normal vector.
''')
w=clf.coef_[0]
ticks=X[:,:8].dot(w)
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(ticks[Y==0],density=True,color='b',bins=22,label='health control',alpha=0.6)
ax.hist(ticks[Y==1],density=True,color='r',bins=22,label='ASD',alpha=0.6)
ax.set_xlabel('param value')
ax.set_ylabel('probability')
quickspine(ax)
# quickleg(ax)
# quicksave('asd group project svm normal vector no init uncertainty')


from matplotlib import cm
norm = matplotlib.colors.Normalize(vmin=0, vmax=len(invres['a']))
colors=[]
for i in range(len(invres['a'])):
    rgba_color = cm.autumn(norm(i),bytes=False) 
    rgb_color=rgba_color[:3]
    colors.append(rgb_color)
numsamplesasd=int(numsamples*adjustratio)
w=clf.coef_[0]
asdticks=X[Y==1,:8].dot(w).reshape(numsamplesasd,-1)
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(asdticks,density=True,bins=22,label='ASD',alpha=1, histtype='bar',stacked=True)

norm = matplotlib.colors.Normalize(vmin=0, vmax=len(invres['h']))
colors=[]
for i in range(len(invres['h'])):
    rgba_color = cm.winter(norm(i),bytes=False) 
    rgb_color=rgba_color[:3]
    colors.append(rgb_color)
w=clf.coef_[0]
healthyticks=X[Y==0,:8].dot(w).reshape(numsamples,-1)
ax.hist(healthyticks,density=True,bins=22,label='health control',color=colors,alpha=1, histtype='bar',stacked=True)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(asdticks,density=True,bins=22,label='ASD',alpha=1, histtype='bar',stacked=True)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(healthyticks,density=True,bins=22,label='health control',alpha=1, histtype='bar',stacked=True)


#--------------------------------------------------------------------------
print(
'''
get all individual ellipse, project it to some axies together with group inverse
''')

axis=[0,1]
paramspace=[0,2]
reso=100

def main():
    fig=plt.figure(figsize=(8,8))
    # plot the stack of individual inverses
    ax=fig.add_subplot(221)
    quickspine(ax)
    ax.set_xlim(paramspace); ax.set_xlabel(theta_names_[axis[0]])
    ax.set_ylim(paramspace); ax.set_ylabel(theta_names_[axis[1]])
    ax.axis('equal')
    X,Y=np.meshgrid(np.linspace(paramspace[0],paramspace[1], reso),np.linspace(paramspace[0],paramspace[1], reso))
    pos = np.dstack((X, Y))
    groudtruth=[]
    for mu,cov,_ in invres['h']:
        cov=np.array(cov)
        mu=np.array(mu.view(-1))
        mu=mu[axis]
        cov=cov[axis][:,axis]
        surface=multivariate_normal(mu,cov)
        subZ=np.log(surface.pdf(pos))
        groudtruth.append(subZ)
        im=ax.contour(X,Y,subZ, alpha=0.2)
    mu,cov,_ = invres['hgroup']
    cov=np.array(cov)
    mu=np.array(mu.view(-1))
    mu=mu[axis]
    cov=cov[axis][:,axis]
    surface=multivariate_normal(mu,cov)
    subZ=np.log(surface.pdf(pos))
    im=ax.contour(X,Y,subZ, alpha=0.9,cmap='autumn')
    plt.colorbar(im, label='-log likelihood',alpha=1)

    ax=fig.add_subplot(223)
    quickspine(ax)
    ax.set_xlim(paramspace); ax.set_xlabel(theta_names_[axis[0]])
    ax.set_ylim(paramspace); ax.set_ylabel(theta_names_[axis[1]])
    ax.axis('equal')
    groupZ=np.sum(groudtruth, axis=0)
    im=ax.contour(X,Y,groupZ, alpha=1,linestyles='solid', label='summed group')
    plt.colorbar(im, label='-log likelihood',alpha=1)
    ax.set_title('group inverse')
    im=ax.contour(X,Y,subZ, alpha=1,linestyles='solid', label='inferred group', cmap='autumn')
    plt.colorbar(im)

    # ASD
    ax=fig.add_subplot(222)
    quickspine(ax)
    ax.set_xlim(paramspace); ax.set_xlabel(theta_names_[axis[0]])
    ax.set_ylim(paramspace); ax.set_ylabel(theta_names_[axis[1]])
    ax.axis('equal')
    X,Y=np.meshgrid(np.linspace(paramspace[0],paramspace[1], reso),np.linspace(paramspace[0],paramspace[1], reso))
    pos = np.dstack((X, Y))
    groudtruth=[]
    for mu,cov,_ in invres['a']:
        cov=np.array(cov)
        mu=np.array(mu.view(-1))
        mu=mu[axis]
        cov=cov[axis][:,axis]
        surface=multivariate_normal(mu,cov)
        subZ=np.log(surface.pdf(pos))
        groudtruth.append(subZ)
        im=ax.contour(X,Y,subZ, alpha=0.2)
    mu,cov,_ = invres['agroup']
    cov=np.array(cov)
    mu=np.array(mu.view(-1))
    mu=mu[axis]
    cov=cov[axis][:,axis]
    surface=multivariate_normal(mu,cov)
    subZ=np.log(surface.pdf(pos))
    im=ax.contour(X,Y,subZ, alpha=0.9,cmap='autumn')
    plt.colorbar(im, label='-log likelihood',alpha=1)

    ax=fig.add_subplot(224)
    quickspine(ax)
    ax.set_xlim(paramspace); ax.set_xlabel(theta_names_[axis[0]])
    ax.set_ylim(paramspace); ax.set_ylabel(theta_names_[axis[1]])
    ax.axis('equal')
    groupZ=np.sum(groudtruth, axis=0)
    im=ax.contour(X,Y,groupZ, alpha=1,linestyles='solid', label='summed group')
    plt.colorbar(im, label='-log likelihood',alpha=1)
    ax.set_title('group inverse')
    im=ax.contour(X,Y,subZ, alpha=1,linestyles='solid', label='inferred group', cmap='autumn')
    plt.colorbar(im)

    plt.tight_layout()

main()


print('''
samping from the likelihood surface.
the likelihood surface suggest a probability distribution.
''')

# test resampling
mu=torch.tensor([0.,0])
cov=torch.tensor([[1.,0],[0,1]])
distribution=MultivariateNormal(mu,cov)
rawsamples=[]
for _ in range(300):
    sample=distribution.sample()
    rawsamples.append(sample.tolist())
raw=np.array(rawsamples)
from scipy.signal import resample
new=resample(rawsamples,1200)



plt.scatter(raw[:,0], raw[:,1], alpha=0.3)
plt.scatter(new[:,0], new[:,1],alpha=0.1)







numsamples=100
adjustratio=len(invres['h'])/len(invres['a'])
alltag=[]
allsamples=[]
for theta,cov,_ in invres['a']:
    distribution=MultivariateNormal(theta.view(-1),cov*0.01)
    samples=[]
    while len(samples)<int(numsamples*adjustratio):
        a=distribution.sample()
        if torch.all(a.clamp(0,2)==a):
            samples.append(a)
    allsamples.append(torch.stack(samples))
    alltag+=[1]*int(numsamples*adjustratio)
for theta,cov,_ in invres['h']:
    distribution=MultivariateNormal(theta.view(-1),cov*0.01)
    samples=[]
    while len(samples)<numsamples:
        a=distribution.sample()
        if torch.all(a.clamp(0,2)==a):
            samples.append(a)
    allsamples.append(torch.stack(samples))
    alltag+=[0]*numsamples

allsamples=np.array(torch.cat(allsamples,axis=0))
alltag=np.array(alltag).astype('int')
X, Y=allsamples,alltag
X = X[np.logical_or(Y==0,Y==1)][:,:8]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)
f_importances(np.abs(clf.coef_[0]),theta_names)
plt.show()

# adding auc, roc
y_test_pred = clf.decision_function(X) 
test_fpr, test_tpr, te_thresholds = roc_curve(Y, y_test_pred)
plt.grid()
plt.plot(test_fpr, test_tpr, label=" AUC TEST ="+str(auc(test_fpr, test_tpr)))
plt.plot([0,1],[0,1],'g--')
plt.legend()
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC(ROC curve)")
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.show()



axis=list(np.argsort(np.abs(clf.coef_))[0])[::-1][:2]
axis=[1,0]
print('top parameter index',axis)
y = lambda x: (-clf.intercept_[0]-clf.coef_[0][axis[0]]*x ) / clf.coef_[0][axis[1]]
x=  np.linspace(0,2,30)

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.scatter(X[Y==0,axis[0]], X[Y==0,axis[1]],color='b',label='health control',alpha=0.01)
ax.scatter(X[Y==1,axis[0]], X[Y==1,axis[1]],color='r',label='ASD',alpha=0.01)
ax.plot(x, y(x))
ax.set_xlim(0,2); ax.set_xlabel(theta_names_[axis[0]])
ax.set_ylim(0,2); ax.set_ylabel(theta_names_[axis[1]])
quickleg(ax)
plt.show()

print('''
project the individual thetas on to the normal vector.
''')
w=clf.coef_[0]
ticks=X[:,:8].dot(w)
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.hist(ticks[Y==0],density=True,color='b',bins=22,label='health control',alpha=0.6)
ax.hist(ticks[Y==1],density=True,color='r',bins=22,label='ASD',alpha=0.6)
quickleg(ax)
quickspine(ax)
ax.set_xlabel('param value')
ax.set_ylabel('probability')
# quicksave('asd group project svm normal vector no init uncertainty')




for isub in range(numasub):
    dataname="asub{}".format(str(isub))
    savename=Path("Z:/human/{}".format(foldername))/"invasub{}".format(str(isub))

with open(savename, 'rb') as f:
    res=pickle.load(f)


res[1]
log=[r[2] for r in res]
logg=[r[1] for r in log]

# sample from unkown distribution
x=np.array([r[0] for r in logg])
p=np.exp(np.array([r[1]*-1 for r in logg]))
im=plt.scatter(x[:,4],x[:,3], c=p)
plt.colorbar(im)


indls=np.argsort(p)[::-1]
numtokeep=len(p)//4
largep=p[indls][:numtokeep]
largex=x[indls][:numtokeep]
im=plt.scatter(largex[:,4],largex[:,3], c=largep)
plt.colorbar(im)
plt.show()
im=plt.scatter(x[:,4],x[:,3], c=p)
plt.colorbar(im)


x,y=np.meshgrid(np.linspace(0,1,10),np.linspace(0,1,10))
z=np.random.random(size=(10,10))
plt.scatter(x,y,c=z)

plt.tricontourf(np.linspace(0,1,10),np.linspace(0,1,10),z)



x[:,0][0]
p.shape
# get more sample

'''
get more xs, solve for p
still unkonw distibution
how to sample x


'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

# ----------------------------------------------------------------------
# Plot the progression of histograms to kernels
np.random.seed(1)
N = 20
# X = np.concatenate(
#     (np.random.normal(0.2, 0.2, int(0.5 * N)), np.random.normal(0.5, 0.1, int(0.5 * N)))
# )[:, np.newaxis]

X=x[:,9].reshape(-1,1)

X_plot = np.linspace(-1, 3, 1000)[:, np.newaxis]
bins = np.linspace(-1, 3, 10)

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# histogram 1
ax[0, 0].hist(X[:, 0], bins=bins, fc="#AAAAFF", density=True)
ax[0, 0].text(-3.5, 0.31, "Histogram")

# hist 2
ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc="#AAAAFF", density=True)
ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")


# tophat KDE
kde = KernelDensity(kernel="tophat", bandwidth=0.1).fit(X)
log_dens = kde.score_samples(X_plot)
ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc="#AAAAFF")
ax[1, 0].text(-3.5, 0.31, "Tophat Kernel Density")

# Gaussian KDE
kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(X)
log_dens = kde.score_samples(X_plot)
ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc="#AAAAFF")
ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")


import numpy as np
import matplotlib.pyplot as plt
import warnings

from numpy.linalg.linalg import svd
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import torch
import heapq
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt
import time
torch.manual_seed(42)
from numpy import pi
from InverseFuncs import *
from monkey_functions import *
from pathlib import Path
from plot_ult import *

# loading
import pickle
data_path=Path("Z:\\schro_normal")
with open(data_path/'packed', 'rb') as f:
    log = pickle.load(f)

with open('cmafull_vic2', 'rb') as f:
    log = pickle.load(f)

# theta hist
optimizer=log[-1][0]
cov=optimizer._C
theta=torch.tensor(optimizer._mean).view(-1,1)
finaltheta=theta
cov=optimizer._C
finalcov=torch.tensor(cov)
theta_bar(finaltheta,finalcov)

# theta confidence hist
thetaconfhist(finalcov)


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
    for point in r:
        alltheta.append(torch.tensor(point[0]))
        loglls.append(torch.tensor(point[1]))
alltheta=torch.stack(alltheta)
loglls=torch.tensor(loglls).flatten()

projectedparam=torch.pca_lowrank(torch.tensor(alltheta),2)
projectedparam=projectedparam[0]
# pc space scatter logll
s = plt.scatter(projectedparam[:,0], projectedparam[:,1], c=loglls,alpha=0.3,edgecolors=None, cmap='jet')
plt.clim(min(loglls), max(loglls)) 
plt.xlabel('projected parameters')
plt.ylabel('projected parameters')
c = plt.colorbar()





np.set_printoptions(precision=2)
print((optimizer._mean))

optimizer=log[-1][0]
cov=optimizer._C
theta=torch.tensor(optimizer._mean).view(-1,1)
finaltheta=theta
cov=optimizer._C
finalcov=torch.tensor(cov)


# covariance heatmap
inds=[1,3,5,8,0,2,4,7,6,9,10]
with initiate_plot(5,5,300) as fig:
    ax=fig.add_subplot(1,1,1)
    cov=torch.tensor(cov)
    im=plt.imshow(cov[inds].t()[inds].t(),cmap=plt.get_cmap('bwr'),
        vmin=-torch.max(cov),vmax=torch.max(cov))
    ax.set_title('covariance matrix', fontsize=20)
    x_pos = np.arange(len(theta_names))
    plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
    plt.xticks(x_pos, [theta_names[i] for i in inds],rotation=45,ha='right')
    add_colorbar(im)

# correlation heatmap
b=torch.diag(torch.tensor(cov),0)
S=torch.diag(torch.sqrt(b))
Sinv=torch.inverse(S)
correlation=Sinv@cov@Sinv
with initiate_plot(5,5,300) as fig:
    ax=fig.add_subplot(1,1,1)
    im=ax.imshow(correlation[inds].t()[inds].t(),cmap=plt.get_cmap('bwr'),
        vmin=-torch.max(correlation),vmax=torch.max(correlation))
    ax.set_title('correlation matrix', fontsize=20)
    x_pos = np.arange(len(theta_names))
    plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
    plt.xticks(x_pos, [theta_names[i] for i in inds],rotation=45,ha='right')
    add_colorbar(im)


# eig cov heatmap
ev, evector=torch.eig(torch.tensor(cov),eigenvectors=True)
ev=ev[:,0]
ev,esortinds=ev.sort(descending=False)
evector=evector[esortinds]
with initiate_plot(5,5,300) as fig:
    ax=fig.add_subplot(1,1,1)
    img=ax.imshow(evector[:,inds].t(),cmap=plt.get_cmap('bwr'),
            vmin=-torch.max(evector),vmax=torch.max(evector))
    add_colorbar(img)
    ax.set_title('eigen vectors of Hessian')
    x_pos = np.arange(len(theta_names))
    plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')


with initiate_plot(5,1,300) as fig:
    ax=fig.add_subplot(1,1,1)
    x_pos = np.arange(len(theta_names))
    # Create bars and choose color
    ax.bar(x_pos, ev, color = 'blue')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yscale('log')
    # ax.set_ylim(min(plotdata),max(plotdata))
    ax.set_yticks([0.1,100])
    ax.set_xlabel('eigen values, log scale')
    plt.tick_params(axis='y', which='minor')
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    plt.gca().invert_yaxis()




# compare 2 monkeys
data_path=Path("Z:\\schro_pert")
with open(data_path/'longonlypacked_schro_pert', 'rb') as f:
    slog = pickle.load(f)

data_path=Path("Z:\\bruno_pert")
with open(data_path/'cmafull_packed_bruno_pert', 'rb') as f:
    blog = pickle.load(f)

# remove goal radius param
theta_names=theta_names[:6]+theta_names[-4:]
theta_mean=theta_mean[:6]+theta_mean[-4:]

# two monkey theta hist and confidence hist
log=slog
finalcov=torch.tensor(log[-1][0]._C)
finaltheta=torch.tensor(log[-1][0]._mean).view(-1,1)
finaltheta=torch.cat([finaltheta[:6],finaltheta[-4:]])
finalcov = finalcov[torch.arange(finalcov.size(0))!=6] 
finalcov = finalcov[:,torch.arange(finalcov.size(1))!=6] 
theta_bar(finaltheta,finalcov)
thetaconfhist(finalcov)

log=blog
finalcov=torch.tensor(log[-1][0]._C)
finaltheta=torch.tensor(log[-1][0]._mean).view(-1,1)
finaltheta=torch.cat([finaltheta[:6],finaltheta[-4:]])
finalcov = finalcov[torch.arange(finalcov.size(0))!=6] 
finalcov = finalcov[:,torch.arange(finalcov.size(1))!=6] 
theta_bar(finaltheta,finalcov)
thetaconfhist(finalcov)

# totegher hist
x = np.random.normal(1, 2, 99)
y = np.random.normal(-1, 3, 99)
bins = np.linspace(-10, 10, 11)

plt.hist([t1.flatten(), t2.flatten()], label=['x', 'y'])

plt.bar([i for i in range(len(finaltheta))], finaltheta,yerr=torch.diag(finalcov)**0.5,color = 'tab:blue')

plt.legend(loc='upper right')
plt.show()




a=theta_bar(t1,c1, label='schro',width=0.5,shift=-0.2)
theta_bar(t2,c2,ax=a, label='bruno',width=0.5,shift=0.2)
a.get_figure()






# density
densities=[0.0001, 0.0005, 0.001, 0.005]
filenamecommom=Path('Z:\\schro_normal')
logs=[None]*4
for i, density in enumerate(densities):
    try:
        with open(filenamecommom/('d'+str(i+1)+'_packed'), 'rb') as f:
            log = pickle.load(f)
        logs[i]=log
    except:
        continue
filenamecommom=Path('Z:\\bruno_normal')
logs=[None]*4
for i, density in enumerate(densities):
    try:
        with open(filenamecommom/('d'+str(i+1)), 'rb') as f:
            log = pickle.load(f)
        logs[i]=log
    except:
        continue

xs,ys,errs=d_process(logs)
d_noise()
d_obs_degree()


gauss_div(ys[0,2],errs[0,2],ys[0,4],errs[0,4])
gauss_div(ys[1,2],errs[1,2],ys[1,4],errs[1,4])

gauss_div(0,1,1,2)



# uncertainty growth rate
log=slog
sc=torch.tensor(log[-1][0]._C)
st=torch.tensor(log[-1][0]._mean).view(-1,1)
sv=st[2]**2*st[4]**2/(st[2]**2+st[4]**2)
sw=st[3]**2*st[5]**2/(st[3]**2+st[5]**2)


def isum(a,b):
    return a*b/a+b

isum(sc[2]**2,sc[4]**2)

log=blog
bc=torch.tensor(log[-1][0]._C)
bt=torch.tensor(log[-1][0]._mean).view(-1,1)
bv=bt[2]**2*bt[4]**2/(bt[2]**2+bt[4]**2)
bw=bt[3]**2*bt[5]**2/(bt[3]**2+bt[5]**2)



trialtypes=['pert','non pert']
a=barpertacc([63.36783413567739,55.58846184159118],trialtypes,label='schro',shift=-0.2)
barpertacc([57.37014305912993,43.945449613900934],trialtypes,label='bruno',shift=0.2,ax=a)
a.get_figure()


slog[-1][0]._C
slog[-1][1]
np.array(slog[-1][1]).transpose()
np.cov(np.array(slog[-1][1]).transpose())
plt.imshow(np.cov(np.array(slog[-1][1]).transpose()))
plt.imshow(slog[-1][0]._C)
[i**0.5 for i in np.diag(slog[-1][0]._C)]
[i**0.5 for i in np.diag(np.cov(np.array(blog[-1][1]).transpose()))]





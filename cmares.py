# plotting of immedite inverse result, using cmaes.

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
torch.manual_seed(42)
from FireflyEnv import ffacc_real
from Config import Config
np.set_printoptions(precision=3)


# load model
arg = Config()
env=ffacc_real.FireFlyPaper(arg)
env.debug=True
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
agent_=TD3.load('trained_agent/re1re1repaper_3_199_198.zip')
# agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()

# loading data
data_path=Path("Z:\\schro_normal")
with open(data_path/'packed', 'rb') as f:
    df = pickle.load(f)
print('process data')
states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
print('done process data')





# bar plots of inferred theta -----------------------------------------------

with open('Z:\\q_normal/cmad1_packed', 'rb') as f:
    log = pickle.load(f)

# theta hist fig
optimizer=log[-1][0]
cov=optimizer._C
theta=torch.tensor(optimizer._mean).view(-1,1)
finaltheta=theta
cov=optimizer._C
finalcov=torch.tensor(cov)
theta_bar(finaltheta,finalcov)

# confidence hist fig
thetaconfhist(finalcov)

# density obs fig
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
# plt.plot(xs,ys[:,2:4])
# plt.plot(xs,ys[:,4:6])
# plt.plot(xs,ys[:,2:4]/ys[:,4:6])
# plt.plot(xs,ys[:,6])
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(xs,ys[:,2]/ys[:,4])
ax.set_xlabel('optical flow density',fontsize=12)
ax.set_ylabel('observation reliable degree',fontsize=12)
ax.set_title('forward observation noise vs optical flow density', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)



# trend plots of inferred theta ---------------------------------------------------

# log ll vs gen fig
optimizer=log[-1][0]
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


# converge and heatmaps plot ---------------------------------------------------

# pca space of all samples
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

mu=torch.mean(projectedparamall,axis=0)
aroundsolution=allsamples[:len(allsamples)//2]
aroundsolution.sort(key=lambda x: x[0])
alltheta=np.vstack([x[1] for x in aroundsolution])
loglls=[x[0] for x in aroundsolution]
pccov=transition.T@torch.tensor(np.cov(alltheta.T)).float()@transition

with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(111)
    plot_cov_ellipse(pccov, mu, alpha_factor=1,nstd=1,ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    s = ax.scatter(projectedparamall[:,0], projectedparamall[:,1], c=(logllsall),alpha=0.5,edgecolors='None', cmap='jet')
    ax.set_xlabel('projected parameters')
    ax.set_ylabel('projected parameters')
    c = fig.colorbar(s, ax=ax)
    ax.locator_params(nbins=3, axis='y')
    ax.locator_params(nbins=3, axis='x')
    # c.clim(min(np.log(loglls)), max(np.log(loglls))) 
    c.set_label('- log likelihood')
    # c.set_ticks([min((loglls)),max((loglls))])
    c.ax.locator_params(nbins=4)



# conditional uncertainty
# case 0, use math
# two monkey theta hist and confidence hist
# compare 2 monkeys
data_path=Path("Z:\\schro_pert")
with open(data_path/'longonlypacked_schro_pert', 'rb') as f:
    slog = pickle.load(f)

data_path=Path("Z:\\bruno_pert")
with open(data_path/'cmafull_packed_bruno_pert', 'rb') as f:
    blog = pickle.load(f)
log=slog
finalcov=torch.tensor(log[-1][0]._C)
finaltheta=torch.tensor(log[-1][0]._mean).view(-1,1)
st=torch.cat([finaltheta[:6],finaltheta[-4:]])
finalcov = finalcov[torch.arange(finalcov.size(0))!=6] 
sc = finalcov[:,torch.arange(finalcov.size(1))!=6] 

log=blog
finalcov=torch.tensor(log[-1][0]._C)
finaltheta=torch.tensor(log[-1][0]._mean).view(-1,1)
bt=torch.cat([finaltheta[:6],finaltheta[-4:]])
finalcov = finalcov[torch.arange(finalcov.size(0))!=6] 
bc = finalcov[:,torch.arange(finalcov.size(1))!=6] 

# paramindex=0
# covyy=np.diag(cov)[paramindex]
# covxy=np.concatenate((np.diag(cov)[:paramindex], np.diag(cov)[paramindex+1:]))
# fullcov=np.asfarray(cov)
# temp = fullcov[np.arange(fullcov.shape[0])!=paramindex] 
# covxx = temp[:,np.arange(temp.shape[1])!=paramindex] 
# conditional_cov(covyy,covxx, covxy)

plt.bar(list(range(sc.shape[0])),np.sqrt(np.array([conditional_cov_block(sc, paramindex) for paramindex in range(sc.shape[0])])))
plt.bar(list(range(sc.shape[0])),np.sqrt(np.diag(sc)))

conderr=np.sqrt(np.array([conditional_cov_block(sc, paramindex) for paramindex in range(sc.shape[0])]))
stderr=np.sqrt(np.diag(sc))
twodatabar(conderr,stderr,labels=['conditional std','marginal std'],ylabel='param value')

theta_bar(st.view(-1),err=conderr)
theta_bar(st,sc )



corr=correlation_from_covariance(cov)

# covariance heatmap
inds=[1, 3, 5, 7, 0, 2, 4,6, 8, 9]
theta_names=theta_names[:6]+theta_names[-4:]
theta_mean=theta_mean[:6]+theta_mean[-4:]

with initiate_plot(5,5,300) as fig:
    ax=fig.add_subplot(1,1,1)
    cov=torch.tensor(cov)
    im=plt.imshow(cov[inds].t()[inds].t(),cmap=plt.get_cmap('bwr'),
        vmin=-torch.max(cov),vmax=torch.max(cov))
    ax.set_title('covariance matrix', fontsize=20)
    c=plt.colorbar(im,fraction=0.046, pad=0.04)
    c.set_label('covariance')
    x_pos = np.arange(len(theta_names))
    plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
    plt.xticks(x_pos, [theta_names[i] for i in inds],rotation=45,ha='right')
    

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
    c=plt.colorbar(im,fraction=0.046, pad=0.04)
    c.set_label('correlation')
    x_pos = np.arange(len(theta_names))
    plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
    # plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
    plt.xticks(x_pos, [theta_names[i] for i in inds],rotation=45,ha='right')


# eig cov heatmap
ev, evector=torch.eig(torch.tensor(cov),eigenvectors=True)
ev=ev[:,0]
ev,esortinds=ev.sort(descending=False)
evector=evector[esortinds]
with initiate_plot(5,5,300) as fig:
    ax=fig.add_subplot(1,1,1)
    img=ax.imshow(evector[:,inds].t(),cmap=plt.get_cmap('bwr'),
            vmin=-torch.max(evector),vmax=torch.max(evector))
    c=plt.colorbar(img,fraction=0.046, pad=0.04)
    c.set_label('parameter weight')
    ax.set_title('eigen vectors of covariance matrix')
    x_pos = np.arange(len(theta_names))
    plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
    ax.set_xticks([])

with initiate_plot(5,1,300) as fig:
    ax=fig.add_subplot(1,1,1)
    x_pos = np.arange(len(theta_names))
    # Create bars and choose color
    ax.bar(x_pos, torch.sqrt(ev), color = 'blue')
    for i, v in enumerate(torch.sqrt(ev)):
        ax.text( i-0.4,v+0.2 , '{:.2f}'.format(v.item()), color='blue')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    # ax.set_yscale('log')
    # ax.set_ylim(min(plotdata),max(plotdata))
    # ax.set_yticks([0.1,100])
    ax.set_xlabel('sqrt of eigen values')
    # plt.tick_params(axis='y', which='minor')
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    plt.gca().invert_yaxis()


# compare 2 monkeys --------------------------------------------------------------

with open(Path("Z:\\schro_pert")/'longonlypacked_schro_pert', 'rb') as f:
    slog = pickle.load(f)
with open(Path("Z:\\bruno_pert")/'cmafull_packed_bruno_pert', 'rb') as f:
    blog = pickle.load(f)

# remove goal radius param
theta_names=theta_names[:6]+theta_names[-4:]
theta_mean=theta_mean[:6]+theta_mean[-4:]

# two monkey theta hist and confidence hist
log=slog
finalcov=torch.tensor(log[-1][0]._C)
finaltheta=torch.tensor(log[-1][0]._mean).view(-1,1)
st=torch.cat([finaltheta[:6],finaltheta[-4:]])
finalcov = finalcov[torch.arange(finalcov.size(0))!=6] 
sc = finalcov[:,torch.arange(finalcov.size(1))!=6] 
theta_bar(st,sc)
# thetaconfhist(sc)

log=blog
finalcov=torch.tensor(log[-1][0]._C)
finaltheta=torch.tensor(log[-1][0]._mean).view(-1,1)
bt=torch.cat([finaltheta[:6],finaltheta[-4:]])
finalcov = finalcov[torch.arange(finalcov.size(0))!=6] 
bc = finalcov[:,torch.arange(finalcov.size(1))!=6] 
theta_bar(bt,bc)
# thetaconfhist(bc)

# totegher hist
yerr=torch.diag(finalcov)**0.5 # using std
plt.hist([st.flatten(), bt.flatten()], label=['x', 'y'])
# plt.bar([i for i in range(len(finaltheta))], finaltheta,yerr=yerr,color = 'tab:blue')
plt.legend(loc='upper right')
plt.show()

# err using std
# bt=torch.tensor(blog[-1][0]._mean)
# st=torch.tensor(slog[-1][0]._mean)
# bc=torch.tensor(np.sqrt(np.diag(blog[-1][0]._C))).float()
# sc=torch.tensor(np.sqrt(np.diag(slog[-1][0]._C))).float()

# err using confidence interval of the range
sci=get_ci(slog) # check here to use the range of stablized logll
sci=np.delete(sci,(6),axis=1)

bci=get_ci(blog)
bci=np.delete(bci,(6),axis=1)

a=theta_bar(st,sc, label='schro',width=0.5,shift=-0.2, err=sci)
theta_bar(bt,bc,ax=a, label='bruno',width=0.5,shift=0.2, err=bci)
a.get_figure()


# density vs observation noise ---------------------------------------------------------------
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

xs,ys,errs=d_process(logs,densities)
d_noise(xs,ys,errs)
d_kalman_gain(xs,ys,errs)



# uncertainty growth rate
log=slog
sc=torch.tensor(log[-1][0]._C)
st=torch.tensor(log[-1][0]._mean).view(-1,1)
sv=st[2]**2*st[4]**2/(st[2]**2+st[4]**2)
sw=st[3]**2*st[5]**2/(st[3]**2+st[5]**2)

log=blog
bc=torch.tensor(log[-1][0]._C)
bt=torch.tensor(log[-1][0]._mean).view(-1,1)
bv=bt[2]**2*bt[4]**2/(bt[2]**2+bt[4]**2)
bw=bt[3]**2*bt[5]**2/(bt[3]**2+bt[5]**2)




slog[-1][0]._C
slog[-1][1]
np.array(slog[-1][1]).transpose()
np.cov(np.array(slog[-1][1]).transpose())
plt.imshow(np.cov(np.array(slog[-1][1]).transpose()))
plt.imshow(slog[-1][0]._C)
[i**0.5 for i in np.diag(slog[-1][0]._C)]
[i**0.5 for i in np.diag(np.cov(np.array(blog[-1][1]).transpose()))]





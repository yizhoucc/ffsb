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
from numpy import pi, std
from InverseFuncs import *
from monkey_functions import *
from pathlib import Path
from plot_ult import *
from stable_baselines3 import TD3
import pandas as pd
import seaborn as sns
torch.manual_seed(42)

np.set_printoptions(precision=3)


folder=Path("Z://")
invres = sorted(list(folder.rglob("bruno_normal/fixre*")), key=os.path.getmtime,reverse=True)

names,thetas, covs, errs=[],[],[],[]
for inv in invres:
    finaltheta,finalcov,err=process_inv(inv)
    thetas.append(finaltheta)
    covs.append(finalcov)
    errs.append(err)
    names.append(str(inv))
names=[0.0001, 0.0005, 0.001, 0.005]
names=['density = {}'.format(str(n)) for n in names]
ax=multimonkeyobs(names,thetas,covs, errs)
ax.get_figure()


for inv in invres:
    finaltheta,finalcov,err=process_inv(inv)
    print(inv, finaltheta)
    theta_bar(finaltheta,finalcov,err=err)
    # theta_bar(finaltheta,finalcov)

# bar plots of inferred theta -----------------------------------------------

# theta hist fig
finaltheta,finalcov,err=process_inv('Z:\\q_normal/cmad1_packed')
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
with open('Z:/simulation/invbliefsimulationpert100','rb') as f:
    log=pickle.load(f)
path=Path('Z:/simulation')
datafile=path/'simulation500'
with open(datafile, 'rb') as f:
    _, _, _, groundtruth = pickle.load(f)
# pca space of all samples ---------------------------------------------------
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

allthetamu=np.mean(0,alltheta)
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

with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(111)
    plot_cov_ellipse(pccov, mu, alpha=1,nstd=1,ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    scatters = ax.scatter(projectedparamall[:,0], projectedparamall[:,1], c=(logllsall),alpha=0.5,edgecolors='None', cmap='jet')
    ax.set_xlabel('projected parameters')
    ax.set_ylabel('projected parameters')
    c = fig.colorbar(scatters, ax=ax)
    ax.locator_params(nbins=3, axis='y')
    ax.locator_params(nbins=3, axis='x')
    # c.clim(min(np.log(loglls)), max(np.log(loglls))) 
    c.set_label('- log likelihood')
    # c.set_ticks([min((loglls)),max((loglls))])
    c.ax.locator_params(nbins=4)
    # plot ground truth for simulation
    try:
        groundtruthpc=(groundtruth-allthetamu)@np.array(v)@np.linalg.inv(np.diag(s))
        ax.scatter(groundtruthpc[0],groundtruthpc[1],color='k')
    except NameError:
        pass




pcaedxy=(groundtruth-torch.mean(torch.tensor(alltheta),0))@transition

pcaedxy@torch.inverse(transition).T+torch.mean(torch.tensor(alltheta))

# conditional uncertainty -------------------------------------------------------
# case 0, use math
# two monkey theta hist and confidence hist
# compare 2 monkeys
data_path=Path("Z:\\schro_pert")
with open(data_path/'longonlypacked_schro_pert', 'rb') as f:
    slog = pickle.load(f)
log=slog
finalcov=torch.tensor(log[-1][0]._C)
finaltheta=torch.tensor(log[-1][0]._mean).view(-1,1)
st=torch.cat([finaltheta[:6],finaltheta[-4:]])
finalcov = finalcov[torch.arange(finalcov.size(0))!=6] 
sc = finalcov[:,torch.arange(finalcov.size(1))!=6] 


data_path=Path("Z:\\bruno_pert")
with open(data_path/'cmafull_packed_bruno_pert', 'rb') as f:
    blog = pickle.load(f)
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


# covariance heatmap -----------------------------------------------------
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
    

# correlation heatmap -----------------------------------------------------
inds=[1, 3, 5, 7, 0, 2, 4,6, 8, 9]
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


# eig cov heatmap -----------------------------------------------------
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
    ax.bar(x_pos, torch.sqrt(ev), color = color_settings['hidden'])
    for i, v in enumerate(torch.sqrt(ev)):
        ax.text( i-0.4,v+0.2 , '{:.2f}'.format(v.item()), color=color_settings['hidden'])
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
    # plt.gca().invert_yaxis()


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

ax=theta_bar(st,sc, label='schro',width=0.5,shift=-0.2, err=sci)
theta_bar(bt,bc,ax=ax, label='bruno',width=0.5,shift=0.2, err=bci)
ax.get_figure()


# plot multi monkey theta in one same hist plot------------------------------------------------
folder=Path("Z:/human/")
invres = sorted(list(folder.glob("*")), key=os.path.getmtime,reverse=True)
logls=['Z:/human/subinvsidehgroup','Z:/human/paperhgroup']
monkeynames=['side', 'straight' ]

logls=['Z:/human/fixragroup','Z:/human/clusterpaperhgroup']
monkeynames=['ASD', 'Ctrl' ]

logls=[
    Path('Z:/bruno_pert/cmafull_b_pert'),
    Path('Z:/schro_pert/cmafull_packed_schro_pert'),
    Path('Z:/victor_pert/cmafull_victor_pert_ds'),
    Path('Z:/q_pert/cma180paper_packed'),
]
monkeynames=['bruno', 'schro', 'victor', 'quigley']

mus,covs,errs=[],[],[]
for inv in logls:
    finaltheta,finalcov, err=process_inv(inv,ind=60)
    mus.append(finaltheta)
    covs.append(finalcov)
    errs.append(err)

ax=multimonkeytheta(monkeynames, mus, covs, errs, )
ax.set_yticks([0,1,2])
ax.plot(np.linspace(-1,9),[2]*50)
ax.get_figure()




# multi monkey first eig vector
logls=[
    Path('Z:/bruno_pert/cmafull_b_pert'),
    Path('Z:/schro_pert/cmafull_packed_schro_pert'),
    Path('Z:/victor_pert/cmafull_victor_pert_ds'),
    Path('Z:/q_pert/cma180paper_packed'),
]
monkeynames=['bruno', 'schro', 'victor', 'quigley']

covs, evs, evectors=[],[],[]
for inv in logls:
    _,finalcov, _=process_inv(inv)
    covs.append(finalcov)
    ev, evector=torch.eig(torch.tensor(finalcov),eigenvectors=True)
    ev=ev[:,0]
    ev,esortinds=ev.sort(descending=False)
    evector=evector[:,esortinds]
    firstevector=evector[:,0].float().view(-1,1)
    firstev=ev[0]
    firstevector=firstevector*torch.sign(torch.min(firstevector)+torch.max(firstevector))
    evs.append(firstev) # make the largest component alaways positive for visualization
    evectors.append(firstevector/torch.sum(torch.abs(firstevector)))
ax=multimonkeyeig(monkeynames, evectors )
ax.set_title('mattered most parameter for each monkey')
ax.set_yticks([0.5,0,-0.2])
ax.set_ylabel('importantce ratio')
ax.get_figure()




# density vs observation noise ---------------------------------------------------------------
# density

logls = sorted(list(folder.rglob("q_normal/*re*")), key=os.path.getmtime,reverse=True)

densities=[0.0001, 0.0005, 0.001, 0.005]
logs=[]
for i, filename in enumerate(logls):
    with open(filename, 'rb') as f:
        log = pickle.load(f)
    logs.append(log)

xs,ys,errs=d_process(logs,densities)
d_noise(xs,ys,errs)
d_kalman_gain(xs,ys,errs)



# plot multi density theta in one same hist plot
logls = sorted(list(folder.rglob("bruno_normal/d*")),reverse=False)
densities=[0.0001, 0.0005, 0.001, 0.005]
mus,covs,errs=[],[],[]
for inv in logls:
    finaltheta,finalcov, err=process_inv(inv)
    mus.append(finaltheta)
    covs.append(finalcov)
    errs.append(err)

ax=multimonkeytheta(densities, mus, covs, errs, )
ax.set_yticks([0,1,2])
ax.get_figure()



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




# validate the inverse on test set (recacluate the log likelihood)
def getlogll(x):
    with torch.no_grad():
        return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=0.01,num_iteration=1, states=states, samples=5,gpu=False).item()

agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()
env=ffacc_real.FireFlyPaper(arg)

# testing partial fix (allow process change) vs harder fix (only allow obs change)
print('loading data')
datapath=Path("Z:\\q_normal\\packed")
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df=df[df.target_r>180]
df=df[df.floor_density==0.0001]
# floor density are in [0.0001, 0.0005, 0.001, 0.005]
# q monkey density are in [0.000001, 0.0001, 0.001, 0.005]
df=df[-100:]
print('process data')
states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
print('done process data')

# density 1
with open('Z:/q_normal/re1packed', 'rb') as f: # 10.915373802185059
    log = pickle.load(f)
finaltheta=torch.tensor(log[-1][0]._mean).view(1,-1)
getlogll(finaltheta)


with open('Z:/q_normal/1re1packed', 'rb') as f: # 9.958659172058105
    log = pickle.load(f)
finaltheta=torch.tensor(log[-1][0]._mean).view(1,-1)
getlogll(finaltheta)

# density 2
with open('Z:/q_normal/re2packed', 'rb') as f: # 9.958659172058105
    log = pickle.load(f)
finaltheta=torch.tensor(log[-1][0]._mean).view(1,-1)
getlogll(finaltheta)


with open('Z:/q_normal/1re2packed', 'rb') as f: # 9.655684471130371
    log = pickle.load(f)
finaltheta=torch.tensor(log[-1][0]._mean).view(1,-1)
getlogll(finaltheta)


# testing partial fix (allow process obs change) or unfix all fit performance
print('loading data')
datapath=Path("Z:\\bruno_normal\\packed")
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df=df[df.target_r>180]
df=df[df.floor_density==0.0001]
# floor density are in [0.0001, 0.0005, 0.001, 0.005]
# q monkey density are in [0.000001, 0.0001, 0.001, 0.005]
df=df[-100:]
print('process data')
states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
print('done process data')

# density 1
with open('Z:/bruno_normal/d1', 'rb') as f: # 10.915373802185059
    log = pickle.load(f)
finaltheta=torch.tensor(log[-1][0]._mean).view(1,-1)
getlogll(finaltheta)


with open('Z:/bruno_normal/re1packed', 'rb') as f: # 9.958659172058105
    log = pickle.load(f)
finaltheta=torch.tensor(log[-1][0]._mean).view(1,-1)
getlogll(finaltheta)

# density 2
with open('Z:/q_normal/re2packed', 'rb') as f: # 9.958659172058105
    log = pickle.load(f)
finaltheta=torch.tensor(log[-1][0]._mean).view(1,-1)
getlogll(finaltheta)


with open('Z:/q_normal/1re2packed', 'rb') as f: # 9.655684471130371
    log = pickle.load(f)
finaltheta=torch.tensor(log[-1][0]._mean).view(1,-1)
getlogll(finaltheta)






# distribution of likelihood of each desnity-------------------------------------


with open('brunoobsdenslog500','rb') as f:
    obsvls,p = pickle.load(f)


# seperate
with initiate_plot(9,3) as fig:
    for i in range(len(densities)):
        if i!=0:
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax=fig.add_subplot(1,4,i+1, sharey=ax)
        else:
            ax=fig.add_subplot(1,4,1)
            ax.set_ylabel('probability')
        ax.set_xlim(0,max(obsvls))
        ax.set_xticks([0,0.1,0.2,0.5])
        pdd,d=p[i], densities[i]
        ax.fill_between(obsvls,pdd, label='density {}'.format(str(d)),color=colors[i],alpha=0.8,edgecolor='none')
        ax.set_xlabel('observation noise std [m/s]')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0,np.max(p)*1.2)
        ax.legend()
        
# together
with initiate_plot(3,3) as fig:
    ax=fig.add_subplot(1,1,1)
    for i in range(len(densities)):
        pdd,d=p[i], densities[i]
        # ax.fill_between(obsvls,pdd, label='density {}'.format(str(d)),color=colors[i],alpha=1/len(densities),edgecolor='none')
        ax.plot(obsvls,pdd, label='density {}'.format(str(d)),color=colors[i],alpha=1)

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel('probability')
    ax.set_xlim(0,max(obsvls))
    ax.set_xticks([0,0.1,0.2,0.5])
    ax.set_xlabel('observation noise std [m/s]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0,np.max(p)*1.2)
    leg=ax.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

# find the peak (best fit obs value)
obsvls[np.argmax(p,axis=1)]


# liklihood single trial ---------------------------------------------------------------------------------

ind=np.random.randint(0,len(tasks))
res=likelihoodvstime(agent, actions, tasks, phi, theta, env, action_var=0.01, states=states, samples=5,ep=ind)
res=np.concatenate([np.zeros(1),res])
with initiate_plot(4,3) as fig:
    ax=fig.add_subplot(111)
    ax.plot([0.1*t for t in range(len(res)-1)],np.diff(np.log(res))) # change in log ll
    # ax.set_yticks([])
    # ax.set_yticklabels([])
    ax.set_ylabel('chagne in log likelihood')
    # ax.set_xlim(0,max(obsvls))
    # ax.set_xticks([0,0.1,0.2,0.5])
    ax.set_xlabel('time [s]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_ylim(0,np.max(p)*1.2)
    leg=ax.legend()

irc_pert(agent, env, phi, finaltheta,etask=tasks[ind],initv=actions[ind][0][0],initw=actions[ind][0][1],mkactions=actions[ind],pert=pert)


# stop distribution ---------------------------------------------------------------------------------
print('loading data')
datapath=Path("Z:\\victor_normal\\packed")
ith=1 # ith density 
savename=datapath.parent/('preall'+datapath.name)
# savename=datapath.parent/('fullfixre{}'.format(str(ith))+datapath.name)
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df=df[df.target_r>200]
densities=sorted(pd.unique(df.floor_density))
# df=df[df.floor_density==densities[ith]]
print('process data')
states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
print('done process data')
with open(savename, 'rb') as f:
    log = pickle.load(f)
finalcov=torch.tensor(log[-1][0]._C)
finaltheta=torch.tensor(log[-1][0]._mean).view(-1,1)
bt=torch.cat([finaltheta[:6],finaltheta[-4:]])
finalcov = finalcov[torch.arange(finalcov.size(0))!=6] 
bc = finalcov[:,torch.arange(finalcov.size(1))!=6] 
ev, evector=torch.eig(torch.tensor(bc),eigenvectors=True)
ev=ev[:,0]
ev,esortinds=ev.sort(descending=False)
evector=evector[:,esortinds]
firstevector=evector[:,0].float().view(-1,1)
firstev=ev[0]
theta_init=bt-firstevector*firstev*5
theta_init=torch.cat([theta_init[:6],finaltheta[6].view(1,1),theta_init[6:]])
theta_final=bt+firstevector*firstev*5
theta_final=torch.cat([theta_final[:6],finaltheta[6].view(1,1),theta_final[6:]])




# given a position or use a random ind
ind=np.random.randint(0,len(tasks))
thistask=np.array(tasks[ind])
indls=similar_trials2this(tasks,thistask, ntrial=20)

mkstops,ircstops=get_stops(states, tasks, thistask, env, finaltheta, agent)

# plot stop distribution with axis (single theta)
with initiate_plot(3,3) as fig:
    ax=fig.add_subplot(111)
    goal=plt.Circle(np.array(thistask)*2,0.26,color=color_settings['goal'], alpha=0.5,label='target', edgecolor='none')
    ax.add_patch(goal)
    ax.axis('equal')
    ax.set_xlabel('world x [m]')
    ax.set_ylabel('world y [m]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i in range(10):
        mkstops,ircstops=get_stops(states, tasks, thistask, env, finaltheta, agent)
        ax.scatter(ircstops[:,0]*2,ircstops[:,1]*2,color=color_settings['model'], label='model',s=1)
    ax.scatter(mkstops[:,0]*2,mkstops[:,1]*2, color=color_settings['a'], label='monkey')
    ax.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


# plot stop distribution no axis (single theta)
with initiate_plot(3,3) as fig:
    ax=fig.add_subplot(111)
    goal=plt.Circle([0,0],0.13,color=color_settings['goal'], alpha=0.5,label='target', edgecolor='none')
    ax.add_patch(goal)
    ax.axis('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])

    ax.scatter(mkstops[:,0]-thistask[0],mkstops[:,1]-thistask[1], color=color_settings['a'], label='monkey')
    ax.scatter(ircstops[:,0]-thistask[0],ircstops[:,1]-thistask[1],color=color_settings['model'], label='model')
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(np.linspace(0,1/4,2), np.linspace(-0.2, -0.2,2), c='k')
    # ax.plot(np.linspace(-1, -1), np.linspace(0, 1/4), c='k')
    ax.text(0.1, -0.23, s=r'$50cm$', fontsize=fontsize)
    # ax.text(-130, 0, s=r'$50cm$', fontsize=fontsize)
    ax.legend()


# stop distribution changes when varying theta 

ind=np.random.randint(0,len(tasks))
thistask=np.array(tasks[ind])
nplot=5
delta=(theta_final-theta_init)/(nplot-1)
fontsize=12
with initiate_plot(3*nplot,3) as fig:
    for n in range(nplot):
        theta=(n-1)*delta+theta_init
        theta=torch.clamp(theta, 1e-3)
        with suppress():
            mkstops,ircstops=get_stops(states, tasks, thistask, env, theta, agent)
        ax=fig.add_subplot(1,nplot,n+1)
        goal=plt.Circle([0,0],0.13,color=color_settings['goal'], alpha=0.5,label='target', edgecolor='none')
        ax.add_patch(goal)
        ax.axis('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])

        ax.scatter(mkstops[:,0]-thistask[0],mkstops[:,1]-thistask[1], color=color_settings['a'], label='monkey')
        ax.scatter(ircstops[:,0]-thistask[0],ircstops[:,1]-thistask[1],color=color_settings['model'], label='model')
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(np.linspace(0,1/4,2), np.linspace(-0.2, -0.2,2), c='k')
        # ax.plot(np.linspace(-1, -1), np.linspace(0, 1/4), c='k')
        ax.text(0.1, -0.23, s=r'$50cm$', fontsize=fontsize)
        # ax.text(-130, 0, s=r'$50cm$', fontsize=fontsize)
        if n==nplot-1:
            ax.legend()
plt.show()



# stop distribution changes when varying theta
pert=None
ind=np.random.randint(0,len(tasks))
thistask=np.array(tasks[ind])
fontsize=12
nplot=3
delta=(theta_final-theta_init)/(nplot-1)
with initiate_plot(3*nplot,3) as fig:
    for n in range(nplot):
        theta=n*delta+theta_init
        theta=torch.clamp(theta, 1e-3)
        print(theta)
        with suppress():
            ircstopls=[]
            for i in range(10):
                mkstops,ircstops=get_stops(states, tasks, thistask, env, theta, agent,pert=pert)
                ircstopls.append(ircstops)
            ircstopls=np.vstack(ircstopls)
        ax=fig.add_subplot(1,nplot,n+1)
        goal=plt.Circle([0,0],0.13,color=color_settings['goal'], alpha=0.3,label='target', edgecolor='none')
        ax.add_patch(goal)
        ax.axis('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])

        # ax.scatter(ircstops[:,0]-thistask[0],ircstops[:,1]-thistask[1],color=color_settings['model'], label='model',s=5)

        im=sns.kdeplot(ircstopls[:,0]-thistask[0], ircstopls[:,1]-thistask[1],ax=ax,levels=6, thresh=1e-2, fill=True,color=color_settings['model'],cbar=True)

        ax.scatter(mkstops[:,0]-thistask[0],mkstops[:,1]-thistask[1], color=color_settings['a'], label='monkey')




pert=None
ind=np.random.randint(0,len(tasks))
thistask=np.array(tasks[ind])
fontsize=12
nplot=3
delta=(theta_final-theta_init)/(nplot-1)
with initiate_plot(3*nplot,3) as fig:
    ax=fig.add_subplot(111)
    for n in range(nplot):
        theta=n*delta+theta_init
        theta=torch.clamp(theta, 1e-3)
        print(theta)
        with suppress():
            ircstopls=[]
            for i in range(10):
                mkstops,ircstops=get_stops(states, tasks, thistask, env, theta, agent,pert=pert)
                ircstopls.append(ircstops)
            ircstopls=np.vstack(ircstopls)

        goal=plt.Circle([n,0],0.13,color=color_settings['goal'], alpha=0.3,label='target', edgecolor='none')
        ax.add_patch(goal)
        ax.axis('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])

        # ax.scatter(ircstops[:,0]-thistask[0],ircstops[:,1]-thistask[1],color=color_settings['model'], label='model',s=5)

        im=sns.kdeplot(ircstopls[:,0]-thistask[0]+n, ircstopls[:,1]-thistask[1],ax=ax,levels=6, thresh=1e-2, fill=True,color=color_settings['model'],cbar=True)

        ax.scatter(mkstops[:,0]-thistask[0]+n,mkstops[:,1]-thistask[1], color=color_settings['a'], label='monkey')


x = np.random.normal(np.tile(np.random.uniform(15, 35, 10), 1000), 4)
y = np.random.normal(np.tile(np.random.uniform(940, 1000, 10), 1000), 10)

# kdeplot = sns.jointplot(x, y, kind="kde", cbar=True, xlim=[10, 40], ylim=[920, 1020])

kdeplot = sns.kdeplot(ircstopls[:,0]-thistask[0]+n, ircstopls[:,1]-thistask[1],levels=6, thresh=1e-2, fill=True,color=color_settings['model'],cbar=True)

plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
# pos_joint_ax = kdeplot.ax_joint.get_position()
# pos_marg_x_ax = kdeplot.ax_marg_x.get_position()
# kdeplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width,pos_joint_ax.height])
# kdeplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])

# get the current colorbar ticks
# cbar_ticks = kdeplot.axes[-1].get_yticks()
cbar_ticks = kdeplot.get_yticks()
# get the maximum value of the colorbar
# _, cbar_max = kdeplot.fig.axes[-1].get_ylim()
_, cbar_max = kdeplot.get_ylim()

# change the labels (not the ticks themselves) to a percentage
# kdeplot.fig.axes[-1].set_yticklabels([f'{t / cbar_max * 100:.1f} %' for t in cbar_ticks])
kdeplot.set_yticklabels([f'{t / cbar_max * 100:.1f} %' for t in cbar_ticks])

cset = ax.contourf(ircstopls, cmap="viridis")
cbar=f.colorbar(cset)
cbar_ticks = cbar.ax.get_yticks()
# get the maximum value of the colorbar
# _, cbar_max = kdeplot.fig.axes[-1].get_ylim()
_, cbar_max = cbar.ax.get_ylim()
cbar.ax.set_yticklabels([f'{t / cbar_max * 100:.1f} %' for t in cbar_ticks])



plt.show()



from scipy import stats

mean, cov = [0, 2], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, size=50).T

kde = stats.gaussian_kde(ircstopls.T)
xx, yy = np.mgrid[0:1:.01, -0.5:0.5:.01]
density = kde(np.c_[xx.flat, yy.flat].T).reshape(xx.shape)

f, ax = plt.subplots()
cset = ax.contourf(xx, yy, density, cmap="viridis")
cbar=f.colorbar(cset)
cbar_ticks = cbar.ax.get_yticks()
# get the maximum value of the colorbar
# _, cbar_max = kdeplot.fig.axes[-1].get_ylim()
_, cbar_max = cbar.ax.get_ylim()
cbar.ax.set_yticklabels([f'{t / cbar_max * 100:.1f} %' for t in cbar_ticks])




# stop distribution in pert, one trial
ind=np.random.randint(low=0,high=len(df))
assert len(df)==len(tasks)
pert=np.array([df.iloc[ind].perturb_v,df.iloc[ind].perturb_w])
pert=np.array(down_sampling_(pert.T))/400
pert=pert.astype('float32')
thistask=np.array(tasks[ind])
# plot stop distribution with axis
with initiate_plot(3,3) as fig:
    ax=fig.add_subplot(111)
    goal=plt.Circle(np.array(thistask)*2,0.26,color=color_settings['goal'], alpha=0.5,label='target', edgecolor='none')
    ax.add_patch(goal)
    ax.axis('equal')
    ax.set_xlabel('world x [m]')
    ax.set_ylabel('world y [m]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i in range(10):
        with suppress():
            mkstops,ircstops=get_stops(states, tasks, thistask, env, finaltheta, agent,pert=pert)
        ax.scatter(ircstops[:,0]*2,ircstops[:,1]*2,color=color_settings['model'], label='model',s=1)
    # ax.scatter(mkstops[:,0]*2,mkstops[:,1]*2, color=color_settings['a'], label='monkey')
    # ax.set_xlim(left=0.)
    # ax.set_ylim(bottom=0.)
    ax.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())



# check if victor not affected by density

data_path=Path("Z:\\victor_normal")
process_inv(data_path/'preallpacked')

densities=sorted(pd.unique(df.floor_density))
d=df[df.d]

len(df[df.rewarded])/len(df)


# human data, a vs h
data_path=Path("Z:/human")
theta,_,_=process_inv(data_path/'fixrhgroup', removegr=False)
print('finished')

# simulation data
data_path=Path("Z:/simulation")
theta,_,_=process_inv(data_path/'invbliefsimulationpert100', removegr=False)












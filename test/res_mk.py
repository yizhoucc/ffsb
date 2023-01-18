# result mk figures 

from unittest import skip
from sklearn.linear_model import LogisticRegression
import pandas as pd

from plot_ult import datawash
from pathlib import Path
import pickle
from plot_ult import *
from stable_baselines3 import TD3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt
from stable_baselines3 import TD3
torch.manual_seed(42)
from numpy import pi
from InverseFuncs import *
from monkey_functions import *
from FireflyEnv import ffacc_real
from Config import Config
# from cma_mpi_helper import run
from pathlib import Path
arg = Config()


# % theta hist ---------------------
datafolder=Path(os.getenv('datafolder'))
logls=[
    datafolder/('bruno_pert/cmafull_b_pert'),
    datafolder/('schro_pert/cmafull_packed_schro_pert'),
    datafolder/('q_pert/cma180paper_packed'),
    datafolder/('victor_pert/cmafull_victor_pert_ds'),
    datafolder/('jimmy_pert/des1sub200packed'),
    datafolder/('marco_pert/des1sub200packed'),
]
# logls=[
#     Path('Z:/bruno_pert/cmafull_b_pert'),
#     Path('Z:/schro_pert/cmafull_packed_schro_pert'),
#     Path('Z:/q_pert/cma180paper_packed'),
#     Path('Z:/victor_pert/cmafull_victor_pert_ds'),
#     Path('Z:/jimmy_pert/des1sub200packed'),
#     Path('Z:/marco_pert/des1sub200packed'),
# ]
monkeynames=['bruno', 'schro', 'quigley','victor','jimmy','maraco']

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


# plot the inv res scatteers bar (new formating to align with asd data)
subshift=0.08
with initiate_plot(7,2,300) as fig:
    ax=fig.add_subplot(111)
    quickspine(ax)
    ax.set_xticks(list(range(10)))
    ax.set_xticklabels(theta_names, rotation=45, ha='right')
    ax.set_ylabel('inferred param value')
    colory=np.linspace(0,2,50) # the verticle range of most parameters
    colory=torch.linspace(0,2,50) # the verticle range of most parameters
    for ithsub, logfile in enumerate(logls): # each mk subject
        if ithsub==0: # bruno
            basecolor=[41/255,171/255,1]
        elif ithsub==3:
            basecolor=[57/255,181/255,74/255]
        else:
            basecolor=[0,0,1]
        log=process_inv(logfile,ind=60)
        for i, mu in enumerate(log[0]): # each parameter
            std=(torch.diag(log[1])**0.5)[i]*0.1
            prob=lambda y: torch.exp(-0.5*(y-mu)**2/std**2)
            proby=prob(colory)
            for j in range(len(colory)-1):
                c=proby[j].item()
                plt.plot([i-(ithsub)*subshift,i-(ithsub)*subshift],[colory[j],colory[j+1]],c=basecolor, alpha=c,linewidth=3)


# % example inferred belief in one trial ---------------------
# load the inferred theta
theta,_,_=process_inv("Z:/bruno_pert/cmafull_b_pert", removegr=False)
env=ffacc_real.FireFlyPaper(arg)
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()
critic=agent_.critic.qf0.cpu()

# load the behavior data
filenamecommom=Path('Z:\\bruno_pert')
with open(filenamecommom/'packed', 'rb') as f:
    df = pickle.load(f)
    df=datawash(df)
    df=df[(df.category=='normal') & (df.floor_density==0.005)] 
    df=df[df.perturb_start_time.isnull()]
states,actions,tasks=monkey_data_downsampled(df,factor=0.0025)

# select a trial to plot
ind=np.random.randint(low=0, high=len(tasks))
s,a,t=states[ind],actions[ind],tasks[ind]
while len(a)<20:
    ind=np.random.randint(low=0, high=len(tasks))
    s,a,t=states[ind],actions[ind],tasks[ind]

with initiate_plot(2, 2, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax1 = fig.add_subplot(111)
    quickspine(ax1)
    ax1.set_xlabel('world x [cm]')
    ax1.set_ylabel('world y [cm]')
    goalcircle = plt.Circle((t[0],t[1]), 0.13,color=color_settings['goal'], edgecolor='none', alpha=0.5)
    ax1.add_patch(goalcircle)
    ax1.set_xlim([-0.1,1.1])
    ax1.set_ylim([-0.6,0.6])
    ax1.set_xticks([0,1])
    ax1.set_yticks([-0.5,0.,0.5])
    ax1.set_xticklabels([0,200])
    ax1.set_yticklabels([-100,0.,100])    
    with torch.no_grad():
            agent_actions=[]
            agent_beliefs=[]
            agent_covs=[]
            agent_states=[]
            env.reset(phi=phi,theta=theta,goal_position=t,vctrl=a[0,0],wctrl=a[0,1])
            epbliefs=[]
            epbcov=[]
            epactions=[]
            epstates=[]
            done=False
            while not done:
                action = agent(env.decision_info)[0]
                _,_,done,_=env.step(torch.tensor(action).reshape(1,-1)) 
                epactions.append(action)
                epbliefs.append(env.b)
                epbcov.append(env.P)
                epstates.append(env.s)
            # print(env.get_distance(env.s))
            # agent_dev_costs.append(torch.stack(env.trial_dev_costs))
            # agent_mag_costs.append(torch.stack(env.trial_mag_costs))
            agent_actions.append(torch.stack(epactions))
            agent_beliefs.append(torch.stack(epbliefs))
            agent_covs.append(epbcov)
            agent_states.append(torch.stack(epstates))
            estate=torch.stack(agent_states)[0,:,:,0]
            estate[:,0]=estate[:,0]-estate[0,0]
            estate[:,1]=estate[:,1]-estate[0,1]
            # agent path, red
            print('ploting')
            # ax1.plot(estate[:,0],estate[:,1], color=color_settings['s'],alpha=0.5)
            # agent belief path
            for dt in range(0,len(agent_covs[0])):
                if dt%5==0 or dt==len(agent_beliefs[0][:,:,0])-1:
                    cov=agent_covs[0][dt][:2,:2]
                    pos=  [agent_beliefs[0][:,:,0][dt,0],
                            agent_beliefs[0][:,:,0][dt,1]]
                    plot_cov_ellipse(cov, pos, nstd=2, color=color_settings['b'], ax=ax1,alpha=0.5)
    # mk state, blue
    ax1.plot(s[:,0],s[:,1],color=color_settings['s'],alpha=0.9)



# % radial error and miss rate vs density ---------------------
# load the behavior data
monkeys=['bruno', 'schro','victor','quigley']

mk2beh={}
for m in monkeys:
    datapath=Path("Z:\\{}_normal\\packed".format(m))
    with open(datapath,'rb') as f:
        df = pickle.load(f)
    df=datawash(df)
    df=df[df.category=='normal']
    densities=sorted(df.floor_density.unique())
    percentrewarded = [len(df[(df.floor_density==i) & (df.rewarded)])/(len(df[df.floor_density==i])+1e-8) for i in densities]
    radiualerr=[np.mean(df[(df.floor_density==i)].relative_radius_end) for i in densities]
    mk2beh[m]=[densities,percentrewarded,radiualerr]

for k,v in mk2beh.items():
    with initiate_plot(4,3) as fig:
        ax=fig.add_subplot(111)
        ax2 = ax.twinx()
        ax.bar([i-0.2 for i in range(len((v[0])))], v[2], label=k,color='k',width=0.4)
        ax.set_xlabel('density')
        ax.set_ylabel('radial error [cm]')
        ax.set_xticks(list(range(len((v[0])))))
        ax.set_xticklabels(v[0])
        datamax=max(v[2])
        roundmax=np.round(max(v[2]),decimals=-1)
        plotmax=roundmax if roundmax>datamax else roundmax+10
        ax.set_yticks([0,plotmax])
        ax.spines['top'].set_visible(False)

        ax2.bar([i+0.2 for i in range(len((v[0])))], [1-vv for vv in v[1]], label=k,color='r',width=0.4)
        ax2.set_xlabel('density')
        ax2.set_ylabel('miss rate')
        ax2.set_xticks(list(range(len((v[0])))))
        ax2.set_xticklabels(v[0])
        ax2.set_yticks([0,np.round(1-min(v[1]),decimals=1)+0.1])
        ax2.spines['top'].set_visible(False)

        ax2.text(0,0.8,'monkey {}'.format(k))



# % most likely obs noise vs density ---------------------

with open('brunoobsdenslog500','rb') as f:
    obsvls,p = pickle.load(f)
basecolor=hex2rgb(color_settings['o'])
colors=colorshift(basecolor, basecolor, 1, len(densities))

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
    
    peaks=obsvls[np.array([np.argmax(smooth(pp,3)) for pp in p])]
    for i,each in enumerate(peaks):
        ax.plot(each,ax.get_ylim()[1]/100,marker='^',color=colors[i])



# % victor respond differently to pert ---------------------
# load victor pert data and control pert data
data={}
filenamecommom=Path('Z:\\victor_pert')
with open(filenamecommom/'packed', 'rb') as f:
    df = pickle.load(f)
    df=datawash(df)
    df=df[(df.category=='normal') & (df.floor_density==0.005)] # verify the density
    # df=df[~df.perturb_start_time.isnull()] # pert only trial
data['vdf']=df
data['vpert']=monkey_data_downsampled(df,factor=0.0025)
data['vpertmisc']=df_downsamplepert(df,factor=0.0025)

filenamecommom=Path('Z:\\bruno_pert')
with open(filenamecommom/'packed', 'rb') as f:
    df = pickle.load(f)
    df=datawash(df)
    df=df[(df.category=='normal') & (df.floor_density==0.005)] # verify the density
    # df=df[~df.perturb_start_time.isnull()] # pert only trial
data['bdf']=df
data['bpert']=monkey_data_downsampled(df,factor=0.0025)
data['bpertmisc']=df_downsamplepert(df,factor=0.0025)

filenamecommom=Path('Z:\\jimmy_pert')
with open(filenamecommom/'packed', 'rb') as f:
    df = pickle.load(f)
    df=datawash(df)
    df=df[(df.category=='normal') & (df.floor_density==0.005)] # verify the density
    # df=df[~df.perturb_start_time.isnull()] # pert only trial
data['jdf']=df
data['jpert']=monkey_data_downsampled(df,factor=0.0025)
data['jpertmisc']=df_downsamplepert(df,factor=0.0025)

filenamecommom=Path('Z:\\marco_pert')
with open(filenamecommom/'packed', 'rb') as f:
    df = pickle.load(f)
    df=datawash(df)
    df=df[(df.category=='normal') & (df.floor_density==0.005)] # verify the density
    # df=df[~df.perturb_start_time.isnull()] # pert only trial
data['mdf']=df
data['mpert']=monkey_data_downsampled(df,factor=0.0025)
data['mpertmisc']=df_downsamplepert(df,factor=0.0025)


# select a task, and get similar tasks to this
done=False
while not done:
    ind=np.random.randint(0,len(data['vpert'][2]))
    done=data['vpertmisc'][1][ind][0]!=0
# ind=227
thetask=data['vpert'][2][ind]
thispert=data['vpertmisc'][1][ind]
plt.plot(data['vpertmisc'][0][ind])

# plot paths
indls=similar_trials2thispert(data['vpert'][2], thetask,thispert, ntrial=5,pertmeta=data['vpertmisc'][1])

substates=[data['vpert'][0][i] for i in indls]
subactions=[data['vpert'][1][i] for i in indls]
subtasks=np.array(data['vpert'][2])[indls]
subperts=[data['vpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['vpertmisc'][1])[indls]
# for p in subperts:
#     plt.plot(p)

ax=plotoverhead_simple(substates,thetask,color='r',label='victor',ax=None)

indls=similar_trials2thispert(data['bpert'][2], thetask,thispert, ntrial=5,pertmeta=data['bpertmisc'][1])
substates=[data['bpert'][0][i] for i in indls]
subactions=[data['bpert'][1][i] for i in indls]
subtasks=np.array(data['bpert'][2])[indls]
subperts=[data['bpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['bpertmisc'][1])[indls]
ax=plotoverhead_simple(substates,thetask,color='b',label='bruno',ax=ax)
ax.get_figure()


# jimmy
done=False
while not done:
    ind=np.random.randint(0,len(data['jpert'][2]))
    done=data['jpertmisc'][1][ind][0]!=0
# ind=227
thetask=data['jpert'][2][ind]
thispert=data['jpertmisc'][1][ind]
# plt.plot(data['jpertmisc'][0][ind])
indls=similar_trials2thispert(data['jpert'][2], thetask,thispert, ntrial=3,pertmeta=data['jpertmisc'][1])
substates=[data['jpert'][0][i] for i in indls]
subactions1=[data['jpert'][1][i] for i in indls]
subtasks=np.array(data['jpert'][2])[indls]
subperts=[data['jpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['jpertmisc'][1])[indls]
ax=plotoverhead_simple(substates,thetask,color='r',label='jimmy',ax=None)
indls=similar_trials2thispert(data['bpert'][2], thetask,thispert, ntrial=3,pertmeta=data['bpertmisc'][1])
substates=[data['bpert'][0][i] for i in indls]
subactions=[data['bpert'][1][i] for i in indls]
subtasks=np.array(data['bpert'][2])[indls]
subperts=[data['bpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['bpertmisc'][1])[indls]
ax=plotoverhead_simple(substates,thetask,color='b',label='bruno',ax=ax)
ax.get_figure()

with initiate_plot(2,1,300) as f:
    ax=f.add_subplot(111)
    quickspine(ax)
    for a in subactions1:
        plt.plot(a,'r', label='jimmy')
    for a in subactions:
        plt.plot(a,'b',label='bruno')
    quickleg(ax)

# marco
done=False
while not done:
    ind=np.random.randint(0,len(data['mpert'][2]))
    done=data['mpertmisc'][1][ind][0]!=0
# ind=227
thetask=data['mpert'][2][ind]
thispert=data['mpertmisc'][1][ind]
plt.plot(data['mpertmisc'][0][ind])
indls=similar_trials2thispert(data['mpert'][2], thetask,thispert, ntrial=3,pertmeta=data['mpertmisc'][1])
substates=[data['mpert'][0][i] for i in indls]
subactions1=[data['mpert'][1][i] for i in indls]
subtasks=np.array(data['mpert'][2])[indls]
subperts=[data['mpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['mpertmisc'][1])[indls]
ax=plotoverhead_simple(substates,thetask,color='r',label='marco',ax=None)
indls=similar_trials2thispert(data['bpert'][2], thetask,thispert, ntrial=3,pertmeta=data['bpertmisc'][1])
substates=[data['bpert'][0][i] for i in indls]
subactions=[data['bpert'][1][i] for i in indls]
subtasks=np.array(data['bpert'][2])[indls]
subperts=[data['bpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['bpertmisc'][1])[indls]
ax=plotoverhead_simple(substates,thetask,color='b',label='bruno',ax=ax)
ax.get_figure()

with initiate_plot(2,1,300) as f:
    ax=f.add_subplot(111)
    quickspine(ax)
    for a in subactions1:
        plt.plot(a,'r', label='marco')
    for a in subactions:
        plt.plot(a,'b',label='bruno')
    quickleg(ax)


# does pert angle affect jimmy
pertrewardeddf=df[(df.rewarded)&(~df.perturb_start_time.isnull())]
pertunrewardeddf=df[(~df.rewarded)&(~df.perturb_start_time.isnull())]
plt.hist(pertrewardeddf.perturb_wpeak)
plt.hist(pertunrewardeddf.perturb_wpeak,alpha=0.3)

plt.hist(pertrewardeddf.perturb_vpeak)
plt.hist(pertunrewardeddf.perturb_vpeak,alpha=0.3)

# perttheta=np.array([np.max(np.abs(i[:,1])) for i in data['jpertmisc'][0]])
# pertinds=np.where(perttheta!=0)[0]
# pertmask = np.zeros(len(perttheta), dtype=int)
# pertmask[pertmask] = 1
# rewardmask=np.array(data['jdf'].rewarded)*1
# rewardedinds=np.array([i for i,a in enumerate(data['jdf'].rewarded) if a])
# rewardedpert = (pertmask == 1) & (rewardmask == 1)

# pertthetamasked=perttheta[pertmask]
# # rewardedmasked=rewarded[pertmask]
# # plt.hist(pertthetamasked,bins=30)
# plt.hist(pertthetamasked[rewardedpert],bins=30)

# sum(rewarded)/len(df)
# sum(rewardedmasked)/len(pertmask)

# show victor is good/same when doing normal trials ------------------
# select a non pert task, and get similar tasks to this
done=False
while not done:
    ind=np.random.randint(0,len(data['vpert'][2]))
    done=data['vpertmisc'][1][ind][0]==0
# ind=227
thetask=data['vpert'][2][ind]
thispert=data['vpertmisc'][1][ind]
plt.plot(data['vpertmisc'][0][ind])


# plot paths
indls=similar_trials2thispert(data['vpert'][2], thetask,thispert, ntrial=5,pertmeta=data['vpertmisc'][1])

substates=[data['vpert'][0][i] for i in indls]
subactions=[data['vpert'][1][i] for i in indls]
subtasks=np.array(data['vpert'][2])[indls]
subperts=[data['vpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['vpertmisc'][1])[indls]
# for p in subperts:
#     plt.plot(p)

ax=plotoverhead_simple(substates,thetask,color='r',label='victor',ax=None)
indls=similar_trials2thispert(data['bpert'][2], thetask,thispert, ntrial=5,pertmeta=data['bpertmisc'][1])
substates=[data['bpert'][0][i] for i in indls]
subactions=[data['bpert'][1][i] for i in indls]
subtasks=np.array(data['bpert'][2])[indls]
subperts=[data['bpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['bpertmisc'][1])[indls]
ax=plotoverhead_simple(substates,thetask,color='b',label='bruno',ax=ax)
ax.get_figure()


# check other two monkey, if doing good pert
# marco seems good by overhead path
filenamecommom=Path('Z:\\marco_pert')
with open(filenamecommom/'packed', 'rb') as f:
    df = pickle.load(f)
    df=datawash(df)
    df=df[(df.category=='normal') & (df.floor_density==0.005)] # verify the density
    # df=df[~df.perturb_start_time.isnull()] # pert only trial
data['mdf']=df
data['mpert']=monkey_data_downsampled(df,factor=0.0025)
data['mpertmisc']=df_downsamplepert(df,factor=0.0025)

# select a task, and get similar tasks to this
done=False
while not done:
    ind=np.random.randint(0,len(data['mpert'][2]))
    done=data['mpertmisc'][1][ind][0]!=0
# ind=227
thetask=data['mpert'][2][ind]
thispert=data['mpertmisc'][1][ind]
plt.plot(data['mpertmisc'][0][ind])

# plot paths
indls=similar_trials2thispert(data['mpert'][2], thetask,thispert, ntrial=5,pertmeta=data['mpertmisc'][1])
substates=[data['mpert'][0][i] for i in indls]
subactions=[data['mpert'][1][i] for i in indls]
subtasks=np.array(data['mpert'][2])[indls]
subperts=[data['mpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['mpertmisc'][1])[indls]

ax=plotoverhead_simple(substates,thetask,color='r',label='marco',ax=None)
indls=similar_trials2thispert(data['bpert'][2], thetask,thispert, ntrial=5,pertmeta=data['bpertmisc'][1])
substates=[data['bpert'][0][i] for i in indls]
subactions=[data['bpert'][1][i] for i in indls]
subtasks=np.array(data['bpert'][2])[indls]
subperts=[data['bpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['bpertmisc'][1])[indls]
ax=plotoverhead_simple(substates,thetask,color='b',label='bruno',ax=ax)
ax.get_figure()

# plot actions
for suba in subactions:
    plt.plot(suba)
for p in subperts:
    plt.plot(p)
for suba in substates:
    plt.plot(suba)

# marco reward status
len(data['mdf'][data['mdf'].rewarded])/len(data['mdf'])
len(data['bdf'][data['bdf'].rewarded])/len(data['bdf'])
len(data['vdf'][data['vdf'].rewarded])/len(data['vdf'])

len(data['mdf'][(data['mdf'].rewarded) & data['mdf'].perturb_start_time.isnull()])/len(data['mdf'][data['mdf'].perturb_start_time.isnull()])
len(data['bdf'][(data['bdf'].rewarded) & data['bdf'].perturb_start_time.isnull()])/len(data['bdf'][data['bdf'].perturb_start_time.isnull()])
len(data['vdf'][(data['vdf'].rewarded) & data['vdf'].perturb_start_time.isnull()])/len(data['vdf'][data['vdf'].perturb_start_time.isnull()])





# all mk that has pert together ----------------------
# select a task
done=False
while not done:
    ind=np.random.randint(0,len(data['mpert'][2]))
    done=data['mpertmisc'][1][ind][0]!=0
# ind=227
# ind=6172
thetask=data['mpert'][2][ind]
thispert=data['mpertmisc'][1][ind]
plt.plot(data['mpertmisc'][0][ind])

indls=similar_trials2thispert(data['mpert'][2], thetask,thispert, ntrial=3,pertmeta=data['mpertmisc'][1])
substates=[data['mpert'][0][i] for i in indls]
subactions=[data['mpert'][1][i] for i in indls]
subtasks=np.array(data['mpert'][2])[indls]
subperts=[data['mpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['mpertmisc'][1])[indls]
ax=plotoverhead_simple(substates,thetask,color='r',label='marco',ax=None)

indls=similar_trials2thispert(data['bpert'][2], thetask,thispert, ntrial=3,pertmeta=data['bpertmisc'][1])
substates=[data['bpert'][0][i] for i in indls]
subactions=[data['bpert'][1][i] for i in indls]
subtasks=np.array(data['bpert'][2])[indls]
subperts=[data['bpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['bpertmisc'][1])[indls]
ax=plotoverhead_simple(substates,thetask,color='b',label='bruno',ax=ax)

indls=similar_trials2thispert(data['vpert'][2], thetask,thispert, ntrial=3,pertmeta=data['vpertmisc'][1])
substates=[data['vpert'][0][i] for i in indls]
subactions=[data['vpert'][1][i] for i in indls]
subtasks=np.array(data['vpert'][2])[indls]
subperts=[data['vpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['vpertmisc'][1])[indls]
ax=plotoverhead_simple(substates,thetask,color='k',label='victor',ax=ax)


indls=similar_trials2thispert(data['jpert'][2], thetask,thispert, ntrial=3,pertmeta=data['jpertmisc'][1])
substates=[data['jpert'][0][i] for i in indls]
subactions=[data['jpert'][1][i] for i in indls]
subtasks=np.array(data['jpert'][2])[indls]
subperts=[data['jpertmisc'][0][i] for i in indls]
subpertmeta=np.array(data['jpertmisc'][1])[indls]
ax=plotoverhead_simple(substates,thetask,color='g',label='jimmy',ax=ax)
ax.get_figure()

quicksave('allmkpert overhead ind={}'.format(ind),fig=ax.get_figure())



# % value map from critic ---------------------

# load the inferred theta
theta,_,_=process_inv("Z:/bruno_pert/cmafull_b_pert", removegr=False)

# load the agent and env
env=ffacc_real.FireFlyPaper(arg)
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()
critic=agent_.critic.qf0.cpu()

# sample all possible tasks evenly
X,Y=np.linspace(0,1.2,50),np.linspace(-1,1,100)
Xgrid,Ygrid=np.meshgrid(X,Y)
tars=[]
for x,y in zip(Xgrid.flatten(), Ygrid.flatten()):
    d,a=xy2pol([x,y],rotation=False)
    if 0.3<d<1. and -0.6<a<0.6:
        tars.append((a,d))
tars=np.array(tars)

# vary the situations (decision info) and calculate values
values=[]
with torch.no_grad():
    env.reset(phi=phi,theta=theta)
    base=env.decision_info.flatten().clone().detach() 
    # base[2]=0.001; base[4]=0.001
    for a,d in tars:
        base[0]=d; base[1]=a
        v=critic(torch.cat([base,agent(base).flatten()])).item()
        values.append(v)
    values=np.asarray(values)

plt.subplot(111,polar='True')
plt.scatter(tars[:,0],tars[:,1],s=1,c=values)
plt.ylim(0.2,1)

plt.subplot(111,polar='True')
plt.tricontourf(tars[:,0],tars[:,1],values,9)
plt.ylim(0.,1)


# % value map from actor ---------------------

# load the inferred theta
theta,_,_=process_inv("Z:/bruno_pert/cmafull_b_pert", removegr=False)

# load the agent and env
env=ffacc_real.FireFlyPaper(arg)
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()
critic=agent_.critic.qf0.cpu()

# sample all possible tasks evenly
X,Y=np.linspace(0,1.2,33),np.linspace(-1,1,33)
Xgrid,Ygrid=np.meshgrid(X,Y)
tars=[]
for x,y in zip(Xgrid.flatten(), Ygrid.flatten()):
    d,a=xy2pol([x,y],rotation=False)
    if 0.3<d<1. and -0.6<a<0.6:
        tars.append([x,y])
tars=np.array(tars)

def evaltrial(env,agent,task,phi,theta,nsample=5):
    env.debug=1
    env.cost_scale=0.1
    values=[]
    for i in range(nsample):
        env.reset(phi=phi,theta=theta,goal_position=task)
        done=False
        while not done:
            _,_,done,_=env.step(agent(env.decision_info))
        v=np.sum(env.trial_rewards)-np.sum(env.trial_costs)
        values.append(v)
    meanvalue=np.mean(values)
    return meanvalue

# vary the situations (decision info) and calculate values
values=[]
for x,y in tars:
    with torch.no_grad():
        # impose symetry, by taking the higher value
        v=max(evaltrial(env,agent,[x,y],phi,theta,nsample=10),evaltrial(env,agent,[x,-y],phi,theta,nsample=10))
        values.append(v)
values=np.array(values)
# plot
with initiate_plot(2, 2,300) as fig:
    ax=fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlabel('world x [cm]')
    ax.set_ylabel('world y [cm]')
    quickspine(ax)
    ax.set_xticks(mytick(ax.get_xticks(),3,0))
    ax.set_yticks(mytick(ax.get_yticks(),3,0))
    im=ax.tricontourf(tars[:,0],tars[:,1],normalizematrix(values),11)
    ax.set_xlim(0,1)
    fig.colorbar(im, ax=ax,label='value')
    # ax.scatter([0.],[0.],marker='*')

# save the value map
with open('valuemap33', 'wb+') as f:
    pickle.dump((values,tars), f, protocol=pickle.HIGHEST_PROTOCOL)
with open('valuemap33', 'rb') as f:
    values,tars= pickle.load(f)

# % histogram of skips number vs value ----------------------

# # get monkey data (bruno) random skip
# filenamecommom=Path('Z:\\bruno_normal')
# with open(filenamecommom/'packed', 'rb') as f:
#     df = pickle.load(f)
# data['bnormdf']=df
# skipdf=data['bnormdf'][data['bnormdf'].category=='skip']
# normaldf=data['bnormdf'][data['bnormdf'].category=='normal'] 

# get monkey data (victor)
filenamecommom=Path('Z:\\victor_normal')
with open(filenamecommom/'packed', 'rb') as f:
    df = pickle.load(f)
data['vnormdf']=df
skipdf=data['vnormdf'][data['vnormdf'].category=='skip']
normaldf=data['vnormdf'][data['vnormdf'].category=='normal']


# get skips and non skips targets
tarskip=np.vstack([(np.array(skipdf.target_y)/400),(np.array(skipdf.target_x)/400)])
tarnormal=np.vstack([(np.array(normaldf.target_y)/400),(np.array(normaldf.target_x)/400)])
tarskippolar=xy2pol(tarskip,rotation=False)
tarnormpolar=xy2pol(tarnormal,rotation=False)

# plot the hist

# distance
nbin=20
skipd=np.histogram(tarskippolar[0],nbin)
hist=skipd[0]/np.histogram(tarnormpolar[0],bins=skipd[1])[0]
# plt.bar(skipd[1][:-1],hist,width=1/nbin)
with initiate_plot(2,2,300) as f:
    ax=f.add_subplot(111)
    ax.bar(skipd[1][:-1],hist,width=(max(skipd[1][:-1])-min(skipd[1][:-1]))/nbin)
    quickspine(ax)
    ax.set_xlim(0.25,1)
    ax.set_xlabel('target distance [2 m]')
    ax.set_ylabel('skip probability')


# angle
skipd=np.histogram(tarskippolar[1],nbin)
hist=skipd[0]/np.histogram(tarnormpolar[1],bins=skipd[1])[0]
# plt.bar(skipd[1][:-1],hist,width=1/nbin)
with initiate_plot(2,2,300) as f:
    ax=f.add_subplot(111)
    ax.bar(skipd[1][:-1],hist,width=(max(skipd[1][:-1])-min(skipd[1][:-1]))/nbin)
    quickspine(ax)
    # ax.set_xlim(0,1)
    ax.set_xlabel('target angle [rad]')
    ax.set_ylabel('skip probability')

# value
tarskipvalues=[]
for x,y in tarskip.T:
    dstothis=[(xx-x)**2+(yy-y)**2 for xx,yy in zip(tars[:,0],tars[:,1]) ]
    nearest3=np.argsort(dstothis)[:3]
    v=np.mean(values[nearest3])
    tarskipvalues.append(v)
tarnormvalues=[]
for x,y in tarnormal.T:
    dstothis=[(xx-x)**2+(yy-y)**2 for xx,yy in zip(tars[:,0],tars[:,1]) ]
    nearest3=np.argsort(dstothis)[:3]
    v=np.mean(values[nearest3])
    tarnormvalues.append(v)
# values of the skipped trials overhead
plt.scatter(tarskip[0],tarskip[1],alpha=0.6,c=tarskipvalues)
plt.scatter(tarnormal[0],tarnormal[1],alpha=0.6,c=tarnormvalues)
# values distribution 
plt.hist(tarskipvalues,density=True)
plt.hist(tarnormvalues,density=True,alpha=0.5)
# they have the same value mean
np.mean(tarskipvalues)
np.mean(tarnormvalues)

# skip and non skip value compare hist
with initiate_plot(2,2,300) as f:
    ax=f.add_subplot(111)
    quickspine(ax)
    ax.hist(normalizematrix(tarskipvalues),density=True,label='skip',alpha=0.9)
    ax.hist(normalizematrix(tarnormvalues),density=True,label='non skip',alpha=0.7)
    ax.set_xlabel('value')
    ax.set_ylabel('probability')
    leg=ax.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

# skip prob vs value
nbin=20
skipd=np.histogram(normalizematrix(tarskipvalues),nbin)
nonskipd=np.histogram(normalizematrix(tarnormvalues),bins=skipd[1])
hist=skipd[0]/(nonskipd[0]+skipd[0])
with initiate_plot(2,2,300) as f:
    ax=f.add_subplot(111)
    quickspine(ax)
    ax.bar(skipd[1][:-1]+0.5*(max(skipd[1][:-1])-min(skipd[1][:-1]))/nbin,hist,width=(max(skipd[1][:-1])-min(skipd[1][:-1]))/nbin)
    ax.set_xlabel('value')
    ax.set_xlim(0,0.5)
    ax.set_ylim(0,0.3)
    ax.set_ylabel('skip probability')

# all trial value hist
with initiate_plot(2,2,300) as f:
    ax=f.add_subplot(111)
    quickspine(ax)
    ax.hist(np.hstack([normalizematrix(tarskipvalues),normalizematrix(tarnormvalues)]),density=True,bins=20)
    ax.set_xlabel('value')
    ax.set_xlim(0,1)
    ax.set_ylabel('probability')




# % model the history skip effect, by logistic regression ---------------------
filenamecommom=Path('Z:\\victor_normal')
with open(filenamecommom/'packed', 'rb') as f:
    df = pickle.load(f)
# o1 label the trial type list
subdf=df[(df.category=='skip') | (df.category=='normal')]
tasks=np.vstack([(np.array(subdf.target_y)/400),(np.array(subdf.target_x)/400)])
prevd=normalizematrix(np.array(subdf.target_r/400))
preva=normalizematrix(np.array(subdf.target_theta/400))
preverr=normalizematrix(np.array(subdf.relative_radius_end/400))
encodedtrialtype=[1 if each=='skip' else 0  for each in subdf.category]
rewarded=[1 if eachrewarded else 0 for eachrewarded in list(subdf.rewarded)]
totalvalues=[]
for x,y in tasks.T:
    dstothis=[(xx-x)**2+(yy-y)**2 for xx,yy in zip(tars[:,0],tars[:,1]) ]
    nearest3=np.argsort(dstothis)[:3]
    v=np.mean(values[nearest3])
    totalvalues.append(v)
totalvalues=normalizematrix(totalvalues)


# make some fake to test
# no dynamic, just a sin
# encodedtrialtype=np.sin(np.linspace(1,100,1000))*0.5+0.5
# encodedtrialtype=np.round(encodedtrialtype)
# dynamic
# def fakedynamic(buffer):
#     lastskip=buffer[0]==1
#     numskip=sum(buffer)
#     if lastskip and numskip<3:
#         return 1
#     elif numskip==0:
#         return 1
#     else: return 0
# def fakedynamic(buffer):
#     p=0
#     for i,x in enumerate(buffer):
#         p+=np.random.random(1)/(7-i)
#     return int(np.round(p))
# def fakedynamic(buffer):
#     buffer=np.array(buffer)
#     weights=np.array([-0.2,-0.1,-0.2,0.2,0.8])
#     return int(np.round(buffer@weights))


# encodedtrialtype=[]
# buffer=deque([0,1,0,1,0])
# for _ in range(100000):
#     x=fakedynamic(buffer)
#     encodedtrialtype.append(buffer.popleft())
#     noise=np.random.random(1)
#     if noise<0.2:
#         x=1
#     elif noise>0.8:
#         x=0
#     buffer.append(x)
# encodedtrialtype=np.array(encodedtrialtype)
# plt.plot(encodedtrialtype[:67])
num_steps=15
lrdata=[]
for i in range(num_steps, len(encodedtrialtype)-num_steps):
    inputtype=np.zeros(num_steps)
    inputreward=np.zeros(num_steps)
    inputvalue=np.zeros(num_steps)
    inputpreverr=np.zeros(num_steps)
    inputprevd=np.zeros(num_steps)
    inputpreva=np.zeros(num_steps)
    for j in range(0,num_steps):
        inputvalue[j]=totalvalues[i-j]
    for j in range(1,num_steps):
        inputtype[j]=encodedtrialtype[i-j]
        inputreward[j]=rewarded[i-j]
        # inputvalue[j]=totalvalues[i-j]
        # inputpreverr[j]=preverr[i-j]
        # inputprevd[j]=prevd[i-j]
        # inputpreva[j]=preva[i-j]
    lrdata.append(np.hstack([inputtype,inputreward,inputvalue, inputpreverr, inputprevd, inputpreva]))
lrdata=np.array(lrdata)

x=lrdata
y=encodedtrialtype[num_steps:][:len(lrdata)]
x=np.array(x)
y=np.array(y)

# train set and test set and eval
totalacc=0
for i in range(2):
    splitpoint=np.random.randint(0,len(lrdata)-200)
    trainx=np.concatenate((x[:splitpoint],x[splitpoint+100:]),axis=0)
    trainy=np.concatenate((y[:splitpoint],y[splitpoint+100:]),axis=0)
    valx=x[splitpoint:splitpoint+100]
    valy=y[splitpoint:splitpoint+100]
    testx=x[splitpoint+100:splitpoint+155]
    testy=y[splitpoint+100:splitpoint+155]
    # use distance and angle to predict cur type
    model = LogisticRegression(fit_intercept=False)
    model.fit(trainx,trainy)
    print("val acc: %.2f" % model.score(valx, valy, sample_weight=None))
    totalacc+=model.score(testx, testy, sample_weight=None)
    print("acc: %.2f" % model.score(testx, testy, sample_weight=None))
    totalacc+=model.score(testx, testy, sample_weight=None)
    # print('mean acc ',totalacc/10 )

    # coef histogram
    with initiate_plot(3,2,300) as f:
        ax=f.add_subplot(111)
        # num_steps=len(model.coef_.flatten())//2
        ax.bar(list(range(num_steps)),model.coef_.flatten()[:num_steps],label='skipped')
        ax.set_ylabel('logistic regression coef')
        ax.set_xlabel('previous ith trial type')
        quickspine(ax)
        # if use rewarded also
        ax.bar(np.arange(num_steps,num_steps*2)+3*1,model.coef_.flatten()[num_steps:num_steps*2],label='rewarded')
        ax.bar(np.arange(num_steps*2,num_steps*3)+3*2,model.coef_.flatten()[num_steps*2:num_steps*3],label='value')   
        ax.bar(np.arange(num_steps*3,num_steps*4)+3*3,model.coef_.flatten()[num_steps*3:num_steps*4],label='radial err')
        ax.bar(np.arange(num_steps*4,num_steps*5)+3*4,model.coef_.flatten()[num_steps*4:num_steps*5],label='distance')
        ax.bar(np.arange(num_steps*5,num_steps*6)+3*5,model.coef_.flatten()[num_steps*5:num_steps*6],label='angle')
        ax.legend(loc='upper right', bbox_to_anchor=(0,0))
        # ax.set_xticks([0,num_steps,num_steps+3,num_steps*2+3,num_steps*2+6,num_steps*3+6])
        # ax.set_xticklabels([-1,-5]*3)
        ax.set_xticklabels([])


    # prediction
    with initiate_plot(2,1,300) as f:
        ax=f.add_subplot(111)
        quickspine(ax)
        n=len(testy)
        plt.bar(list(range(n)),testy,label='ground truth skips')
        plt.bar(list(range(n)),-1*(model.predict(testx)),label='predicted skips')
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_xlim(0,n)
        ax.set_xlabel('test set trials')
        ax.legend(loc='upper right', bbox_to_anchor=(0,0))
        # ax.text(-0,1,'prediction from {} step history'.format(str(num_steps)))





# vary n steps
maxnumsteps=8
numsamples=100
totalaccs=[]
for num_steps in range(1,maxnumsteps+1):
    lrdata=[]
    for i in range(num_steps, len(encodedtrialtype)-num_steps):
        inputtype=np.zeros(num_steps)
        inputreward=np.zeros(num_steps)
        inputvalue=np.zeros(num_steps)
        # inputpreverr=np.zeros(num_steps)
        # inputprevd=np.zeros(num_steps)
        # inputpreva=np.zeros(num_steps)
        for j in range(0,num_steps):
            inputvalue[j]=totalvalues[i-j]
        for j in range(1,num_steps):
            inputtype[j]=encodedtrialtype[i-j]
            inputreward[j]=rewarded[i-j]
            # inputpreverr[j]=preverr[i-j]
            # inputprevd[j]=prevd[i-j]
            # inputpreva[j]=preva[i-j]
        lrdata.append(np.hstack([inputtype,inputreward,inputvalue, inputpreverr, inputprevd, inputpreva]))
    lrdata=np.array(lrdata)

    x=lrdata
    y=encodedtrialtype[num_steps:][:len(lrdata)]
    x=np.array(x)
    y=np.array(y)

    totalacc=[]
    for i in range(numsamples):
        splitpoint=np.random.randint(0,len(lrdata)-500)
        trainx=np.concatenate((x[:splitpoint],x[splitpoint+100:]),axis=0)
        trainy=np.concatenate((y[:splitpoint],y[splitpoint+100:]),axis=0)
        valx=x[splitpoint:splitpoint+100]
        valy=y[splitpoint:splitpoint+100]
        testx=x[splitpoint+100:splitpoint+500]
        testy=y[splitpoint+100:splitpoint+500]
        # use distance and angle to predict cur type
        model = LogisticRegression(fit_intercept=False)
        model.fit(trainx,trainy)
        # print("val acc: %.2f" % model.score(valx, valy, sample_weight=None))
        # print("acc: %.2f" % model.score(testx, testy, sample_weight=None))
        totalacc.append(model.score(testx, testy, sample_weight=None))
        # totalacc.append(model.score(testx, testy, sample_weight=None)-(1-np.count_nonzero(testy)/len(testy))) # above chance
    # print('mean acc ',totalacc/numsamples)
    totalaccs.append(totalacc)

totalaccs=np.array(totalaccs)

with initiate_plot(2,2,300) as f:
    ax=f.add_subplot(111)
    yerr=np.std(totalaccs,axis=1)
    # yerr=percentile_err2d(totalaccs,axis=1,percentile=95).T
    ax.errorbar(list(range(1,maxnumsteps+1)),np.mean(totalaccs,axis=1),yerr=yerr)
    plt.plot([1,maxnumsteps],[0.8881578947368421,0.8881578947368421], label='chance level')
    quickspine(ax)
    ax.set_xticks([1,maxnumsteps-1])
    ax.set_xticklabels([-1,-maxnumsteps+1])
    ax.set_xlabel('number of previous history')
    ax.set_ylabel('prediction accuracy')
    ax.text(1,1,'skip prediction acc sampled from {} samples, \nerrorbar shows std'.format(numsamples))
    ax.legend()
    # quicksave('acc vs numhistory')



# vary variable combinations  ---------------------

def combination(x):
    res=[]
    for c in range(len(x)): # count
        for starti in range(len(x)-c):
            i=starti
            curres=[]
            while len(curres)<c+1 and i<len(x):
                curres.append(x[i])
                i+=1
            if curres:
                res.append(curres)
    return res

dependentvarnames=['skip','rewarded','value']
dependentvarvalues=[encodedtrialtype,rewarded,totalvalues]
dependentvarsets=combination([0,1,2])

num_steps=6
numsamples=100
totalaccs=[]
totalcoefs=[]
for paramset in dependentvarsets:
        lrdata=[]
        for i in range(num_steps, len(encodedtrialtype)-num_steps):
            inputtype=np.zeros(num_steps)
            inputreward=np.zeros(num_steps)
            inputvalue=np.zeros(num_steps)
            inputpreverr=np.zeros(num_steps)
            inputprevd=np.zeros(num_steps)
            inputpreva=np.zeros(num_steps)
            for j in range(0,num_steps):
                if 2 in paramset:
                    inputvalue[j]=totalvalues[i-j]            
            for j in range(1,num_steps):
                if 0 in paramset:
                    inputtype[j]=encodedtrialtype[i-j]
                if 1 in paramset:
                    inputreward[j]=rewarded[i-j]
                # inputpreverr[j]=preverr[i-j]
                # inputprevd[j]=prevd[i-j]
                # inputpreva[j]=preva[i-j]
            lrdata.append(np.hstack([inputtype,inputreward,inputvalue, inputpreverr, inputprevd, inputpreva]))
        lrdata=np.array(lrdata)

        x=lrdata
        y=encodedtrialtype[num_steps:][:len(lrdata)]
        x=np.array(x)
        y=np.array(y)
        totalcoef=[]
        totalacc=[]
        for i in range(numsamples):
            splitpoint=np.random.randint(0,len(lrdata)-300)
            trainx=np.concatenate((x[:splitpoint],x[splitpoint+100:]),axis=0)
            trainy=np.concatenate((y[:splitpoint],y[splitpoint+100:]),axis=0)
            valx=x[splitpoint:splitpoint+100]
            valy=y[splitpoint:splitpoint+100]
            testx=x[splitpoint+100:splitpoint+300]
            testy=y[splitpoint+100:splitpoint+300]
            # use distance and angle to predict cur type
            model = LogisticRegression(fit_intercept=False)
            model.fit(trainx,trainy)
            # print("val acc: %.2f" % model.score(valx, valy, sample_weight=None))
            # print("acc: %.2f" % model.score(testx, testy, sample_weight=None))
            totalcoef.append(model.coef_.flatten())
            totalacc.append(model.score(testx, testy, sample_weight=None))
            # totalacc.append(model.score(testx, testy, sample_weight=None)-(1-np.count_nonzero(testy)/len(testy))) # above chance
        # print('mean acc ',totalacc/numsamples)
        totalaccs.append(totalacc)
        totalcoefs.append(totalcoef)
totalaccs=np.array(totalaccs)
totalcoefs=np.array(totalcoefs)

# acc
with initiate_plot(3,2,300) as f:
    ax=f.add_subplot(111)
    quickspine(ax)   
    ax.bar(list(range(len(dependentvarsets))),np.mean(totalaccs,axis=1),yerr=np.std(totalaccs,axis=1))
    ax.set_xticks(list(range(len(dependentvarsets))))
    ax.set_xticklabels([''.join([str(ss) for ss in s]) for s in dependentvarsets])
    ax.set_xlabel('dependent variable sets')
    ax.set_ylabel('skip prediction acc')
    ax.text(0,-1,'0,1,2 stands for type, reward, and value')
    # quicksave('dependent variable sets acc')


# coefs
nbyn=int(len(dependentvarsets)**0.5) if len(dependentvarsets)**0.5//1==0 else int(len(dependentvarsets)**0.5//1+1)
nrow=nbyn
ncol=len(dependentvarsets)//nrow if (len(dependentvarsets)//nrow)//1==0 else int(np.ceil(len(dependentvarsets)/nrow))
fig,axs=plt.subplots(nrow, ncol,figsize=(2*nrow,2*ncol) ,dpi=300,sharey=True)
for i in range(nrow):
    for j in range(ncol):
        cur=i*ncol+j
        ax=axs[i,j]
        quickspine(ax)
        coefmu=np.mean(totalcoefs[cur],axis=0)
        coefstd=np.std(totalcoefs[cur],axis=0)
        ax.bar(list(range(num_steps)),coefmu[:num_steps],yerr=coefstd[:num_steps], label='skipped')
        ax.bar(np.arange(num_steps,num_steps*2)+3*1,coefmu[num_steps:num_steps*2],yerr=coefstd[num_steps:num_steps*2],label='rewarded')
        ax.bar(np.arange(num_steps*2,num_steps*3)+3*2,coefmu[num_steps*2:num_steps*3],yerr=coefstd[num_steps*2:num_steps*3],label='value')   
        if i!=nrow-1:
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            ax.set_xticks([num_steps//2+1,num_steps+num_steps//2+4,num_steps*2+num_steps//2+7])
            ax.set_xticklabels(dependentvarnames)
            ax.set_xlabel('previous ith trial type')
        if j==0:
            ax.set_ylabel('coef')
        if i==j==0:
            ax.legend(loc='upper right', bbox_to_anchor=(-1,0))
    # quicksave('dependent combination coef')





with initiate_plot(2,2,300) as f:
    ax=plt.subplot(nbyn, int(np.ceil(len(dependentvarsets)//nbyn)),len(dependentvarsets))

    ax.errorbar(list(range(1,maxnumsteps+1)),np.mean(totalaccs,axis=1),yerr=np.std(totalaccs,axis=1))
    plt.plot([1,maxnumsteps],[0.8881578947368421,0.8881578947368421], label='chance level')
    quickspine(ax)
    ax.set_xticks([1,maxnumsteps-1])
    ax.set_xticklabels([-1,-maxnumsteps+1])
    ax.set_xlabel('number of previous history')
    ax.set_ylabel('prediction accuracy')
    ax.text(1,1,'skip prediction acc sampled from {} samples, \nerrorbar shows std'.format(numsamples))
    ax.legend()



# % predicted percentage of skip based on values ---------------------

# value
x=tarskipvalues+tarnormvalues[:len(tarskipvalues)]
y=[1]*len(tarskipvalues)+[0]*len(tarskipvalues)
rawx=np.array(normalizematrix(x)).reshape(-1, 1)
rawy=np.array(y).reshape(-1, 1)

totalacc=0
for i in range(10):
    xy=np.hstack([rawx,rawy])
    np.random.shuffle(xy)
    x,y=xy[:,0].reshape(-1, 1),xy[:,1].reshape(-1, 1)
    splitpoint=len(xy)-200
    trainx=np.concatenate((x[:splitpoint],x[splitpoint+100:]),axis=0)
    trainy=np.concatenate((y[:splitpoint],y[splitpoint+100:]),axis=0)
    valx=x[splitpoint:splitpoint+100]
    valy=y[splitpoint:splitpoint+100]
    testx=x[splitpoint+100:splitpoint+200]
    testy=y[splitpoint+100:splitpoint+200]
    # use distance and angle to predict cur type
    model = LogisticRegression(fit_intercept=True)
    model.fit(trainx,trainy)
    print("val acc: %.2f" % model.score(valx, valy, sample_weight=None))
    totalacc+=model.score(testx, testy, sample_weight=None)
    print("acc: %.2f" % model.score(testx, testy, sample_weight=None))
    totalacc+=model.score(testx, testy, sample_weight=None)
    print('coef',model.coef_)
# print('mean acc ',totalacc/10 )
model.coef_
model.intercept_
xs=np.linspace(0,1,20)
plt.plot(xs,model.intercept_[0]+model.coef_[0]*xs)
plt.ylim(0,1)
plt.plot(xs,np.ones_like(xs)*0)

len(x)
np.count_nonzero(y)
np.count_nonzero(xy[:,1])

# prediction
with initiate_plot(2,1,300) as f:
    ax=f.add_subplot(111)
    quickspine(ax)
    n=len(testy)
    plt.bar(list(range(n)),testy.flatten(),label='ground truth skips')
    plt.bar(list(range(n)),-model.predict(testx),label='predicted skips')
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlim(0,n)
    ax.set_xlabel('test set trials')
    ax.legend(loc='upper right', bbox_to_anchor=(0,0))
    ax.text(-100,1,'prediction from {} step history'.format(str(num_steps)))



# vary 2 param ---------------------
'''
we found that process noise w and obs noise w are possitively correlated,
test if this is true
by varying the two parameters
'''
# load the data
theta,_,_=process_inv("Z:/bruno_pert/cmafull_b_pert", removegr=False)
with open('Z:/victor_pert/packed','rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df=df[df.target_r>200]
states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)


# vary the two params
midpoint=np.array(theta).T
gridreso=15
dx,dy=np.zeros((11)),np.zeros((11))
dx[3]=1 # pro noise
dy[5]=1 # obs noise
X,Y=np.linspace(0,2,gridreso), np.linspace(0,2,gridreso)
paramls=[]
for i in range(gridreso):
    for j in range(gridreso):
        htheta=torch.tensor(dx*X[i]+dy*Y[j]+midpoint).float().view(-1,1)
        paramls.append(htheta)


Z=ray.get([getlogll.remote(each) for each in paramls])
with open('distinguishparamZnoisecost3finer19', 'wb+') as f:
        pickle.dump((paramls,Z), f, protocol=pickle.HIGHEST_PROTOCOL)


# with open('distinguishparamZnoisecost3finer19', 'rb') as f:
#     paramls,Z= pickle.load(f) 




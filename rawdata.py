# ploting related to just raw data, no irc or agent involved

import pickle
from plot_ult import plotpert, quickoverhead, smooth
from FireflyEnv import ffacc_real
from matplotlib.pyplot import xlabel
import pandas as pd
import numpy as np
from scipy.ndimage.measurements import label
import torch
from scipy.signal import medfilt
from scipy.stats import norm
import neo
from pathlib import Path
from rult import *
from monkey_functions import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# check stop distribution given a task

densities=sorted(pd.unique(df.floor_density))
for dens in densities:
    print('process data')
    states, actions, tasks=monkey_data_downsampled(df[df.floor_density==dens],factor=0.0025)
    print('done process data')

    ind=torch.randint(low=0,high=len(tasks),size=(1,))
    indls=similar_trials(ind, tasks, actions)
    indls=indls[:10]



# check error rate and radiual err with density for 4 monkey
monkeys=['bruno', 'schro','victor','q']

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
        ax.bar(list(range(len((v[0])))), [1-vv for vv in v[1]], label=k,color='k')
        ax.legend()
        ax.set_xlabel('density')
        ax.set_ylabel('miss rate')
        ax.set_xticks(list(range(len((v[0])))))
        ax.set_xticklabels(v[0])
        ax.set_yticks([0,np.round(1-min(v[1]),decimals=1)+0.1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

for k,v in mk2beh.items():
    with initiate_plot(4,3) as fig:
        ax=fig.add_subplot(111)
        ax.bar(list(range(len((v[0])))), v[2], label=k,color='k')
        ax.legend()
        ax.set_xlabel('density')
        ax.set_ylabel('radial error [cm]')
        ax.set_xticks(list(range(len((v[0])))))
        ax.set_xticklabels(v[0])
        datamax=max(v[2])
        roundmax=np.round(max(v[2]),decimals=-1)
        plotmax=roundmax if roundmax>datamax else roundmax+10
        ax.set_yticks([0,plotmax])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)




# multi monkey multi density bar, not working 
monkeynames=['bruno', 'schro']
performance=[mk2beh[i][2] for i in monkeynames]
performance[1].append(0)
ax=multimonkeyerr(densities,performance,monkeynames) # performance: n monkey x m density
ax.get_figure()


# check what victor do in pert
print('loading data')
datapath=Path("Z:\\victor_pert\\victor_pert_ds")
with open(datapath,'rb') as f:
    df = pd.read_pickle(f)

plotmetatrend(df.iloc[:600])
df.keys()
plt.plot(smooth(df.rewarded,30))
plt.plot(smooth(df.trial_dur,30))
plt.plot(smooth(df.relative_radius_end,30))
plt.plot(smooth(df.target_r-df.relative_radius_end,30))
plt.plot(smooth(df.perturb_vpeak,30))
plt.plot(smooth(df.relative_angle_end,30))
plt.plot((df.perturb_start_time))
plt.plot((df.perturb_start_time_ori-df.perturb_start_time))
df.iloc[2]
plt.scatter(df.perturb_vpeak,df.relative_radius_end,alpha=0.2)
plt.scatter(df.perturb_start_time,df.relative_radius_end,alpha=0.2)


x=np.asfarray(df.perturb_vpeak)
y=np.asfarray(df.relative_radius_end)


scatter_hist(sdf[:9999].perturb_wpeak, sdf[:9999].relative_angle_end)
scatter_hist(bdf[:9999].perturb_wpeak, bdf[:9999].relative_angle_end)
scatter_hist(df.perturb_wpeak, df.relative_angle_end)

# pert peak. victor shows a smell, larger per larger err
scatter_hist(sdf[:9999].perturb_vpeak, sdf[:9999].relative_radius_end)
scatter_hist(bdf[:9999].perturb_vpeak, bdf[:9999].relative_radius_end)
scatter_hist(df.perturb_vpeak, df.relative_radius_end)

# pert start time. victor shows larger err when pert start early
scatter_hist(sdf[:9999].perturb_start_time, sdf[:9999].relative_radius_end)
scatter_hist(bdf[:9999].perturb_start_time, bdf[:9999].relative_radius_end)
scatter_hist(df.perturb_start_time, df.relative_radius_end)


scatter_hist(df[~df.perturb_start_time.isnull()].perturb_vpeak+df[~df.perturb_start_time.isnull()].perturb_wpeak, df[~df.perturb_start_time.isnull()].relative_radius_end)



plt.plot(smooth(df[~df.perturb_start_time.isnull()].relative_radius_end/(abs(df[~df.perturb_start_time.isnull()].perturb_vpeak)+abs(df[~df.perturb_start_time.isnull()].perturb_wpeak)),100))

# if we assume pertpeak affect err, this trend is decreasing
res = stats.linregress((abs(df[~df.perturb_start_time.isnull()].perturb_vpeak)+abs(df[~df.perturb_start_time.isnull()].perturb_wpeak)), df[~df.perturb_start_time.isnull()].relative_radius_end)



states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)


ind=np.random.randint(low=100,high=300)
df.iloc[ind]

env.reset(phi=phi,theta=theta,goal_position=tasks[ind])
epactions,epbliefs,epbcov,epstateswo=run_trial(agent=agent,env=env,given_state=None,given_action=actions[ind])

env.reset(phi=phi,theta=theta_final,goal_position=tasks[ind])
epactions,epbliefs,epbcov,epstates=run_trial(agent=agent,env=env,given_state=states[ind],given_action=actions[ind])
quickoverhead({'pert':epstates,'no pert':epstateswo})




df=datawash(df)
df=df[df.category=='normal']
df[~df.perturb_start_time.isnull()]
print('process data')
bs, ba, bt=monkey_data_downsampled(df,factor=0.0025)
print('done process data')

# check two mk difference in normal
densities=[0.0001, 0.0005, 0.001, 0.005]
filenamecommom=Path('Z:\\schro_normal')
with open(filenamecommom/'packed', 'rb') as f:
    dfs = pickle.load(f)
    dfs=datawash(dfs)
    dfs=dfs[dfs.category=='normal']
filenamecommom=Path('Z:\\bruno_normal')
with open(filenamecommom/'packed', 'rb') as f:
    dfb = pickle.load(f)
    dfb=datawash(dfb)
    dfb=dfb[dfb.category=='normal']

densityacc_s = [len(dfs[(dfs.floor_density==i) & (dfs.rewarded)])/(len(dfs[dfs.floor_density==i])+1e-8) for i in densities]
densityacc_b = [len(dfb[(dfb.floor_density==i) & (dfb.rewarded)])/(len(dfb[dfb.floor_density==i])+1e-8) for i in densities]
twodatabar(densityacc_s,densityacc_b,labels=['Schro','Bruno'],shift=0.4,width=0.5,ylabel='sucess rate',xlabel=densities,xname='density')

densityrerr_s = [np.mean(dfs[(dfs.floor_density==i)].relative_radius_end) for i in densities]
densityrerrerr_s = [np.std(dfs[(dfs.floor_density==i)].relative_radius_end) for i in densities]
densityrer_b = [np.mean(dfb[(dfb.floor_density==i)].relative_radius_end) for i in densities]
densityrerrerr_b = [np.std(dfb[(dfb.floor_density==i)].relative_radius_end) for i in densities]
twodatabar(densityrerr_s,densityrer_b,err1=None,err2=None,labels=['Schro','Bruno'],shift=0.4,width=0.5,ylabel='radial error [cm]',xlabel=densities,xname='density')

len(dfs[(dfs.rewarded)])/len(dfs)
len(dfb[(dfb.rewarded)])/len(dfb)

plot(smooth(dfs.rewarded,30))
plot(smooth(dfb.rewarded,30))


# check two mk difference in pert,use bruno index 0 as test
print('loading data')
datapath=Path("Z:\\bruno_pert\\packed_bruno_pert")
savename=datapath.parent/('cmafull_'+datapath.name)
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df[~df.perturb_start_time.isnull()]
print('process data')
bs, ba, bt=monkey_data_downsampled(df,factor=0.0025)
print('done process data')

print('loading data')
datapath=Path("Z:\\schro_pert\\packed_schro_pert")
savename=datapath.parent/('cmafull_'+datapath.name)
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df[~df.perturb_start_time.isnull()]
print('process data')
ss, sa, st=monkey_data_downsampled(df,factor=0.0025)
print('done process data')


ind=np.random.randint(0,min(len(bt),len(st)))
thistask=bt[ind]
thisaction=ba[ind][0]
btrials=similar_trials(ind,bt,ba)
strials=similar_trials2this(st,sa,thistask,thisaction)


for bind in btrials:
    plt.plot(ba[bind][:,1],color='blue',alpha=0.6)
for sind in strials:
    plt.plot(sa[sind][:,1],color='red',alpha=0.6)

for bind in btrials:
    plt.plot(ba[bind][:,0],color='tab:blue',alpha=0.6)
for sind in strials:
    plt.plot(sa[sind][:,0],color='orange',alpha=0.6)

fig = plt.figure(figsize=[3, 3])
ax = fig.add_subplot()
for bind in btrials:
    ax.plot(bs[bind][:,0],bs[bind][:,1],color='tab:blue',alpha=0.6, label='Bruno')
for sind in strials:
    ax.plot(ss[sind][:,0],ss[sind][:,1],color='orange',alpha=0.6, label='Schro')
plt.xlabel('world x')
plt.ylabel('world y')
goalcircle = plt.Circle((bt[btrials[0]][0],bt[btrials[0]][1]),0.13, color='y', alpha=0.5)
ax.add_patch(goalcircle)
plt.axis('scaled')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())





# my testing
from pathlib import Path
folderpath=Path("D:\mkdata\\q_pert")
sessions=[x for x in folderpath.iterdir() if x.is_dir()]
# convert and downsample, saving
for eachsession in sessions:
    ext=MonkeyDataExtractor(folder_path=eachsession)
    trajectory=ext()
    # trajectory.to_pickle(eachsession.parent/(str(eachsession)+'_full'))
    dfdownsample(trajectory,eachsession.parent/(str(eachsession)+'_ds'))
    print('file saved')
# pack all ds data together
folderpath=Path("D:\mkdata\\q_normal")
sessions=[x for x in folderpath.iterdir() if x.is_file()]
packed=None
for eachsession in sessions:
    with open(eachsession, 'rb') as f:
        df = pickle.load(f)
        packed=pd.concat([packed,df])
packed.to_pickle('packed')
# load packed
folderpath=Path("D:\mkdata\\q_pert")
with open(folderpath/'packed', 'rb') as f:
        df = pickle.load(f)



# rewarded curve
data=list(trajectory.rewarded)
data=[1 if each else 0 for each in data]
x = [i for i in range(len(data))]
y = data
plt.plot(x, smooth(y,20), lw=2,label='reward')


# reward/trial curve
rewardrate=[r/d for r,d in zip(list(trajectory.rewarded),list(trajectory.trial_dur))]
x = [i for i in range(len(data))]
y = rewardrate
plt.plot(x, smooth(y,20), lw=2,label='reward rate')



# cat curve
data=list(trajectory.category)
vocab={'skip':0,'normal':1,"crazy":2,'lazy':3,'wrong_target':4}
data=[vocab[each] for each in data]
x = [i for i in range(len(data))]
y = data
# plt.plot(y)
plt.plot(x, smooth(y,20), lw=2,label='trial type')
plt.xlabel('trial number')
plt.ylabel('reward, reward rate and trial type')
plt.legend()
print(vocab)


x = [i for i in range(len(trajectory))]
y = trajectory.perturb_wpeak*trajectory.perturb_vpeak
plt.plot(x, smooth(y,20), lw=2,label='reward rate')



# new trial data function
'''
    we need:
    given mk data, return what agent will do given mk situation

    given task data, return what agent do on himself

'''


# first make something that can take perturbation




# print(result['task'][trial_i])
if input['use_mk_data']:
    env.reset(phi=input['phi'],theta=input['theta'],
    goal_position=result['task'][trial_i],
    vctrl=input['mkdata']['actions'][trial_i][0][0], 
    wctrl=input['mkdata']['actions'][trial_i][0][1])
else:
    env.reset(phi=input['phi'],
    theta=input['theta'],
    goal_position=result['task'][trial_i])
# prepare trial var
epbliefs=[]
epbcov=[]
epactions=[]
epstates=[]






env.trial_timer

plt.plot(df.iloc[ind].perturb_v/200)
plt.plot(df.iloc[ind].perturb_w/180*pi)


plt.plot(down_sampling(df.iloc[ind].perturb_v/200,0.0012,0.1))
df[~df.perturb_start_time.isnull()]


ind=torch.randint(size=(),low=0,high=1000,).item()
given_state=states[ind]
given_action=actions[ind]

pertv=0.5
pertw=0.5
given_state[:,3]+=torch.ones(given_state.shape[0])*pertv
given_state[:,4]+=torch.ones(given_state.shape[0])*pertw




env.reset(phi=phi,theta=theta,goal_position=tasks[ind])
epactions,epbliefs,epbcov,epstates=run_trial(agent=agent,env=env,given_state=given_state,given_action=given_action)

env.reset(phi=phi,theta=theta,goal_position=tasks[ind])
epactions,epbliefs,epbcov,epstates=run_trial(agent=agent,env=env,given_state=given_state,given_action=None)

env.reset(phi=phi,theta=theta,goal_position=tasks[ind])
epactions,epbliefs,epbcov,epstates=run_trial(agent=agent,env=env,given_state=None,given_action=given_action)

env.reset(phi=phi,theta=theta,goal_position=tasks[ind])
epactions,epbliefs,epbcov,epstates=run_trial(agent=agent,env=env,given_state=None,given_action=None)


#  state same
quickoverhead({'state':epstates,'belief':epbliefs})

#  action same
plt.plot(given_action[:,0])
plt.plot(given_action[:,1])
plt.plot(torch.stack(epactions)[:,0])
plt.plot(torch.stack(epactions)[:,1])

env.s
env.observations(env.s)


env.reset(phi=phi,theta=phi)
env.noise_scale
env.theta
env.phi

theta=torch.tensor([[0.5],
        [1.5],
        [0.1],
        [0.1],
        [0.9],
        [0.9],
        [0.1300],
        [0.4641],
        [0.8830],
        [0.4254],
        [0.2342]])



# conclusion, this is not a good way
# either state-action pair, or nothing
# should add obs input. the obs is the noise or bias.;


env.trial_timer
env.debug=1
env.obs_traj=torch.ones(40,2)
env.pro_traj=torch.ones(99)

env.episode_len=70


v=actions[ind][:,0][1:]
w=actions[ind][:,1][1:]
pertv=down_sampling(df.iloc[ind].perturb_v/200,0.0012,0.1)
pertw=down_sampling(df.iloc[ind].perturb_w/180*pi,0.0012,0.1)
v,w,pertv,pertw=torch.tensor(v),torch.tensor(w),torch.tensor(pertv).float(),torch.tensor(pertw).float()
overlap=min(len(v),len(pertv))
pertv=pertv[:overlap] +v[:overlap]
pertw=pertw[:overlap] +w[:overlap]
v,w=v[:overlap],w[:overlap]
plotpert_fill(v,pertv, w, pertw, ax=None,alpha=0.8)





fig=plt.figure()
ax=fig.add_subplot(111)
plt.plot(given_state[:,0],given_state[:,1])
plt.plot(torch.cat(epstates,1).t()[:,0],torch.cat(epstates,1).t()[:,1])
plt.plot(torch.cat(epbliefs,1).t()[:,0],torch.cat(epbliefs,1).t()[:,1])
goal=plt.Circle(tasks[ind],0.13)
ax.add_patch(goal)
ax.axis('equal')

def pert_overhead(state,pertstate,belief, pertbelief, ax=None,alpha=0.7):
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if ax is None:
            ax = fig.add_subplot(111)
        ax.plot(torch.cat(state,1).t()[:,0],torch.cat(state,1).t()[:,1], label='no pert')
        ax.plot(torch.cat(pertstate,1).t()[:,0],torch.cat(pertstate,1).t()[:,1],label='pert')
        # ax.plot(torch.cat(belief,1).t()[:,0],torch.cat(belief,1).t()[:,1])
        # ax.plot(torch.cat(pertbelief,1).t()[:,0],torch.cat(pertbelief,1).t()[:,1])
        goal=plt.Circle(tasks[ind],0.13,color=color_settings['goal'], alpha=alpha,label='target')
        ax.add_patch(goal)
        ax.axis('equal')
        ax.set_xlabel('world x')
        ax.set_ylabel('world y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # legend and label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    return ax

# rand a pert trial
ind=torch.randint(size=(),low=0,high=1000,).item()

pertv=down_sampling(df.iloc[ind].perturb_v/200,0.0012,0.1)
pertw=down_sampling(df.iloc[ind].perturb_w/180*pi,0.0012,0.1)
torch.tensor(pertv).float()
torch.tensor(pertw).float()
# pert=torch.stack([torch.tensor(pertv).float(),torch.tensor(pertw).float()]).t()
# pert=torch.ones(99,2)*0.2

theta=torch.tensor([[0.5],
        [1.5],
        [0.1],
        [0.1],
        [0.9],
        [0.9],
        [0.1300],
        [0.1],
        [0.1],
        [0.4254],
        [0.2342]])

env.reset(phi=phi,theta=theta,goal_position=tasks[ind])
epactions,epbliefs,epbcov,epstates=run_trial(agent=agent,env=env)

env.reset(phi=phi,theta=theta,goal_position=tasks[ind])
env.pro_traj=pert
epactionpert,epbliefspert,epbcovpert,epstatespert=run_trial(agent=agent,env=env)

pert_overhead(epstates,epstatespert,1,2)

plt.show()
plt.plot(torch.stack(epactions,1).t(), label='no pert')
plt.plot(torch.stack(epactionpert,1).t(),label='pert')
plt.xlabel('time dt')
plt.ylabel('control magnitude')
plt.legend()

quickoverhead({'state without pert':epstates, 'state with pert': epstatespert, 'belief with pert':epbliefspert})

fig=plt.figure()
ax=fig.add_subplot(111)
plt.plot(torch.cat(epstates,1).t()[:,0],torch.cat(epstates,1).t()[:,1])
plt.plot(torch.cat(epstatespert,1).t()[:,0],torch.cat(epstatespert,1).t()[:,1])
plt.plot(torch.cat(epbliefs,1).t()[:,0],torch.cat(epbliefs,1).t()[:,1])
goal=plt.Circle(tasks[ind],0.13)
ax.add_patch(goal)
ax.axis('equal')

theta=torch.tensor([[0.5],
        [1.5],
        [0.9],
        [0.9],
        [0.1],
        [0.1],
        [0.1300],
        [0.1],
        [0.1],
        [0.4254],
        [0.2342]])
     
env.reset(phi=phi,theta=theta,goal_position=tasks[ind])
epactions,epbliefs,epbcov,epstates=run_trial(agent=agent,env=env)

env.reset(phi=phi,theta=theta,goal_position=tasks[ind])
env.pro_traj=pert
epactionpert,epbliefspert,epbcovpert,epstatespert=run_trial(agent=agent,env=env)

pert_overhead(epstates,epstatespert,1,2)

plt.show()
plt.plot(torch.stack(epactions,1).t(), label='no pert')
plt.plot(torch.stack(epactionpert,1).t(),label='pert')
plt.xlabel('time dt')
plt.ylabel('control magnitude')
plt.legend()


fig=plt.figure()
ax=fig.add_subplot(111)
plt.plot(torch.cat(epstates,1).t()[:,0],torch.cat(epstates,1).t()[:,1])
plt.plot(torch.cat(epstatespert,1).t()[:,0],torch.cat(epstatespert,1).t()[:,1])
plt.plot(torch.cat(epbliefs,1).t()[:,0],torch.cat(epbliefs,1).t()[:,1])
goal=plt.Circle(tasks[ind],0.13)
ax.add_patch(goal)
ax.axis('equal')





def run_trial_pert(agent,env,pert):

    def _collect():
        epactions.append(action)
        epbliefs.append(env.b)
        epbcov.append(env.P)
        epstates.append(env.s)
        if t<len(pert):
            epactionpert.append(action+pert[t])

    # saves
    epactions,epbliefs,epbcov,epstates, epactionpert=[],[],[],[],[]
    with torch.no_grad():
            done=False
            t=0
            while not done:
                action = agent(env.decision_info)[0]
                _collect()
                _,_,done,_=env.step(torch.tensor(action).reshape(1,-1)) 
                t+=1
    return epactions,epbliefs,epbcov,epstates,epactionpert




plt.bar([i-0.2 for i in range(3)],[0.5990795885219274,.6362665242355454,0.672463768115942],0.5)
plt.bar([i+0.2 for i in range(3)],[0.6400537233248769,.721852642148864,0.7812195386115022],0.5)
plt.xticks=[1,2,3]



def barpertacc(accs,trialtype, ax=None, label=None,shift=0,width=0.4):
    with initiate_plot(6, 4, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if ax is None:
            ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar([i+shift for i in range(len(accs))], accs,width,label=label)
        # title and axis names
        ax.set_ylabel('trial reward rate')
        ax.set_xticks([i for i in range(len(accs))])
        ax.set_xticklabels(trialtype, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
    return ax

# pert and non pert reward rate
trialtypes=['pert','both','non pert']
a=barpertacc([0.5990795885219274,.6362665242355454,0.672463768115942],trialtypes,label='schro',shift=-0.2)
barpertacc([0.6400537233248769,.721852642148864,0.7812195386115022],trialtypes,label='bruno',shift=0.2,ax=a)
a.get_figure()

print('all trial {}, reward% '.format(len(df[df.rewarded])/len(df)))
print('pert trial {}, reward% '.format(len(df[~df.perturb_start_time.isnull()])/len(df[~df.perturb_start_time.isnull()])))
print('non pert trial {}, reward% '.format(len(df[df.perturb_start_time.isnull() & df.rewarded])/len(df[df.perturb_start_time.isnull()])))




# pert and non pert radiu acc
trialtypes=['pert','non pert']
a=barpertacc([63.36783413567739,55.58846184159118],trialtypes,label='schro',shift=-0.2)
barpertacc([57.37014305912993,43.945449613900934],trialtypes,label='bruno',shift=0.2,ax=a)
a.get_figure()

np.mean([i[-1] for i in df[~df.perturb_start_time.isnull()].relative_radius])
np.mean([i[-1] for i in df[df.perturb_start_time.isnull()].relative_radius])



ind=np.random.randint(0,len(df))

df.iloc[ind]


ind=torch.randint(low=0,high=1111,size=(1,))
indls=similar_trials(ind, tasks, actions,ntrial=10)



with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(111)
    for ind in indls:
        ax.plot(df.iloc[ind].pos_x,df.iloc[ind].pos_y, label='path',color='tab:blue',alpha=alpha)
    goal=plt.Circle([df.iloc[ind].target_x,df.iloc[ind].target_y],65,facecolor=color_settings['goal'],edgecolor='none', alpha=alpha,label='target')
    ax.plot(0.,0., "*", color='black',label='start')
    ax.add_patch(goal)
    ax.axis('equal')
    ax.set_xlabel('world x [cm]')
    ax.set_ylabel('world y [cm]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # legend and label
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),loc=2, prop={'size': 6})


# tasktd=[]
# for ind in indls:
#     tasktd.append(tasks[ind])

    
input={
        'agent':agent,
        'theta':theta,
        'phi':phi,
        'env': env,
        'num_trials':len(indls),
        'mkdata':{
                        },                      
        'use_mk_data':False
    }
with suppress():
        res=trial_data(input)



with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(111)
    for i in range(len(indls)):
        ax.plot(-1*res['agent_states'][i][:,1]*200,res['agent_states'][i][:,0]*200, label='path',color='tab:blue',alpha=alpha)
    goal=plt.Circle([res['task'][0][1]*-200,res['task'][0][0]*200],65,facecolor=color_settings['goal'],edgecolor='none', alpha=alpha,label='target')
    ax.plot(0.,0., "*", color='black',label='start')
    ax.add_patch(goal)
    ax.axis('equal')
    ax.set_xlabel('world x [cm]')
    ax.set_ylabel('world y [cm]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # legend and label
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),loc=2, prop={'size': 6})



with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(111)
    for i in range(len(indls)):
        ax.plot(-1*res['agent_states'][i][:,1]*200,res['agent_states'][i][:,0]*200, label='path',color='tab:blue',alpha=alpha)
    goal=plt.Circle([res['task'][0][1]*-200,res['task'][0][0]*200],65,facecolor=color_settings['goal'],edgecolor='none', alpha=alpha,label='target')
    ax.plot(0.,0., "*", color='black',label='start')
    ax.add_patch(goal)
    ax.axis('equal')
    ax.set_xlabel('world x [cm]')
    ax.set_ylabel('world y [cm]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # legend and label
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),loc=2, prop={'size': 6})

env.reset(phi=phi,theta=theta,goal_position=tasks[ind])
epactions,epbliefs,epbcov,epstates=run_trial(agent=agent,env=env,given_action=None, given_state=None)
ax=quickoverhead({'state':epstates})


tasklist=[]
for ind in indls:
    tasklist.append(tasks[ind])

def overheadindls(tasklist,ax=None):

    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111) if not ax else ax
        # ax.plot(given_state[:,0],given_state[:,1])
        for eachtask in tasklist:
            env.reset(phi=phi,theta=theta,goal_position=eachtask)
            epactions,epbliefs,epbcov,epstates=run_trial(agent=agent,env=env,given_action=None, given_state=None)
            ax.plot(torch.cat(epstates,1).t()[:,0],torch.cat(epstates,1).t()[:,1],color='tab:blue', label='path')
        # ax.plot(torch.cat(epstates,1).t()[:,0],torch.cat(epstates,1).t()[:,1], label='state')
        # ax.plot(torch.cat(epbliefs,1).t()[:,0],torch.cat(epbliefs,1).t()[:,1],label='belief')
        goal=plt.Circle(tasks[ind],0.13,color=color_settings['goal'], alpha=alpha,label='target')
        ax.add_patch(goal)
        ax.axis('equal')
        ax.set_xlabel('world x')
        ax.set_ylabel('world y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # legend and label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),loc=2, prop={'size': 6})


# run pert trial
# select a pert trial
pertdf=df[~df.perturb_start_time.isnull()]

ind=np.random.randint(low=0,high=len(pertdf))
pertdf.iloc[ind].perturb_start_time
pert=np.array([pertdf.iloc[ind].perturb_v,pertdf.iloc[ind].perturb_w])

# check different downsampling ways
# plt.plot(pertdf.iloc[ind].pos_x)
plt.plot(down_sampling(pert.T))
# plt.plot(down_sampling_(pert.T))
# down_sampling(pert.T).shape
# down_sampling_(pert.T)
# len(pertdf.iloc[ind].pos_x)

pert=np.array(down_sampling_(pert.T))
pert.shape


pertdf=trajectory[~trajectory.perturb_start_time.isnull()]
pert=np.array([trajectory.iloc[ind].perturb_v,trajectory.iloc[ind].perturb_w])



pert=np.array([df.iloc[ind].perturb_v,df.iloc[ind].perturb_w])
pert=np.array(down_sampling_(pert.T))/400


plotoverhead_mk([1])
ind=np.random.randint(low=0,high=len(df))
dfoverhead_single(df,ind)
dfctrl_single(df,ind)



trialtypes=['pert','non pert']
a=barpertacc([63.36783413567739,55.58846184159118],trialtypes,label='schro',shift=-0.2)
barpertacc([57.37014305912993,43.945449613900934],trialtypes,label='bruno',shift=0.2,ax=a)
a.get_figure()



    # count reward % by density
    for eachdensity in [0.0001, 0.0005, 0.001, 0.005]:  
        print('density {}, reward% '.format(eachdensity),len(df[df.floor_density==eachdensity][df.rewarded])/len(df[df.floor_density==eachdensity]))
    # count reward by pert
    print('all trial {}, reward% '.format(len(df[df.rewarded])/len(df)))
    print('pert trial {}, reward% '.format(len(df[~df.perturb_start_time.isnull() & df.rewarded])/len(df[~df.perturb_start_time.isnull()])))
    print('non pert trial {}, reward% '.format(len(df[df.perturb_start_time.isnull() & df.rewarded])/len(df[df.perturb_start_time.isnull()])))
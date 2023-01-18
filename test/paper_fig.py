

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
matplotlib.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'



    # check agent performance
    print('loading data')
    datapath=Path("Z:\\human\\hgroup")
    with open(datapath,'rb') as f:
        states, actions, tasks = pickle.load(f)
    print('done process data')


    agent_=TD3.load('trained_agent/humancost.zip')
    agent=agent_.actor.mu.cpu()

    ind=np.random.randint(low=0, high=len(tasks))
    env.reset(phi=phi, theta=theta, goal_position=tasks[ind],vctrl=0., wctrl=0.)
    epactions,epbliefs,epbcov,epstates=run_trial(agent,env,given_action=None, given_state=None, action_noise=0.1)
    plt.plot((torch.stack(epactions)))
    plt.ylim(-1,1.1)
    plt.show()

    plt.plot((torch.cat(epstates,1)).T[:,0],(torch.cat(epstates,1)).T[:,1])
    plt.scatter(tasks[ind][0],tasks[ind][1])
    plt.axis('equal')


    agent_=TD3.load('trained_agent/paper.zip')
    agent=agent_.actor.mu.cpu()
    # with open('paper', 'wb+') as f:
    #     pickle.dump(agent, f, protocol=pickle.HIGHEST_PROTOCOL)
    indls=np.random.randint(low=0, high=len(tasks),size=(30,))
    inputirc={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':30,
            'task':[tasks[i] for i in indls],
            'mkdata':{
                            'trial_index': indls,               
                            'task': tasks,                 
                            'actions': actions,                 
                            'states':states,},
            'use_mk_data':False
    }
    with suppress():
        res=trial_data(inputirc)
    print('percentage rewarded', sum([1 if a[-1,1]<=0.13 else 0 for a in res['agent_beliefs']])/30)
    plotoverhead(res)
    plotctrl(res)


    #---------------------------------------------------------------------
    # one monkey's belief overhead
    ind=np.random.randint(low=0, high=len(tasks))
    input={
        'agent':agent,
        'theta':theta,
        'phi':phi,
        'env': env,
        'num_trials':1,
        # 'task':tasks[:20],
        'mkdata':{
                        # 'trial_index': list(range(1)),               
                        'trial_index': ind,               
                        'task': [tasks[ind]],                  
                        'actions': [actions[ind]],                 
                        'states':[states[ind]],
                        },                      
        'use_mk_data':True
    }
    single_trial_overhead()

    #---------------------------------------------------------------------
    #  one monkey trial overhead, not for paper
    ind=torch.randint(low=100,high=300,size=(1,))
    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.plot(states[ind][:,0],states[ind][:,1], color='r',alpha=0.5)
        goalcircle = plt.Circle([tasks[ind][0],tasks[ind][1]], 0.13, color='y', alpha=0.5)
        ax.add_patch(goalcircle)
        ax.set_xlim(0,1)
        ax.set_ylim(-0.6,0.6)


    # IRC on its own---------------------------------------------------
    ind=torch.randint(low=0,high=len(tasks),size=(1,))

    indls=similar_trials(ind, tasks, actions)
    indls=indls[:10]
    with torch.no_grad():
        inputirc={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':10,
            'task':[tasks[i] for i in indls],
            'mkdata':{
                            'trial_index': indls,               
                            'task': tasks,                 
                            'actions': actions,                 
                            'states':states,},
            'use_mk_data':False
        }
        with suppress():
            resirc=trial_data(inputirc)

        inputmk={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':10,
            'mkdata':{
                            'trial_index': indls,               
                            'task': tasks,                 
                            'actions': actions,                 
                            'states':states,
                            },                      
            'use_mk_data':True
        }
        with suppress():
            resmk=trial_data(inputmk)

    plotoverheadhuman(indls,states,tasks,)
    plotoverheadhuman_compare(resirc,resmk)
    plotv_fill(indls,resirc,actions=actions)
    plotw_fill(indls,resirc,actions=actions)


    #  IRC given mk states--------------------------------------------------
    ind=torch.randint(low=0,high=len(tasks),size=(1,))
    indls=similar_trials(ind, tasks, actions,ntrial=5)
    with torch.no_grad():
        inputirc={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':5,
            'task':[tasks[i] for i in indls],
            'mkdata':{
                            'trial_index': indls,               
                            'task': tasks,                 
                            'actions': actions,                 
                            'states':states,
                            },                      
            'use_mk_data':False}
        with suppress():
            resirc=trial_data(inputirc)
        # plotoverhead(resirc)
        inputmk={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':5,
            'mkdata':{
                            'trial_index': indls,               
                            'task': tasks,                 
                            'actions': actions,                 
                            'states':states,
                            },                      
            'use_mk_data':True
        }
        with suppress():
            resmk=trial_data(inputmk)
        # plotoverhead(resmk)
        # ax=plotctrl(resmk, color=['blue','orangered'],prefix='monkey')

    # plotv(indls,resmk,actions=actions)
    # plotw(indls,resmk,actions=actions)
    plotoverheadhuman(indls,states,tasks,)
    plotoverheadhuman_compare(resirc,resmk)
    plotv_fill(indls,resmk,actions=actions)
    plotw_fill(indls,resmk,actions=actions)
    

    # IRC on its own in pert task--------------------------------------------------
    ind=np.random.randint(low=0,high=len(df))
    assert len(df)==len(tasks)
    pert=np.array([df.iloc[ind].perturb_v,df.iloc[ind].perturb_w])
    pert=np.array(down_sampling_(pert.T))/400
    pert=pert.astype('float32')
    env.terminal_vel=0.1
    env.debug=True
    env.episode_len=30
    print(df.iloc[ind].perturb_wpeak)

    irc_pert(agent, env, phi, theta,etask=tasks[ind],initv=actions[ind][0][0],initw=actions[ind][0][1],mkactions=actions[ind],pert=pert)



# load model
arg = Config()
env=ffacc_real.FireFlyPaper(arg)
env.debug=True
env.terminal_vel=0.1
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



# trial structure illustration, not in use
ind=np.random.randint(0,len(actions))
vs=actions[ind][:,0]
ws=actions[ind][:,1]
ts=np.arange(len(vs))*0.1

with initiate_plot(4, 2, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(211)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-1,1)
    ax.fill_betweenx([-1,2],-0.05,0,alpha=0.3,color='blue')
    ax.set_ylabel('forward v [a.u.]', fontsize=9)
    ax.plot(ts,vs)
    
    ax = fig.add_subplot(212)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(ts,ws)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('time [s]')
    ax.set_ylabel('angular w [a.u.]', fontsize=9)
    ax.set_ylim(-1,1)
    ax.fill_betweenx([-1,2],-0.05,0,alpha=0.3,color='blue')



# fig1, all overhead
print('loading data')
datapath=Path("Z:\\bruno_normal\\packed")
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']

overheaddf_tar(df[:1000])
overheaddf_path(df,list(range(1000)))


# fig2, training vs time
agentcommon='rerere_52000_6_6_18_30_2_'
log=[]
for i in range(1,3,1):
    # load the agent check points
    agentname=agentcommon+str(i)
    thisagent=TD3.load('trained_agent/'+agentname).actor_target.mu.cpu()
    with suppress():
        # eval
        philist=[
            torch.tensor([[0.5],
                    [pi/2],
                    [0.5],
                    [0.5],
                    [0.5],
                    [0.5],
                    [0.13],
                    [0.5],
                    [0.5],
                    [0.5],
                    [0.5],
            ]),torch.tensor([[0.5],
                    [pi/2],
                    [0.5],
                    [0.5],
                    [0.5],
                    [0.5],
                    [0.13],
                    [0.9],
                    [0.9],
                    [0.9],
                    [0.9],
            ]),torch.tensor([[0.5],
                    [pi/2],
                    [0.9],
                    [0.9],
                    [0.9],
                    [0.9],
                    [0.13],
                    [0.5],
                    [0.5],
                    [0.5],
                    [0.5],
            ]),torch.tensor([[0.5],
                    [pi/2],
                    [1.5],
                    [1.5],
                    [0.5],
                    [0.5],
                    [0.13],
                    [0.5],
                    [0.5],
                    [0.5],
                    [0.5],
            ]),torch.tensor([[0.5],
                    [pi/2],
                    [0.5],
                    [0.5],
                    [1.5],
                    [1.5],
                    [0.13],
                    [0.5],
                    [0.5],
                    [0.5],
                    [0.5],
            ])
            ]
        tasklist=[[0.9,0.1],[0.7,0.4],[0.7,-0.4],[0.6,0]]
        thislog=[]
        for p in philist:
            for t in tasklist:
                env.reset(goal_position=t, phi=p, theta=p)
                done=False
                while not done:
                    _,_,done,_=env.step(thisagent(env.decision_info))
                # thislog.append((env.trial_sum_reward-env.trial_sum_cost)/env.trial_timer)
                thislog.append((env.trial_sum_reward))
    log.append([np.mean(thislog),np.std(thislog)])
    print(agentname,log[-1] )
with initiate_plot(3, 2, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(211)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot([l[0] for l in log])


# fig 3
#  logll vs gen
with open(data_path/'longonlypacked_schro_pert', 'rb') as f:
    slog = pickle.load(f)
log=slog
optimizer=log[-1][0]
res=[l[2] for l in log]
loglls=[]
for r in res:
    genloglls=[]
    for point in r:
        genloglls.append(point[1])
    loglls.append(np.mean(genloglls))
# gen=[[i]*optimizer.population_size for i in range(optimizer.generation)]
gen=list(range(len(res)))
gen=torch.tensor(gen).flatten()
loglls=torch.tensor(loglls).flatten()

with initiate_plot(3, 2, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(211)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(gen,loglls,alpha=0.8)
    plt.xlabel('generations')
    plt.ylabel('- log likelihood')



# zoom in of last 1/4
with initiate_plot(3, 2, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(211)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(gen[-len(gen)//2:],loglls[-len(gen)//2:],alpha=0.8)
    plt.xlabel('generations')
    plt.ylabel('- log likelihood')



# slice pc ----------------------------------------------------------------------------
print('loading data')
datapath=Path("/data/bruno_pert/packed")
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df=df[df.target_r>200]
print('process data')
states, actions, tasks=monkey_data_downsampled(df[:100],factor=0.0025)
print('done process data')

def getlogll(x):
    with torch.no_grad():
        return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=0.01,num_iteration=1, states=states, samples=5,gpu=False).item()

with open('/data/bruno_pert/cmafull_b_pert', 'rb') as f:
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
npixel=9

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

xlow,xhigh=pcfinaltheta[0]-0.3,pcfinaltheta[0]+0.3
ylow,yhigh=pcfinaltheta[1]-0.3,pcfinaltheta[1]+0.3
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

a=background_data
plt.contourf(a)
plt.contourf(np.log(a))
norma=a
norma=(norma/(np.max(norma)-np.min(norma)))
norma-=np.min(norma)
plt.contourf(norma)

# value=0.1
# newa = np.where(norma>value,norma-value,0)
# plt.contourf(newa)

value=0.3
newa = np.where((1 - norma) < value,1,norma+value)
plt.contourf(newa)

norma=newa
norma=(norma/(np.max(norma)-np.min(norma)))
norma-=np.min(norma)
# plt.contourf(norma)


with initiate_plot(3, 3, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(111)
    plot_cov_ellipse(pccov[:2,:2], pcfinaltheta[:2], alpha=1,nstd=1,ax=ax, edgecolor=[1,1,1])
    im=ax.contourf(X[:,3:],Y[:,3:],-newa[:,3:],cmap='jet')
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
    ax.set_xlim(-0.05,xhigh)
    ax.set_ylim(ylow,yhigh)
    ax.plot(allthetameanspc[:,0],allthetameanspc[:,1])

    try:
        groundtruthpc=(groundtruth-allthetamu)@np.array(v)@np.linalg.inv(np.diag(s))
        ax.scatter(groundtruthpc[0],groundtruthpc[1],color='k')
    except NameError:
        pass



plt.imshow(a)
plt.imshow(a[:,3:])



# vary eig vector by eig value
print('loading data')
datapath=Path("Z:\\bruno_pert\\packed")
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df=df[df.target_r>180]
df=df[df.perturb_start_time.isnull()]
print('process data')
states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
print('done process data')

data_path=Path("Z:\\bruno_pert")
with open(data_path/'cmafull_packed_bruno_pert', 'rb') as f:
    blog = pickle.load(f)
log=blog
finalcov=torch.tensor(log[-1][0]._C)
finaltheta=torch.tensor(log[-1][0]._mean).view(-1,1)
bt=torch.cat([finaltheta[:6],finaltheta[-4:]])
finalcov = finalcov[torch.arange(finalcov.size(0))!=6] 
bc = finalcov[:,torch.arange(finalcov.size(1))!=6] 
ev, evector=torch.eig(torch.tensor(bc),eigenvectors=True)
ev=ev[:,0]
ev,esortinds=ev.sort(descending=False)
evector=evector[:,esortinds]
firstevector=evector[:,1].float().view(-1,1)
firstev=ev[1]

ind=np.random.randint(low=0,high=len(df))
# ind=461 of schro non pert data
theta_init=bt-firstevector*firstev*3
theta_init=torch.cat([theta_init[:6],finaltheta[6].view(1,1),theta_init[6:]])
theta_final=bt+firstevector*firstev*3
theta_final=torch.cat([theta_final[:6],finaltheta[6].view(1,1),theta_final[6:]])
env.terminal_vel=0.1
env.debug=True
vary_theta(agent, env, phi, theta_init, theta_final,5,etask=tasks[ind],initv=actions[ind][0][0],initw=actions[ind][0][1],mkactions=actions[ind])




# vary eig vector by eig value, in pert trial
print('loading data')
datapath=Path("Z:\\bruno_pert\\packed")
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df=df[df.target_r>180]
df=df[~df.perturb_start_time.isnull()]
print('process data')
states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
print('done process data')

data_path=Path("Z:\\bruno_pert")
with open(data_path/'cmafull_packed_bruno_pert', 'rb') as f:
    blog = pickle.load(f)
log=blog
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

ind=np.random.randint(low=0,high=len(df))
assert len(df)==len(tasks)
pert=np.array([df.iloc[ind].perturb_v,df.iloc[ind].perturb_w])
pert=np.array(down_sampling_(pert.T))/400
pert=pert.astype('float32')
env.terminal_vel=0.1
env.debug=True
env.episode_len=30
print(df.iloc[ind].perturb_wpeak)
# ind=1799 strange 

# visualize the pert in a disk
pert_disk(pert)
vary_theta(agent, env, phi, theta_init, theta_final,3,etask=tasks[ind],initv=actions[ind][0][0],initw=actions[ind][0][1],mkactions=actions[ind],pert=pert)

vary_theta_ctrl(agent, env, phi, theta_init, theta_final,3,etask=tasks[ind],initv=actions[ind][0][0],initw=actions[ind][0][1],mkactions=actions[ind],pert=pert)



# visualize pert trial irc vs monkey, single trial ----------------------------------------------------------
irc_pert(agent, env, phi, finaltheta,etask=tasks[ind],initv=actions[ind][0][0],initw=actions[ind][0][1],mkactions=actions[ind],pert=pert)





# seperate pert stregth color bar ----------------------------------------------
with initiate_plot(3,0.2) as fig:
    ax=fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.yaxis.set_ticks([])
    pertcolor=matplotlib.colors.ListedColormap(np.linspace([1,0,0,0],[1,0,0,1],200))
    data=np.linspace(0,1,100)
    data=np.broadcast_to(data, (5,100))
    ax.imshow(data,cmap=pertcolor, extent=[0,1,0,1], aspect='auto')
    ax.set_xticks([0,1])
    ax.set_xlabel('perturbation strength')




# new pert get trail

plt.plot(np.random.normal(0,1,size=pert.shape)*np.asfarray(theta[4:6]).T + pert)






def sample_trials(agent, env, theta, phi, thistask, initv=0.,initw=0., action_noise=0.1,pert=None,):
    agent_actions=[]
    agent_beliefs=[]
    agent_covs=[]
    agent_states=[]
    with torch.no_grad():
        env.reset(phi=phi,theta=theta,goal_position=thistask,vctrl=initv, wctrl=initw)

        env.obs_traj= np.random.normal(0,1,size=pert.shape)*np.asfarray(theta[4:6]).T + pert

        epbliefs=[]
        epbcov=[]
        epactions=[]
        epstates=[]
        done=False
        while not done:
            action = agent(env.decision_info)[0]
            noise=torch.normal(torch.zeros(2),action_noise)
            _action=(action+noise).clamp(-1,1)
            # if pert is not None and int(env.trial_timer)<len(pert):
            #     _action+=pert[int(env.trial_timer)]
            _,_,done,_=env.step(_action) 
            epactions.append(action)
            epbliefs.append(env.b)
            epbcov.append(env.P)
            epstates.append(env.s)
        agent_actions.append(torch.stack(epactions))
        agent_beliefs.append(torch.stack(epbliefs))
        agent_covs.append(epbcov)
        agent_states.append(torch.stack(epstates))
        estate=torch.stack(agent_states)[0,:,:,0].t()
        return_dict={
        'agent_actions':agent_actions,
        'agent_beliefs':agent_beliefs,
        'agent_covs':agent_covs,
        'estate':estate,
        # 'eaction':eaction,
        'etask':thistask,
        'theta':theta,
      }
    return return_dict



# likelihood of trial overhead. check fit quanlity and change in strategy--------------------------------
ntargets=333
ntrial=3
indls=np.random.randint(low=0, high=len(tasks),size=(ntargets,))
newindls=[similar_trials(i,tasks,ntrial=ntrial) for i in indls]
# plt.scatter(tasks[indls,0],tasks[indls,1],s=2)
plt.scatter(tasks[:,0],tasks[:,1],s=2)
for l in newindls:
    plt.scatter(tasks[l,0],tasks[l,1],s=2,color='orange')

res=llmap(newindls, agent, actions, tasks, phi, theta, env, action_var=0.01, num_iteration=1, states=states)

def llmap(indls, agent, actions, tasks, phi, theta, env, action_var=0.01, num_iteration=1, states=states, samples=5,gpu=False):
    res=[]
    for inds in indls:
        subactions=[actions[i] for i in inds]
        subtasks=[tasks[i] for i in inds]
        substates=[states[i] for i in inds]
        with torch.no_grad():
            ll=monkeyloss_(agent, subactions, subtasks, phi, theta, env, action_var=action_var,num_iteration=num_iteration, states=substates, samples=samples,gpu=gpu).item()
        res.append(ll)
    return res

with initiate_plot(3,3,300) as fig:
    ax=fig.add_subplot(111)
    c=ax.scatter(tasks[indls,0],tasks[indls,1],c=res,s=10)
    plt.colorbar(c, label='- log likelihood')

        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.axes.xaxis.set_ticks([0,0.5,1])
    ax.set_xlabel('world x [2m]')
    ax.set_ylabel('world y [2m]' )
    ax.set_aspect('equal')




# check cluster inverse res -------------------------------------------------------------------------
folder=Path('C:/Users/24455/Desktop')
finaltheta,finalcov,err=process_inv(folder/'paperhgroup')


# plot asd vs healthy theta in one same hist plot------------------------------------------------
logls=[folder/'testk8hgroup', folder/'testk8agroup']
monkeynames=[ 'paperh','papera',]

mus,covs,errs=[],[],[]
for inv in logls:
    finaltheta,finalcov, err=process_inv(inv)
    mus.append(finaltheta.view(-1))
    covs.append(finalcov)
    errs.append(err)


ax=multimonkeytheta(monkeynames, mus, covs, errs, shifts=[-0.2,0.2])
ax.set_yticks([0,1,2])
ax.get_figure()



folder=Path('Z:/human')
logls = sorted(list(folder.glob("*")), key=os.path.getatime,reverse=True)
logls=[l  for l in logls if os.path.isfile(l) and os.path.getsize(l)>>10<1024 ]


invs=[logls[2],logls[5]]
mus,covs,errs=[],[],[]
for inv in invs:
    finaltheta,finalcov, err=process_inv(inv)
    mus.append(finaltheta.view(-1))
    covs.append(finalcov)
    errs.append(err)

ax=multimonkeytheta([l.name for l in invs], mus, covs, errs)
ax.set_yticks([0,1,2])
ax.get_figure()






print('loading data')
datapath=Path("Z:\\human\\hgroup")
with open(datapath,'rb') as f:
    states, actions, tasks = pickle.load(f)
print('done process data')


#
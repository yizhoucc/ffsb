# will generate a circle of goal ( actually a gassian)
# and a cov ellipse
from scipy.stats import norm, chi2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import numpy as np
from numpy import pi
import time
import torch
import gym
from stable_baselines.ddpg.policies import LnMlpPolicy, MlpPolicy
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from stable_baselines import DDPG, TD3
import TD3_torch

import TD3_test
from FireflyEnv import ffenv_new_cord, firefly_action_cost, firefly_acc, firefly_accac, firefly_mdp, ffac_1d, ffacc_real
from reward_functions import reward_singleff
from Config import Config
arg=Config()
arg.goal_radius_range=[0.1,0.3]
from FireflyEnv.env_utils import vectorLowerCholesky
from plot_ult import add_colorbar

# # no action cost model
# env=ffenv_new_cord.FireflyAgentCenter(arg)
# model=TD3.load('trained_agent/TD_95gamma_mc_500000_0_23_22_8.zip')

# action cost model
env=firefly_action_cost.FireflyActionCost(arg)
# modelname='trained_agent/TD_action_cost_sg_700000_9_11_6_17'# non ranged working model
modelname='trained_agent/TD_action_cost_700000_2_4_8_44'# ranged model last
model=TD3.load(modelname)

# acc model
env=firefly_acc.FireflyAcc(arg)
model=TD3.load('trained_agent/.zip')
model.set_env(env)

# acc ac model
env=firefly_accac.FireflyAccAc(arg)
model=TD3.load('trained_agent/accac_final_1000000_9_11_20_25.zip')
model.set_env(env)

# acc mdp model
env=firefly_mdp.FireflyMDP(arg)
model=TD3.load('trained_agent/mdp_noise_1000000_2_9_18_8')
model.set_env(env)

# 1d real model
# easy
arg.gains_range =[0.99,1]
arg.goal_radius_range=[25,25.3]
arg.std_range = [0.5,0.51,49.5,50]
arg.mag_action_cost_range= [0.00001,0.000011]
arg.dev_action_cost_range= [0.00001,0.000012]
arg.TERMINAL_VEL = 20  
arg.DELTA_T=0.2
arg.EPISODE_LEN=50
arg.training=True
arg.presist_phi=False
arg.agent_knows_phi=False
env=ffacc_real.FireflyTrue1d_real(arg)
# hard
arg.gains_range =[0.1,5]
arg.goal_radius_range=[1,50]
arg.std_range = [0.01,2,0.01,100]
arg.mag_action_cost_range= [0.00001,0.0001]
arg.dev_action_cost_range= [0.00001,0.00005]
arg.TERMINAL_VEL = 20  
arg.DELTA_T=0.2
arg.EPISODE_LEN=50
arg.training=True
arg.presist_phi=False
arg.agent_knows_phi=False
env=ffacc_real.FireflyTrue1d_real(arg)

model=TD3_torch.TD3.load('trained_agent/1drealsysvel_1000000_12_28_22_1').actor.mu.cpu()

count=0
for i in range(100):
    pos=[]
    pos_hat=[]
    vel=[]
    vel_hat=[]
    vel_std=[]
    actions=[]
    obs=[]
    pos_std=[]
    env.reset()
    env.theta[1]=2
    env.theta[2]=50
    env.theta[4]=8
    env.unpack_theta()
    done=False
    # i=0
    # while i<40:
    while not done:
        with torch.no_grad():
            action=model(env.decision_info)[0]
            # action=torch.ones(1)
            _,_,done,_=env.step(action)
            pos.append(env.s.data[0].item())
            vel.append(env.s.data[1].item())
            pos_hat.append(env.b.data[0].item())
            vel_hat.append(env.b.data[1].item())
            actions.append(action.item())
            obs.append(env.o.item())
            vel_std.append(env.P[-1,-1].item()**0.5)
            pos_std.append(env.P[0,0].item()**0.5)
            # print(env.P, env.episode_time,env.A,env.tau_a)
            print(env.s, env.b,env.P)
            i+=1
    if env.reached_goal():
        count+=1
print(count)

# state and control plots, comparing agent and expert
for vel, vel_hat, vel_std,sysvel in zip(allvel,allvel_hat,allvel_std,allsysvel):
    plt.plot(sysvel)
    plt.plot(vel)
    plt.plot(vel_hat,'r')
    # plt.plot(obs)
    low=[a-b for a, b in zip(vel_hat,vel_std)]
    high=[a+b for a, b in zip(vel_hat,vel_std)]
    plt.fill_between(list(range(len(low))),low, high,color='orange',alpha=0.5)

for actions in allactions:
    plt.plot(actions)

for pos, pos_hat, pos_std in zip(allpos,allpos_hat,allpos_std):
    plt.plot(pos)
    plt.plot(pos_hat, color='r')
    low=[a-b for a, b in zip(pos_hat,pos_std)]
    high=[a+b for a, b in zip(pos_hat,pos_std)]
    plt.fill_between(list(range(len(low))),low, high,color='orange',alpha=0.5)

#- test svd 
# if trajectroy generated from 5 dim, then s is about 5 dim. vice versa 
param=np.random.random((100,5))
trajdata=param@np.random.random((5,17))
trajdata=trajdata+np.random.random((100,17))*0.1
_,s,_=np.linalg.svd(trajdata)
s
trajdata=np.random.random((100,17))
_,s,_=np.linalg.svd(trajdata)
s
# try the svd with our simulation----------------
m=200 # X will be m*n
T=60 # n will be 4*T
env.reset()
obs_traj=torch.distributions.Normal(0,torch.tensor(1.)).sample(sample_shape=torch.Size([100]))*50
pro_traj=torch.distributions.Normal(0,torch.tensor(1.)).sample(sample_shape=torch.Size([100]))
trial_data=sample_trials(m,phi=env.phi, goal_pos=env.goalx,timelimit=T,obs_traj=obs_traj,pro_traj=pro_traj)
minlen=len(trial_data['allpos_hat'][0])
for a in trial_data['allpos_hat']:
    if len(a)<minlen:
        minlen=len(a)
X=[]
for pos,vel,posstd,velstd in zip(trial_data['allpos_hat'],
trial_data['allvel_hat'],trial_data['allpos_std'],trial_data['allvel_std']):
    trialdata=pos[:minlen]+vel[:minlen]+posstd[:minlen]+velstd[:minlen]
    X.append(trialdata)

arrayX=np.array(X)
u,s,v=np.linalg.svd(arrayX)
s


for a in trial_data['allvel']:
    plt.plot(a)
for a in trial_data['allvel_hat']:
    plt.plot(a)
for a in trial_data['allpos_hat']:
    plt.plot(a)
for a in trial_data['allpos_std']:
    plt.plot(a)
for a in trial_data['allobs']:
    plt.plot(a)
for a in trial_data['allactions']:
    plt.plot(a)


def sample_trials(number_trials,theta=None,phi=None,goal_pos=None, timelimit=None,pro_traj=None, obs_traj=None):
    count=0
    allpos=[]
    allvel=[]
    allvel_hat=[]
    allpos_hat=[]
    allactions=[]
    allobs=[]
    allvel_std=[]
    allpos_std=[]
    alld=[]
    allsysvel=[]
    alldi=[]
    for i in range(number_trials):
        pos=[]
        pos_hat=[]
        vel=[]
        vel_hat=[]
        vel_std=[]
        actions=[]
        obs=[]
        pos_std=[]
        sysvel=[]
        di=[]
        env.reset(theta=theta, phi=phi, goal_position=goal_pos,obs_traj=obs_traj,pro_traj=pro_traj)
        di.append(env.decision_info.tolist())
        done=False
        while not done:
            with torch.no_grad():
                action=model(env.decision_info)[0]
                _,_,done,_=env.step(action)
                pos.append(env.s.data[0].item())
                vel.append(env.s.data[1].item())
                pos_hat.append(env.b.data[0].item())
                vel_hat.append(env.b.data[1].item())
                actions.append(action.item())
                obs.append(env.o.item())
                vel_std.append(env.P[-1,-1].item()**0.5)
                pos_std.append(env.P[0,0].item()**0.5)
                sysvel.append(env.sys_vel.item())
                di.append(env.decision_info.tolist())
                if timelimit is not None:
                    done=False
                    if env.episode_time>=timelimit:
                        break
        allpos.append(pos)
        allvel.append(vel)
        allvel_hat.append(vel_hat)
        allpos_hat.append(pos_hat)
        allactions.append(actions)
        allobs.append(obs)
        allvel_std.append(vel_std)
        allpos_std.append(pos_std)
        allsysvel.append(sysvel)
        d=env.get_distance()[1].item()
        alld.append(d)
        alldi.append(di)
        if env.reached_goal():
            count+=1
    # plt.hist(alld)
    return_dict={   'allpos':allpos,
                    'allpos_hat':allpos_hat,
                    'allpos_std':allpos_std,
                    'allvel':allvel,
                    'allvel_hat':allvel_hat,
                    'allvel_std':allvel_std,
                    'allsysvel':allsysvel,
                    'alld':alld,
                    'alldi':alldi,
                    'allobs':allobs,
                    'allactions':allactions,

    }
    return return_dict


def plot_agentvsexpert(theta,phi=None):
    trial_data=sample_trials(20, theta=theta, phi=phi, goal_pos=[env.goalx,env.goaly])
    for actions in trial_data['allactions']:
        plt.plot(actions)


# theta=torch.nn.Parameter(torch.tensor(theta_estimation))
true_theta=torch.nn.Parameter(torch.tensor(true_theta))
theta=true_theta.clone().detach()
theta[1]+=0.5
# theta[4]+=5
theta[2]+=100
theta[5]+=5
theta=torch.nn.Parameter(torch.tensor(theta))
phi=torch.tensor(phi)

plot_agentvsexpert(true_theta)
plot_agentvsexpert(theta)









# 1d acc model
env=ffac_1d.FireflyTrue1d(arg)
model=TD3.load('trained_agent/1d_easy1000000_9_25_5_14')
model.set_env(env)



# check sparse------------------------------------------
# with l2
baselines_mlp_model =TD3.load('trained_agent/1d_easy1000000_0_30_12_20.zip')
agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[512,512],n_inputs=15,n_actions=1)
w1=agent.fc1.weight
plt.hist(torch.mean(w1.clone().detach(),0))

baselines_mlp_model =TD3_torch.TD3.load('trained_agent/1d_easy1000000_0_30_12_20.zip')
for name, value in baselines_mlp_model.actor.named_parameters():
    if '0.weigh' in name:
        plt.hist(torch.mean(value.clone().detach().to('cpu'),1))
        plt.plot(torch.mean(value.clone().detach().to('cpu'),0))
        break

# no l2
baselines_mlp_model =TD3.load('trained_agent/1d_easy1000000_9_30_5_20.zip')
agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[512,512],n_inputs=15,n_actions=1)
w1=agent.fc1.weight
plt.hist(torch.mean(w1.clone().detach(),1))




# series of plots describing the trials
#-------------------------------------------------------
# max_distance=1
# uppertau=8.08
# param_range_dict={  'tau_range':[0.05, uppertau],
#                     }
env.terminal_vel= 0.02 
env.dt=0.2
env.episode_len=35
env.std_range = [0.02,0.1,0.02,0.1]
# env.setup(arg, max_distance=max_distance,param_range_dict=param_range_dict)

decision_info=env.reset(
    pro_noise_stds=torch.Tensor([0.2,0.3]),
    obs_noise_stds=torch.Tensor([0.2,0.3]),
        # pro_gains=torch.Tensor([0.5,1]),
        # obs_gains=torch.Tensor([0.51,1])
)
trial_index=1
decision_info=env.reset(
        goal_position=task_info[trial_index]['pos'],
        phi=task_info[trial_index]['phi'],
        theta=task_info[trial_index]['theta'])

env.reset()
phi=env.phi.detach().clone()
theta=env.theta.detach().clone()

phi[3,0]=9
theta[3,0]=9
phi[1,0]=9
theta[1,0]=9
decision_info=env.reset(
        # goal_position=task_info[trial_index]['pos'],
        phi=phi,
        theta=theta)
decision_info=env.reset()
done=False
while not done:
    action,_=model.predict(env.decision_info)
    decision_info,_,done,_=env.step(action)
    # fig=plot_belief(env,title=('action',action,"velocity", env.s[-2:],'tau:',env.phi[9]),kwargs={'title':action})
    # fig.savefig("{}.png".format(env.episode_time))
    # print(env.decision_info[0,:2],action)        
    # print(env.s[:,0],en v.phi[:-2,0])
    # print(env.episode_time)
    # print(env.s[3,0],env.b[3,0])
    print(action)


# Bangbang control
env.std_range=[1e-4,1e-3,1e-4,1e-3]
decision_info=env.reset()
action,s=model.predict(env.decision_info)
print(s)


while env.episode_time*env.dt<s:
    decision_info,_,done,_=env.step(action)
    fig=plot_belief(env,title=('CTRL',action,"velocity", env.s[-2:],'tau:',env.phi[9]))
action=[-a for a in action]
while env.s[3,0]>=0:
    decision_info,_,done,_=env.step(action)
    fig=plot_belief(env,title=('-CTRL',action,"velocity", env.s[-2:],'tau:',env.phi[9]))

# bangbang control discrete (more like rl agent)


plot_vw_distribution(env, model, number_trials=10,is1d=True)
correct_rate(env, model, number_trials=100, correct_threshold=0.5)

task_info=get_wrong_trial(env,model)
correct_rate(env, model, number_trials=100, task_info=task_info)


# change 2 param of 1d agent plt
paramnames=['distance to goal',
'current estimation of v',
'time',
'cov xx',
'cov xv',
'cov xv',
'cov vv',
'time',
'control gain',
'control noise std',
'obs gain',
'obs noise std',
'goal radius',
'tau',
'mag cost',
'dev cost'
]
param_pair=[0,10]
param_range=[[0.4,1],[0.001,99]]
number_pixels=10
env.reset()
theta=env.theta
phi=env.phi
decision_info=env.decision_info.clone().detach()
background_data=np.zeros((number_pixels,number_pixels))
X,Y=np.meshgrid(np.linspace(param_range[0][0],param_range[0][1],number_pixels),np.linspace(param_range[1][0],param_range[1][1],number_pixels))
for i in range(background_data.shape[0]):
    for j in range(background_data.shape[1]):
        d=decision_info.clone()[0]
        d[param_pair[0]]=X[i,j]
        d[param_pair[1]]=Y[i,j]
        background_data[i,j]=model.predict(d.view(1,-1))[0]

fig = plt.figure(figsize=[9, 9])
ax = fig.add_subplot(221)
im=ax.imshow(background_data)
add_colorbar(im)
ax.set_title('forward velocity V control of the agent', fontsize=20)
ax.set_xlabel('{}'.format(paramnames[param_pair[0]]), fontsize=15)
ax.set_ylabel('{}'.format(paramnames[param_pair[1]]), fontsize=15)

#- action distribution of 1d agent------------------------------------------------
env=ffac_1d.FireflyTrue1d(arg)
env.reset()
number_trials=10
true_theta=env.phi
theta=env.phi.clone().detach()
theta[2]=0.2
theta[1]=0.2
theta[4]=0.6
phi=env.phi
print(phi)

phi=torch.tensor(phi).cuda()
theta=torch.tensor(theta_estimation).cuda()
task=450*torch.ones(1).cuda()
theta=torch.nn.Parameter(theta)
true_theta=torch.nn.Parameter(torch.tensor(true_theta))

episode=0
astates_ep = []
aobs_ep=[]
aactions_ep = []
aposition_errorbar=[]
aposition_estimate=[]
av_estimate=[]
av_errorbar=[]
estates_ep = []
eobs_ep=[]
eactions_ep = []
eposition_errorbar=[]
eposition_estimate=[]
ev_estimate=[]
ev_errorbar=[]

t=0
env.reset(phi=phi,theta=theta,goal_position=task)
while t<len(eactions[episode]):
    action = agent(env.decision_info)[0]
    if estates is not None and t+1<len(estates[episode]):
        _,done=env(eactions[episode][t],task_param=theta,state=estates[episode][t]) # here
        astates_ep.append(env.s)
        aobs_ep.append(env.o.item())
        aposition_estimate.append(env.decision_info[0,0].item())
        aposition_errorbar.append(2*torch.sqrt(env.decision_info[0,3]).item()) # 2std
        av_estimate.append(env.decision_info[0,1].item())
        av_errorbar.append(2*torch.sqrt(env.decision_info[0,6]).item()) # 2std
    aactions_ep.append(action)
    t=t+1
env.reset(phi=phi,theta=true_theta,goal_position=etasks[episode][0])
t=0
while t<len(eactions[episode]):
    action = agent(env.decision_info)[0]
    if estates is not None and t+1<len(estates[episode]):
        _,done=env(eactions[episode][t],task_param=true_theta,state=estates[episode][t]) # here
        estates_ep.append(env.s)
        eobs_ep.append(env.o.item())
        eposition_estimate.append(env.decision_info[0,0].item())
        eposition_errorbar.append(2*torch.sqrt(env.decision_info[0,3]).item()) # 2std
        ev_estimate.append(env.decision_info[0,1].item())
        ev_errorbar.append(2*torch.sqrt(env.decision_info[0,6]).item()) # 2std
    eactions_ep.append(action)
    t=t+1


plt.plot([a[1,0].item() for a in astates_ep],color='b')
plt.plot(av_estimate,color='r')
low=[a-b for a, b in zip(av_estimate,av_errorbar)]
high=[a+b for a, b in zip(av_estimate,av_errorbar)]
plt.fill_between(list(range(len(low))),low, high,color='r',alpha=0.5)
plt.plot(aobs_ep,color='y')
plt.xlabel('time, t')
plt.ylabel('velocity')
plt.title('red, agent estimation. yellow, observation. blue, true')


plt.plot(eobs_ep,color='y')
plt.plot([a[1,0].item() for a in estates_ep],color='b')
plt.plot(ev_estimate,color='r')
low=[a-b for a, b in zip(ev_estimate,ev_errorbar)]
high=[a+b for a, b in zip(ev_estimate,ev_errorbar)]
plt.fill_between(list(range(len(low))),low, high,color='r',alpha=0.5)
plt.xlabel('time, t')
plt.ylabel('velocity')
plt.title('red, expert estimation. yellow, observation. blue, true')


plt.plot(eposition_errorbar,'b')
plt.plot(aposition_errorbar,'r')


plt.plot(ev_errorbar,'b')
plt.plot(av_errorbar,'r')



number_trials=10
estates, eactions, etasks, eobs = trajectory(
                agent, phi, true_theta, env, NUM_EP=number_trials,is1d=True,etask=[task], if_obs=True)
astates, aactions, atasks = trajectory(
            agent, phi, theta, env, NUM_EP=number_trials,
            is1d=True,etask=etasks, eaction=eactions,estate=estates)
# agent has a different obs std parameter and thus a different obs distribution
# plot expert actions vs agent actions. they are narrow, clearly seperate into 2 groups
for i in range(number_trials):
    plt.plot(eactions[i],'y',alpha=0.5)
    plt.plot(aactions[i],'r',alpha=0.5)
plt.xlabel('time, t')
plt.ylabel('control')
plt.title('red, agent. yellow, expert')



astates, aactions, atasks = trajectory(
            agent, phi, theta, env, NUM_EP=number_trials,
            is1d=True,etask=etasks, eaction=eactions,)

astates, aactions, atasks = trajectory(
            agent, phi, theta, env, NUM_EP=number_trials,
            is1d=True,etask=etasks, eaction=eactions,estate=estates,test_theta=true_theta)
for i in range(number_trials):
    plt.plot(aactions[i],'r')
    plt.plot(eactions[i],'y')

# agent has same obs distribution as expert, but recieve a different obs std param in actor
# belief is of same distribution, but theta is different.
# amlost same, except for too large paramter the agent never saw. this is expected, because uncertainty is small
astates, aactions, atasks = trajectory(
            agent, phi, true_theta, env, NUM_EP=number_trials,
            is1d=True,etask=etasks, eaction=eactions,estate=estates,test_theta=theta)
for i in range(number_trials):
    plt.plot(aactions[i],'r')
    plt.plot(eactions[i],'y')



# action distribution------------------------------
def return_vw_distribution(env, model,number_trials=10, is1d=False):
    env.reset()
    theta=env.theta
    phi=env.phi
    pos= env.goalx if is1d else [env.goalx,env.goaly] 
    theta=env.reset_task_param()
    v=[]
    w=[]
    d=[]
    mu=[]
    cov=[]
    vv=[]
    vw=[]
    if is1d:
        for trial_num in range(number_trials):
            env.reset(theta=theta.detach(), phi=phi.detach(), goal_position=pos)
            # env.reset()
            done=False
            vs=[]
            ds=[]
            mus=[]
            covs=[]
            vvs=[]
            while not done:
                action,_=model.predict(env.decision_info)
                decision_info,_,done,_=env.step(action)
                vs.append(action[0])
                vvs.append(env.s[1])
            # print(vs, ws)
            print('trial finished at : ',env.episode_time)
            v.append(vs)
            vv.append(vvs)
    else:
        for trial_num in range(number_trials):
            env.reset(theta=theta.detach(), phi=phi.detach(), goal_position=pos)
            # env.reset()
            done=False
            vs=[]
            ws=[]
            ds=[]
            mus=[]
            covs=[]
            vvs=[]
            vws=[]
            while not done:
                action,_=model.predict(env.decision_info)
                decision_info,_,done,_=env.step(action)
                vs.append(action[0])
                ws.append(action[1])
                vvs.append(env.s[3])
                vws.append(env.s[4])
                ds.append(decision_info[0].tolist()[3:3+16])
                mus.append(env.b[:2])
                covs.append(env.P[:2,:2])
            # print(vs, ws)
            print('trial finished at : ',env.episode_time)
            v.append(vs)
            w.append(ws)
            d.append(ds)
            mu.append(mus)
            cov.append(covs)
            vv.append(vvs)
            vw.append(vws)
    return v, w, vv, vw


def plot_vw_distribution(env, model, number_trials=10, is1d=False):
    v,w, vv, vw=return_vw_distribution(env, model,number_trials=number_trials,is1d=is1d)
    fig = plt.figure(figsize=[9, 9])
    ax = fig.add_subplot(221)
    ax1 = fig.add_subplot(222)
    ax.set_title('forward velocity V control of the agent', fontsize=20)
    ax.set_xlabel('time steps, dt={}'.format(env.dt), fontsize=15)
    ax.set_ylim([-1.1,1.1])



    ax2 = fig.add_subplot(223)
    ax2.set_xlabel('time steps, dt={}'.format(env.dt), fontsize=15)
    ax2.set_ylabel('v', fontsize=15)
    ax2.set_ylim([-1.1,1.1])

    for vs in v:
        ax.plot(vs)
    for vvs in vv:
        ax2.plot(vvs)

    if not is1d:
        ax1.set_title('angular velocity W control of the agent', fontsize=20)
        ax1.set_xlabel('time steps, dt={}'.format(env.dt), fontsize=15)
        ax.set_ylabel('control, range=[-1,1]', fontsize=15)
        ax1.set_ylabel('control, range=[-1,1]', fontsize=15)
        ax1.set_ylim([-1.1,1.1])
        ax3 = fig.add_subplot(224)
        ax3.set_xlabel('time steps, dt={}'.format(env.dt), fontsize=15)
        ax3.set_ylabel('w', fontsize=15)
        ax3.set_ylim([-1.1,1.1])
        for ws in w:
            ax1.plot(ws) 
        for vws in vw:
            ax3.plot(vws)


    return fig


def correct_rate(env, model, number_trials=100, task_info=None,correct_threshold=5):
    correct=0
    if task_info is list:
        for i in range(len(task_info)):
                env.reset(goal_position=task_info['pos'],
                            phi=task_info['phi'],
                            theta=task_info['theta'])
                done=False
                while not done:
                    action,_=model.predict(env.decision_info)
                    decision_info,_,done,_=env.step(action)
                    # fig.savefig("{}.png".format(env.episode_time))
                if env.episode_reward>5:
                    correct=correct+1

    else:
        for i in range(number_trials):
            if task_info is dict:
                env.reset(goal_position=task_info['pos'],
                            phi=task_info['phi'],
                            theta=task_info['theta'])
            else:
                env.reset()
                done=False
                while not done:
                    action,_=model.predict(env.decision_info)
                    decision_info,_,done,_=env.step(action)
                    # fig.savefig("{}.png".format(env.episode_time))
                if env.episode_reward>correct_threshold:
                    correct=correct+1
    return correct/number_trials


def get_wrong_trial(env,model, number_trials=100):
    tasks=[]
    while len(tasks)<number_trials:
        task_info={}
        while task_info=={}:
            done=False
            while not done:
                action,_=model.predict(env.decision_info)
                decision_info,_,done,_=env.step(action)
                if env.episode_reward<5:
                    task_info['pos']=[env.goalx,env.goaly]
                    task_info['phi']=env.phi
                    task_info['theta']=env.theta
        tasks.append(task_info)
    return tasks
    # v w and controls for some trials    

#------------------------------------------------------
def plot_belief(env,title='title',**kwargs):
    f1=plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.set_ylim(-1.5,1.5)
    ax.set_xlim(-1.5,1.5)
    pos=env.b[:2,0].detach()
    cov=env.P[:2,:2].detach()
    # plt.plot(pos[0],pos[1],'o')
    # print('test',title)
    if kwargs.get('title'):
        print('test',kwargs.get('title'))
        title=kwargs.get('title')
    plt.title(title)
    plt.plot(env.s.detach()[0,0],env.s.detach()[1,0],'ro')
    plt.plot([pos.detach()[0],pos.detach()[0]+env.decision_info.detach()[0,0]*np.cos(env.b.detach()[2,0]+env.decision_info.detach()[0,1]) ],
    [pos.detach()[1],pos.detach()[1]+env.decision_info.detach()[0,0]*np.sin(env.b.detach()[2,0]+env.decision_info.detach()[0,1])],'g')
    plt.quiver(pos.detach()[0], pos.detach()[1],np.cos(env.b.detach()[2,0].item()),np.sin(env.b.detach()[2,0].item()), color='r', scale=10)
    plot_cov_ellipse(cov, pos, nstd=2,ax=ax)
    # plot_cov_ellipse(np.diag([1,1])*0.05, [env.goalx,env.goaly], nstd=1, ax=ax)
    plot_circle(np.eye(2)*env.phi[8,0].item(),[env.goalx,env.goaly],ax=ax,color='y')
    return f1


def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, color=None, ax=None,alpha=0.5, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        figure=plt.figure(figsize=(10,10))
        ax = plt.gca()
        ax.set_ylim(-1.5,1.5)
        ax.set_xlim(-1.5,1.5)


    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    if color is not None:
        ellip.set_color(color)
    ellip.set_alpha(alpha)
    ax.add_artist(ellip)
    return ellip
    

def plot_circle(cov, pos, color=None, ax=None, **kwargs):

    if ax is None:
        figure=plt.figure(figsize=(10,10))
        ax = plt.gca()
        ax.set_ylim(-1.5,1.5)
        ax.set_xlim(-1.5,1.5)
    assert cov[0,0]==cov[1,1]
    r=cov[0,0]
    c = Circle(pos,r)
    if color is not None:
        c.set_color(color)
    c.set_alpha(0.5)
    ax.add_artist(c)
    return c


def overlap_prob(r, cov, mu ):
    # normalizing_z=2*pi*r**2
    rop=cov.copy()
    rop[0,0]=rop[0,0]+r**2
    rop[1,1]=rop[0,0]+r**2

    # normalizing_z=1/2/pi/np.sqrt(np.linalg.det(rop))


    if type(mu) == list:
        vector_mu=np.asarray(mu).reshape((2,1))
    # expectation=np.exp((-1/2/pi/r**2) * vector_mu.transpose()@np.linalg.inv(rop)@vector_mu)
    expectation=np.exp(-1/2* vector_mu.transpose()@np.linalg.inv(rop)@vector_mu)

    return expectation


def overlap_intergration(r, cov, mu, bins=20):
    xrange=[mu[0]-r*1.1,mu[0]+r*1.1]
    yrange=[mu[1]-r*1.1,mu[1]+r*1.1]
    P=0
    a=np.zeros((bins,bins))
    b=np.zeros((bins,bins))
    xs=np.linspace(xrange[0],xrange[1],bins)
    ys=np.linspace(yrange[0],yrange[1],bins)

    for i in range(bins):
        for j in range(bins):
            if (xs[i]-mu[0])**2+(ys[j]-mu[1])**2<=r**2:                
                expectation=( (1/2/pi/np.sqrt(np.linalg.det(cov)))
                * np.exp(-1/2
                    * (np.array([xs[i],ys[j]]).reshape(1,2)@np.linalg.inv(cov)@np.array([xs[i],ys[j]]).reshape(2,1)) ))
                P=P+expectation*4*r*r/bins/bins
                a[i,j]=(expectation*4*r*r/bins/bins)[0]
                b[i,j]=1

    return P


def overlap_mc(r, cov, mu, nsamples=1000):
    # xrange=[-cov[0,0],cov[0,0]]
    # yrange=[-cov[1,1],cov[1,1]]
    # xrange=[mu[0]-r*1.1,mu[0]+r*1.1]
    # yrange=[mu[1]-r*1.1,mu[1]+r*1.1]

    check=[]
    xs, ys = np.random.multivariate_normal(-mu, cov, nsamples).T
    # plot_overlap(r,cov,mu,title=None)
    for i in range(nsamples):
        # plt.plot(xs[i],ys[i],'.')
        if (xs[i])**2+(ys[i])**2<=r**2:                
            check.append(1)
        else:
            check.append(0)
    P=np.mean(check)
    return P


def plot_overlap(r,cov,mu,title=None):
    f1=plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.set_ylim(-1.5,1.5)
    ax.set_xlim(-1.5,1.5)
    if title is not None:
        ax.title.set_text(str(title))
    plot_cov_ellipse(cov,[0,0],nstd=1,ax=ax)
    plot_circle(np.eye(2)*r,mu,ax=ax,color='r')
    return f1

# mu=[0,0]
# plot_cov_ellipse(cov,[0.,0.])
# plot_cov_ellipse(np.eye(2)*0.1,mu,nstd=1)

# cov=np.array([[0.03,0.03],[0.03,0.05]])
# cov=cov*0.6
# cov=cov*10
# r=0.2
# for i in np.linspace(-0.5,0.5,19):
#     for j in np.linspace(-0.5,0.5,19):
#         plot_overlap(r,cov,[i,j],title=str(str(overlap_intergration(r,cov,[i,j]))+str(overlap_prob(r,cov,[i,j]))))
#         plt.savefig("{}_{}.png".format(str(i),str(j)))

# plot belief with stable baselines agent
# env.reset(
#     pro_noise_stds=torch.Tensor([-0.05*8,-pi/80*8]),
#     obs_noise_stds=torch.Tensor([0.05*8,pi/80*8]))
# env.reset(phi=phi, theta=phi)
# env.reset(theta=theta.detach(), phi=phi.detach(), goal_position=pos)
# while True:
#     env.reset(
#         # pro_noise_stds=torch.Tensor([0.2,0.2])
#             # pro_gains=torch.Tensor([0.5,1]),
#             # obs_gains=torch.Tensor([0.51,1])
#     )
#     done=False
#     # phi=env.phi
#     while not done:
#         action,_=model.predict(env.decision_info)
#         decision_info,_,done,_=env.step(action)
#         # fig=plot_belief(env,title=('action',action,env.phi),kwargs={'title':action})
#         # fig.savefig("{}.png".format(env.episode_time))
#         # print(env.decision_info[0,:2],action)        
#         # print(env.s[:,0],env.phi[:-2,0])
#         # print(env.episode_time)
#     # if env.caculate_reward()==0.:
#     #     break
#     print('final reward ',env.caculate_reward(), env.episode_time)



# plot belief with torch agent, to make sure translate correctly
# import policy_torch
# agent = policy_torch.copy_mlp_weights(model,layers=[128,128])
# env.reset()
# while not env.stop:
#     action = agent(env.decision_info)[0]
#     decision_info,done=env(action, env.theta)
#     fig=plot_belief(env,title=(action,env.phi),kwargs={'title':action})
#     # fig.savefig("{}.png".format(env.episode_time))




# belief distribution of single trial
fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot()
ax.set_xlim([0,1])
ax.set_ylim([-0.5,0.5])
for covs, mus in zip(cov, mu):
    for onecov,onemu in zip(covs, mus):
        plot_cov_ellipse(onecov, onemu, ax=ax,nstd=2,alpha=0.1)

fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot()
ax.set_xlim([0,1])
ax.set_ylim([-0.5,0.5])
for covs, mus in zip(cov, mu):
    for onecov,onemu in zip(covs, mus):
        ax.scatter(onemu[0],onemu[1])



# # testing decision info distribution 
number_trials=10
d_entry=3
t_entry=9
entry_ls=[]
for trial in range(number_trials):
    entry_ls.append(d[trial][t_entry][d_entry])
plt.plot(entry_ls)

# testing decision info growth 
d_entry=5
trial_entry=7
entry_ls=[]
for t in range(len(d[trial_entry])):
    entry_ls.append(d[trial_entry][t][d_entry])
plt.plot(entry_ls)


# mu=torch.Tensor([ 0.2194, 0.2194])
# cov=torch.Tensor([[5.0709e-05, 4.5831e-06],
#         [4.5831e-06, 5.9715e-07]])
# bins=20
# r=0.1873
# mu=torch.Tensor([ 0., -0.])

# xrange=[mu[0]-goal_radius, mu[0]+goal_radius]
# yrange=[mu[1]-goal_radius, mu[1]+goal_radius]
# P=0
v=[]
w=[]
for i in range(1000):
    decision_info= env.decision_info
    decision_info[:,:9]=env.reset_task_param().view(1,-1)
    action,_=model.predict(decision_info)
    v.append(action[0])
    w.append(action[1])
plt.hist(v,bins=100)
plt.plot([true_action[0],true_action[0]],[0,99],color='r')
plt.hist(w,bins=100)
plt.plot([true_action[1],true_action[1]],[0,99],color='r')


true_action,_=model.predict(env.decision_info)
decision_info,_,done,_=env.step(true_action)



# plot policy surface
def plot_policy_surfaces(decision_info,model):
    r_range=[0.0,1.0]
    r_ticks=0.01
    r_labels=[r_range[0]]
    a_range=[-pi/4, pi/4]
    a_ticks=0.05
    a_labels=[a_range[0]]
    while r_labels[-1]+r_ticks<=r_range[-1]:
        r_labels.append(r_labels[-1]+r_ticks)
    while a_labels[-1]+a_ticks<=a_range[-1]:
        a_labels.append(a_labels[-1]+a_ticks)
    policy1_data_v=np.zeros((len(r_labels),len(a_labels)))
    policy1_data_w=np.zeros((len(r_labels),len(a_labels)))
    for ri in range(len(r_labels)):
        for ai in range(len(a_labels)):
            action,_=model.predict(decision_info)
            decision_info[:,0]=r_labels[ri]
            decision_info[:,1]=a_labels[ai]
            action,_=model.predict(decision_info)
            policy1_data_v[ri,ai]=action[0]
            policy1_data_w[ri,ai]=action[1]

    fig, ax = plt.subplots(1, 2,
            gridspec_kw={'hspace': 0.4, 'wspace': 0.2},figsize=(16,16))
    # fig.suptitle('{} and {} policy surface'.format('rua',fontsize=40))
    ax[0].set_title('forward velocity  '+'tau : '+str( env.phi[9]),fontsize=24)
    ax[0].set_xlabel('relative angle',fontsize=18)
    ax[0].set_ylabel('relative distance',fontsize=18)
    im1=ax[0].imshow(policy1_data_v,origin='lower',vmin=-1.,vmax=1.,extent=[a_labels[0],a_labels[-1],r_labels[0],r_labels[-1]])
    # add_colorbar(im1)
    ax[1].set_title('ang velocity',fontsize=24)
    ax[1].set_xlabel('relative angle',fontsize=18)
    ax[1].set_xlabel('relative distance',fontsize=18)
    im2=ax[1].imshow(policy1_data_w,origin='lower',vmin=-1.,vmax=1.,extent=[a_labels[0],a_labels[-1],r_labels[0],r_labels[-1]])
    add_colorbar(im2)
    return fig

def inverseCholesky(vecL):
    """
    Performs the inverse operation to lower cholesky decomposition
    and converts vectorized lower cholesky to matrix P
    P = L L.t()
    """
    size = int(np.sqrt(2 * len(vecL)))
    L = np.zeros((size, size))
    mask = np.tril(np.ones((size, size)))
    L[mask == 1] = vecL
    P = L@(L.transpose())
    return P

# test policy surface when change cov
from FireflyEnv.env_utils import *

env.reset()

decision_info= env.decision_info
true_action,_=model.predict(env.decision_info.detach())
_,_,_,_=env.forward(true_action, env.phi)
decision_info[:,5:5+15]=vectorLowerCholesky(env.P)
plot_policy_surfaces(decision_info.detach(),model)

true_action,_=model.predict(env.decision_info)
_,_,_,_=env.step(true_action)
plot_policy_surfaces(env.decision_info,model)


env.episode_time=env.episode_time+10
print(env.episode_time)
env.decision_info[:,2:4]=env.decision_info[:,2:4]*0+1

decision_info=env.wrap_decision_info()
plot_policy_surfaces(decision_info,model)


# ax.set_xlim([0,1])
# ax.set_ylim([-0.5,0.5])


    # for a in ax.flat:
    #     a.set(xlabel='relative angle', ylabel='relative distance')
    # # for a in ax.flat:
    # #     a.label_outer()

    # ax[2,0].text(-1.3,+5.5,'time {}'.format(str(belief.tolist()[4])),fontsize=24)
    # ax[2,0].text(-3.3,-2.5,'theta {}'.format(str(['{:.2f}'.format(x) for x in (belief.tolist()[20:])])),fontsize=20)
    # ax[2,0].text(-2.3,1.5,'scale bar,  -1                0                +1',fontsize=24)

    # ax[2,0].imshow((np.asarray(list(range(10)))/10).reshape(1,-1))
    # ax[2,0].axis('off')

    # ax[2,1].set_title('P matrix',fontsize=24)
    # ax[2,1].imshow(inverseCholesky(belief.tolist()[5:20]))
    # ax[2,1].axis('off')

    # plt.savefig('./policy plots/{} and {} policy surface {}.png'.format(torch_model1.name,torch_model2.name,name_index))





# new 2d testing, plot v w given theta and theta estimation
env=ffacc_real.Firefly_real_vel(arg)
env.reset()
phi=torch.tensor(phi).float()
theta=torch.tensor(theta_estimation).float()
true_theta=torch.tensor(true_theta).float()
task=[[env.goalx,env.goaly],0.13]
pos=[env.goalx,env.goaly]
estates, eactions, etasks = trajectory(
                agent, phi, true_theta, env, NUM_EP=20,is1d=False,etask=[task])
with torch.no_grad():
    astates, aactions, atasks = trajectory(
                agent, phi, theta, env, NUM_EP=20,
                is1d=False,etask=etasks, eaction=eactions,estate=estates)
eactions=[torch.stack(x) for x in eactions]
aactions=[torch.stack(x) for x in aactions]
for e in eactions:
    plt.plot(e)
for a in aactions:
    plt.plot(a) 
    
for ind in list(range(20)):
    plt.plot(torch.stack(astates[ind])[:,:,0][:,0],torch.stack(astates[ind])[:,:,0][:,1])
for ind in list(range(20)):
    plt.plot(torch.stack(estates[ind])[:,:,0][:,0],torch.stack(estates[ind])[:,:,0][:,1])
plt.plot(env.goalx,env.goaly,'-o')

# new 2d testing, plot v w given theta and theta estimation, monkey
ind=torch.randint(low=100,high=8000,size=(1,))
env=ffacc_real.Firefly_real_vel(arg)
env.reset()
phi=torch.tensor(phi).float()
theta=torch.tensor(theta_estimation).float()
true_theta=torch.tensor(true_theta).float()
_, _, etasks = trajectory(
        agent, phi, phi, env, NUM_EP=20,is1d=False,etask=[tasks[ind]])
with torch.no_grad():
    astates, aactions, atasks,abeliefs = trajectory(
        agent, phi, theta, env, NUM_EP=20,
        is1d=False, etask=etasks, return_belief=True,
        eaction=[torch.tensor(actions[ind]) for i in list(range(20))],
    # estate=None)
        estate=[torch.tensor(states[ind])for i in list(range(20))])
aactions=[torch.stack(x) for x in aactions]
for oneaction in aactions:
    plt.plot(oneaction) 
plt.plot(actions[ind])

plt.plot(torch.tensor(states[ind])[0,:],torch.tensor(states[ind])[1,:])
# plt.plot(torch.tensor(states[ind])[0,:]-states[ind][0,0],torch.tensor(states[ind])[1,:]-states[ind][1,0])
for i in list(range(20)):
    plt.plot(torch.stack(astates[i])[:,:,0][:,0],torch.stack(astates[i])[:,:,0][:,1])
for i in list(range(20)):
    plt.plot(torch.stack(abeliefs[i])[:,:,0][:,0],torch.stack(abeliefs[i])[:,:,0][:,1])
plt.plot(torch.stack(abeliefs[i])[:,:,0][:,2])
plt.plot((states[ind])[2,:])



# test theta cov std. loss change with respct to param change.
deltalist=list(range(10))
deltalist=[(l-5)*0.001 for l in theta_list]
phi=torch.nn.Parameter(phi)
theta=torch.nn.Parameter(theta)
saves=[]
for delta in deltalist:
    with torch.no_grad():
        # loss = getLoss(agent, eactions, etasks, phi, theta, env, num_iteration=1, states=estates, samples=30,gpu=False)
        # theta_=theta.clone().detach()
        theta_[1]+=delta
        theta_=torch.nn.Parameter(theta_)
        loss2 = getLoss(agent, eactions, etasks, phi, theta_, env, num_iteration=1, states=estates, samples=10,gpu=False)
        # print(loss,loss2)
        saves.append(loss2)
plt.plot(saves)


# esitmate policy temperature, plot aw vs theta
env=ffacc_real.Firefly_real_vel(arg)
info=env.reset()
angles=np.linspace(-pi/2,pi/2,10)
aws=[]
for angle in angles:
    info[0,1]=angle
    aws.append(agent(info)[0,1])
plt.plot(angles,aws)
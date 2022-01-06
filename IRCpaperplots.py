
from inspect import EndOfBlock
from pickle import HIGHEST_PROTOCOL
import multiprocessing
from matplotlib.collections import LineCollection
from numpy.core.numeric import NaN
from scipy.ndimage.measurements import label
from torch.nn import parameter
from plot_ult import add_colorbar
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import torch
import warnings
import random
from astropy.convolution import convolve
from numpy import pi
# from numba import njit
from contextlib import redirect_stdout, redirect_stderr, contextmanager, ExitStack
import os
import matplotlib as mpl
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict
import matplotlib.pylab as pl
from plot_ult import *
from InverseFuncs import *
cmaps = OrderedDict()
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
global color_settings
color_settings={
    'v':'tab:blue',
    'w':'orange',
    'cov':'blue',
    'b':'b',
    's':'r',
    '':'',
}
global theta_names
theta_names = [ 'pro gain v',
                'pro gain w',
                'pro noise v',
                'pro noise w',
                'obs noise v',
                'obs noise w',
                'goal radius',
                'action cost v',      
                'action cost w',      
                'init uncertainty x',      
                'init uncertainty y',      
                ]

global theta_mean
theta_mean=[
    0.4,
    1.57,
    0.5,
    0.5,
    0.5,
    0.5,
    0.13,
    0.5,
    0.5,
    0.5,
    0.5    
]

global parameter_range
parameter_range=[
    [0.4,0.6],
    [1.2,1.8],
    [1e-3,1],
    [1e-3,1],
    [1e-3,1],
    [1e-3,1],
    [1e-3,0.5],
    [1e-3,1],
    [1e-3,1],
    [1e-3,1],
    [1e-3,1],
]

# -------------------------------------------------
# load data

import warnings
warnings.filterwarnings('ignore')
from copy import copy
import time
import random
from stable_baselines3 import SAC,PPO
seed=0
random.seed(seed)
import torch
torch.manual_seed(seed)
import numpy as np
np.random.seed(int(seed))
from numpy import pi
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from InverseFuncs import trajectory, getLoss, reset_theta, theta_range,reset_theta_log, single_theta_inverse
from FireflyEnv import ffacc_real
import TD3_torch
from monkey_functions import *
from Config import Config
arg = Config()

arg.presist_phi=True
arg.agent_knows_phi=False
arg.goal_distance_range=[0.1,1]
arg.gains_range =[0.05,1.5,pi/4,pi/1]
arg.goal_radius_range=[0.001,0.3]
arg.std_range = [0.08,1,0.08,1]
arg.mag_action_cost_range= [0.0001,0.001]
arg.dev_action_cost_range= [0.0001,0.005]
arg.dev_v_cost_range= [0.1,0.5]
arg.dev_w_cost_range= [0.1,0.5]
arg.TERMINAL_VEL = 0.1
arg.DELTA_T=0.1
arg.EPISODE_LEN=100
arg.agent_knows_phi=False
DISCOUNT_FACTOR = 0.99
arg.sample=100
arg.batch = 70
# arg.NUM_SAMPLES=1
# arg.NUM_EP=1
arg.NUM_IT = 1 
arg.NUM_thetas = 1
arg.ADAM_LR = 0.0002
arg.LR_STEP = 20
arg.LR_STOP = 0.5
arg.lr_gamma = 0.95
arg.PI_STD=1
arg.presist_phi=False
arg.cost_scale=1

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


# illustrate perterb trial
def pertexample():
    with initiate_plot(3,3,300) as fig:
        ax=fig.add_subplot()
        ax.plot(df.iloc[0].pos_v,color=color_settings['v'],label='real v')
        ax.plot([i*200 for i in df.iloc[0].action_v ],'--',color=color_settings['v'],label='v control')
        ax.plot(df.iloc[0].pos_w ,color=color_settings['w'],label='real w')
        ax.plot([i*200 for i in df.iloc[0].action_w ],'--',color=color_settings['w'],label='w control')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time [dt]')
        ax.set_ylabel('velocity [cm/s and degree/s]')

# illustrate density
def densityexample():
    densities=[0.0001, 0.0005, 0.001, 0.005]
    marker = itertools.cycle(('v', '^', '<', '>')) 
    with initiate_plot(8,2,300) as fig:
        for i,density in enumerate(densities):
            ax=fig.add_subplot(1,4,i+1)    
            ax.set_xticks([]);ax.set_yticks([])
            x=torch.zeros(int(10000*3*density)).uniform_()
            y=torch.zeros(int(10000*3*density)).uniform_()
            for xx,yy in zip(x,y):
                plt.plot(xx,yy, marker = next(marker), linestyle='',color='k')
            ax.set_xlabel('density={}'.format(density))

            


# assume the wrong gain and agent makes sense 
def agent_double_gain():
    phi=torch.tensor([[0.4000],
            [1.57],
            [0.01],
            [0.01],
            [0.01],
            [0.01],
            [0.13],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
    ])
    theta1=torch.tensor([[0.4000],
            [1.57],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
            [0.13],
            [0.9],
            [0.9],
            [0.1],
            [0.1],
    ])
    # double gain theta
    theta2=torch.tensor([[0.8000],
            [1.57],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
            [0.13],
            [0.9],
            [0.9],
            [0.1],
            [0.1],
    ])
    input={
        'agent':agent,
        'theta':theta1,
        'phi':phi,
        'env': env,
        'num_trials':100,
        # 'task':[0.7,-0.3],
        'mkdata':{},
        'use_mk_data':False,
    }

    with suppress():
        res1=trial_data(input)
    errs1=[]
    goalds1=[]
    for trialind in range(res1['num_trials']):
        err=torch.norm(torch.tensor(res1['task'][trialind])-res1['agent_states'][trialind][-1][:2])
        errs1.append(err)
        goalds1.append(torch.norm(torch.tensor(res1['task'][trialind])))
    input={
        'agent':agent,
        'theta':theta2,
        'phi':phi,
        'env': env,
        'num_trials':100,
        'task':res1['task'],
        'mkdata':{},
        'use_mk_data':False,
    }
    with suppress():
        res2=trial_data(input)
    errs2=[]
    goalds2=[]
    for trialind in range(res2['num_trials']):
        err=torch.norm(torch.tensor(res2['task'][trialind])-res2['agent_states'][trialind][-1][:2])
        errs2.append(err)
        goalds2.append(torch.norm(torch.tensor(res2['task'][trialind])))

    errs1=[e.item() for e in errs1]
    goalds1=[e.item() for e in goalds1]
    errs2=[e.item() for e in errs2]
    goalds2=[e.item() for e in goalds2]

    # error scatter plot
    with initiate_plot(3,3,300):
        plt.scatter(goalds1,errs1, alpha=0.3,edgecolors='none')
        plt.scatter(goalds2,errs2, alpha=0.3,edgecolors='none')
        plt.title('error vs goal distance', fontsize=20)
        plt.ylabel('error')
        plt.xlabel('goal distance')

    # overhead view plot
    ax=plotoverheadcolor(res1)
    plotoverheadcolor(res2,linewidth=1,alpha=0.3,ax=ax)
    ax.get_figure()

# large goal radius
def agent_double_goalr():
    theta2=torch.tensor([[0.4],
            [1.57],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
            [0.4],
            [0.9],
            [0.9],
            [0.1],
            [0.1],
    ])
    input={
        'agent':agent,
        'theta':theta1,
        'phi':phi,
        'env': env,
        'num_trials':100,
        # 'task':[0.7,-0.3],
        'mkdata':{},
        'use_mk_data':False,
    }

    with suppress():
        res1=trial_data(input)
    errs1=[]
    goalds1=[]
    for trialind in range(res1['num_trials']):
        err=torch.norm(torch.tensor(res1['task'][trialind])-res1['agent_states'][trialind][-1][:2])
        errs1.append(err)
        goalds1.append(torch.norm(torch.tensor(res1['task'][trialind])))
    input={
        'agent':agent,
        'theta':theta2,
        'phi':phi,
        'env': env,
        'num_trials':100,
        'task':res1['task'],
        'mkdata':{},
        'use_mk_data':False,
    }
    with suppress():
        res2=trial_data(input)
    errs2=[]
    goalds2=[]
    for trialind in range(res2['num_trials']):
        err=torch.norm(torch.tensor(res2['task'][trialind])-res2['agent_states'][trialind][-1][:2])
        errs2.append(err)
        goalds2.append(torch.norm(torch.tensor(res2['task'][trialind])))

    errs1=[e.item() for e in errs1]
    goalds1=[e.item() for e in goalds1]
    errs2=[e.item() for e in errs2]
    goalds2=[e.item() for e in goalds2]

    # error scatter plot
    with initiate_plot(3,3,300):
        plt.scatter(goalds1,errs1, alpha=0.3,edgecolors='none')
        plt.scatter(goalds2,errs2, alpha=0.3,edgecolors='none')
        plt.title('error vs goal distance', fontsize=20)
        plt.ylabel('error')
        plt.xlabel('goal distance')

    # plt.hist(errs1)
    # plt.hist(errs2)

    # overhead view plot
    ax=plotoverheadcolor(res1)
    plotoverheadcolor(res2,linewidth=1,alpha=0.3,ax=ax)
    ax.get_figure()

# verify phi task vs true task
def verify():
    index=ind
    s=states[index]
    a=actions[index]
    env.reset(phi=phi,theta=theta,
        goal_position=tasks[index],vctrl=a[0,0],wctrl=a[0,1])
    vstates=[]
    for aa in a:
        env.step(aa)
        vstates.append(env.s)
    svstates=torch.stack(vstates)[:,:,0]
    svstates=svstates-svstates[0]
    plt.plot(svstates[:,0],svstates[:,1])
    plt.plot(s[:,0],s[:,1])
    plt.show()
    svstates[-1,:2]
    s[-1,:2]


def overhead_skip_density(input):
    fontsize = 9
    target_idexes = np.arange(1500, 2000)
    with initiate_plot(1.8, 1.8, 300) as fig:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
        
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
        
        skipx=[]
        skipy=[]
        for trial_i in range(input['num_trials']):
            # calculate if rewarded
            dx=input['agent_states'][trial_i][-1,1]*400-input['task'][trial_i][1]*400
            dy=input['agent_states'][trial_i][-1,0]*400-input['task'][trial_i][0]*400
            d2goal=(dx**2+dy**2)**0.5
            d2start=(input['agent_states'][trial_i][-1,1]**2+input['agent_states'][trial_i][-1,0]**2)**0.5*400

            if d2goal>200:
                skipx.append(input['task'][trial_i][1]*400)
                skipy.append(input['task'][trial_i][0]*400)
        skipx=torch.stack(skipx).tolist()  
        skipy=torch.stack(skipy).tolist()  
        densObj = kde(np.array([skipx,skipy]))
        colours = makeColours( densObj.evaluate( np.array([skipx,skipy]) ) )
        ax.scatter( skipx, skipy, color=colours)
        # img, extent = myplot(skipx, skipy, 77)
        # ax.contourf(img, extent=extent, origin='lower', cmap=cm.Greys)
        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)

        fig.tight_layout(pad=0)

def diagnose_plot_theta(agent, env, phi, theta_init, theta_final,nplots,):
  def sample_trials(agent, env, theta, phi, etask, num_trials=5):
    # print(theta)
    agent_actions=[]
    agent_beliefs=[]
    agent_covs=[]
    agent_states=[]
    agent_dev_costs=[]
    agent_mag_costs=[]
    with torch.no_grad():
      for trial_i in range(num_trials):
        env.reset(phi=phi,theta=theta,goal_position=etask[0])
        epbliefs=[]
        epbcov=[]
        epactions=[]
        epstates=[]
        t=0
        done=False
        while not done:
          action = agent(env.decision_info)[0]
          _,_,done,_=env.step(torch.tensor(action).reshape(1,-1)) 
          epactions.append(action)
          epbliefs.append(env.b)
          epbcov.append(env.P)
          epstates.append(env.s)
          t=t+1
        agent_dev_costs.append((env.trial_costs))
        # agent_mag_costs.append(torch.stack(env.trial_mag_costs))
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
        'etask':etask,
        'theta':theta,
        'devcosts':agent_dev_costs,
        'magcosts':agent_mag_costs,
      }
    return return_dict
  delta=(theta_final-theta_init)/(nplots-1)
  fig = plt.figure(figsize=[20, 20])
  # curved trails
  etask=[[0.7,-0.3]]
  for n in range(nplots):
    ax1 = fig.add_subplot(6,nplots,n+1)
    theta=(n-1)*delta+theta_init
    ax1.set_xlabel('world x, cm')
    ax1.set_ylabel('world y, cm')
    ax1.set_title('state plot')
    data=sample_trials(agent, env, theta, phi, etask, num_trials=1)
    estate=data['estate']
    ax1.plot(estate[0,:],estate[1,:], color='r',alpha=0.5)
    goalcircle = plt.Circle((etask[0][0],etask[0][1]), theta[6], color='y', alpha=0.5)
    ax1.add_patch(goalcircle)
    ax1.set_xlim([-0.1,1.1])
    ax1.set_ylim([-0.6,0.6])
    agent_beliefs=data['agent_beliefs']
    for t in range(len(agent_beliefs[0][:,:,0])):
      cov=data['agent_covs'][0][t][:2,:2]
      pos=  [agent_beliefs[0][:,:,0][t,0],
              agent_beliefs[0][:,:,0][t,1]]
      plot_cov_ellipse(cov, pos, nstd=2, color=None, ax=ax1,alpha=0.2)

  # v and w
    ax2 = fig.add_subplot(6,nplots,n+nplots+1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('w')
    ax2.set_title('w control')
    agent_actions=data['agent_actions']
    for i in range(len(agent_actions)):
      ax2.plot(agent_actions[i],alpha=0.7)
    ax2.set_ylim([-1.1,1.1])
#   # dev and mag costs
#     ax2 = fig.add_subplot(6,nplots,n+nplots*2+1)
#     ax2.set_xlabel('t')
#     ax2.set_ylabel('costs')
#     ax2.set_title('costs')
#     for i in range(len(data['devcosts'])):
#       ax2.plot(data['devcosts'][0], color='green',alpha=0.7)
#       # ax2.plot(data['magcosts'][0], color='violet',alpha=0.7)

  # straight trails
  etask=[[0.7,0.0]]
  for n in range(nplots):
    ax1 = fig.add_subplot(6,nplots,n+nplots*3+1)
    theta=(n-1)*delta+theta_init
    ax1.set_xlabel('world x, cm')
    ax1.set_ylabel('world y, cm')
    ax1.set_title('state plot')
    data=sample_trials(agent, env, theta, phi, etask, num_trials=1)
    estate=data['estate']
    ax1.plot(estate[0,:],estate[1,:], color='r',alpha=0.5)
    goalcircle = plt.Circle((etask[0][0],etask[0][1]), theta[6], color='y', alpha=0.5)
    ax1.add_patch(goalcircle)
    ax1.set_xlim([-0.1,1.1])
    ax1.set_ylim([-0.6,0.6])
    agent_beliefs=data['agent_beliefs']
    for t in range(len(agent_beliefs[0][:,:,0])):
      cov=data['agent_covs'][0][t][:2,:2]
      pos=  [agent_beliefs[0][:,:,0][t,0],
              agent_beliefs[0][:,:,0][t,1]]
      plot_cov_ellipse(cov, pos, nstd=2, color=None, ax=ax1,alpha=0.2)

  # v and w
    ax2 = fig.add_subplot(6,nplots,n+nplots*4+1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('w')
    ax2.set_title('w control')
    data.keys()
    agent_actions=data['agent_actions']
    for i in range(len(agent_actions)):
      ax2.plot(agent_actions[i],alpha=0.7)
    ax2.set_ylim([-1.1,1.1])

    # # dev and mag costs
    # ax2 = fig.add_subplot(6,nplots,n+nplots*5+1)
    # ax2.set_xlabel('t')
    # ax2.set_ylabel('costs')
    # ax2.set_title('costs')
    # for i in range(len(data['devcosts'])):
    #   ax2.plot(data['devcosts'][0], color='green',alpha=0.7)
    #   # ax2.plot(data['magcosts'][0], color='violet',alpha=0.7)

def policygiventheta():

    env.reset(phi=phi,theta=theta)
    bl=torch.zeros(5)
    # br=torch.tensor([0.0,0.0,0.0,0,0])
    P=torch.eye(5) * 1e-8 
    P[0,0]=(env.theta[9]*0.05)**2 # sigma xx
    P[1,1]=(env.theta[10]*0.05)**2 # sigma yy
    reso=30

    with initiate_plot(15, 10, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # vary distance
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,0]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,0]=1. 
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,1)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.set_title('distance')

        # vary angle
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,1]=-1.5
        beliefr=beliefl.clone().detach()
        beliefr[0,1]=1.5 
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,2)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('angle')

        # vary v
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,2]=-1.
        beliefr=beliefl.clone().detach()
        beliefr[0,2]=1.     
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,3)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('forward v')

        # vary w
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,3]=-1.5
        beliefr=beliefl.clone().detach()
        beliefr[0,3]=1.5   
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,4)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('angular w')
        
        # vary time
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,4]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,4]=30.    
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,5)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('time dt')

        # vary prev v
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,5]=-1.
        beliefr=beliefl.clone().detach()
        beliefr[0,5]=1.    
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,6)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.set_title('previous forward ctrl')


        # vary prev w
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,6]=-1.5
        beliefr=beliefl.clone().detach()
        beliefr[0,6]=15   
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,7)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('previous angular ctrl')
        
        # vary xx
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,7]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,7]=0.2  
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,8)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('xx uncertainty')    
        
        # vary xy
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,8]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,8]=0.2 
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,9)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('xy uncertainty') 

        # vary yy
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,9]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,9]=0.2  
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,10)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('yy uncertainty')

        # vary x heading
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,10]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,10]=0.2 
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,11)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.set_title('x heading uncertainty')
        
        # vary y heading
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,11]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,11]=0.2  
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,12)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.yaxis.set_ticks([])
        ax.set_title('y heading uncertainty')

        # vary heading
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,12]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,12]=0.2
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,13)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.yaxis.set_ticks([])
        ax.set_title('heading uncertainty')

        # vary vv
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,13]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,13]=0.9 
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,14)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.yaxis.set_ticks([])
        ax.set_title('vv uncertainty')

        # vary ww
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,14]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,14]=0.9
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,15)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.yaxis.set_ticks([])
        ax.set_title('ww uncertainty')

        # legend and label
        teal_patch = mpatches.Patch(color=color_settings['v'], label='forward')
        orange_patch = mpatches.Patch(color=color_settings['w'], label='angular')
        ax.legend(handles=[teal_patch,orange_patch],loc='upper right',fontsize=6)

def policygiventhetav2(n):
    env.reset(phi=phi,theta=theta)
    bl=torch.zeros(5)
    # br=torch.tensor([0.0,0.0,0.0,0,0])
    P=torch.eye(5) * 1e-8 
    P[0,0]=(env.theta[9]*0.05)**2 # sigma xx
    P[1,1]=(env.theta[10]*0.05)**2 # sigma yy
    reso=30

    with initiate_plot(10, 20, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # vary ith
        for i in range(n):
            beliefl=env.wrap_decision_info().clone().detach()
            beliefl[0,i]=0.
            beliefr=beliefl.clone().detach()
            beliefr[0,i]=1. 
            delta=(beliefr-beliefl)/(reso-1)
            actions=[]
            with torch.no_grad():
                for n in range(reso):
                    b=beliefl+delta*n
                    action=agent(b)[0]
                    actions.append(action)
            actions=torch.stack(actions)
            ax = fig.add_subplot(10,5,i+1)
            ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
            ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
            ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axes.xaxis.set_ticks([0,0.5,1])
            ax.set_title('{}th'.format(i))


# inversed theta with error bar
def inversed_theta_bar(inverse_data):
    cov=theta_cov(inverse_data['Hessian'])
    theta_stds=stderr(cov)
    x_pos = np.arange(len(theta_names))
    data=[std/mean for std,mean in zip(theta_stds, theta_mean)]
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.bar(x_pos, data, color = 'tab:blue')
        ax.set_xticks(x_pos)
        ax.set_ylabel('inferred parameter std/mean')
        ax.set_xticklabels(theta_names,rotation=45,ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# inversed theta uncertainty by std/mean
def inversed_uncertainty_bar(inverse_data):
    cov=theta_cov(inverse_data['Hessian'])
    theta_stds=stderr(cov)
    x_pos = np.arange(len(theta_names))
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar(x_pos, torch.tensor(inverse_data['theta_estimations'][-1]).flatten(), 
                yerr=theta_stds,color = 'tab:blue')
        # title and axis names
        ax.set_ylabel('inferred parameter value')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(theta_names, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


# 2.1 overhead view, monkey's belief and state in one trial
def single_trial_overhead():
    # load data
    etask=input['mkdata']['task'][0]
    initv=input['mkdata']['actions'][0][0][0]  
    initw=input['mkdata']['actions'][0][0][1]    
    num_trials=input['num_trials']
    # agent_dev_costs=[]
    # agent_mag_costs=[]
    # create plot
    with initiate_plot(3, 3, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('world x, cm')
        ax1.set_ylabel('world y, cm')
        goalcircle = plt.Circle((etask[0],etask[1]), 0.13, color='y', alpha=0.5)
        ax1.add_patch(goalcircle)
        ax1.set_xlim([-0.1,1.1])
        ax1.set_ylim([-0.6,0.6])
        with torch.no_grad():
            for trial_i in range(num_trials):
                agent_actions=[]
                agent_beliefs=[]
                agent_covs=[]
                agent_states=[]
                env.reset(phi=phi,theta=input['theta'],goal_position=etask,vctrl=initv,wctrl=initw)
                epbliefs=[]
                epbcov=[]
                epactions=[]
                epstates=[]
                t=0
                done=False
                while not done:
                    action = input['agent'](env.decision_info)[0]
                    _,_,done,_=env.step(torch.tensor(action).reshape(1,-1)) 
                    epactions.append(action)
                    epbliefs.append(env.b)
                    epbcov.append(env.P)
                    epstates.append(env.s)
                    t=t+1
                print(env.get_distance(env.s))
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
                ax1.plot(estate[:,0],estate[:,1], color='r',alpha=0.5)
                # agent belief path
                for t in range(len(agent_beliefs[0][:,:,0])-1):
                    cov=agent_covs[0][t][:2,:2]
                    pos=  [agent_beliefs[0][:,:,0][t,0],
                            agent_beliefs[0][:,:,0][t,1]]
                    plot_cov_ellipse(cov, pos, nstd=2, color='orange', ax=ax1,alpha_factor=0.7/num_trials)
        # mk state, blue
        ax1.plot(input['mkdata']['states'][0][:,0],input['mkdata']['states'][0][:,1],alpha=0.5)


# 3.3 overhead view, skipped trials boundary similar
def agentvsmk_skip(num_trials=10):
    ind=torch.randint(low=100,high=222,size=(num_trials,))
    input={
        'agent':agent,
        'theta':theta,
        'phi':phi,
        'env': env,
        'num_trials':num_trials,
        # 'task':tasks[:222],
        'mkdata':{
                        'trial_index': ind,               
                        # 'task': [tasks[i][0] for i in ind],                  
                        # 'actions': [actions[i][0] for i in ind],                 
                        # 'states':[states[i][0] for i in ind],
                        'task': tasks,                 
                        'actions': actions,                 
                        'states':states,
                        },                      
        'use_mk_data':True
    }
    with suppress():
        res1=trial_data(input)
    # plotoverhead_skip(res1)
    plotoverhead(res1)

    input={
        'agent':agent,
        'theta':theta,
        'phi':phi,
        'env': env,
        'num_trials':num_trials,
        'task':[tasks[i] for i in ind],
        'mkdata':{},                      
        'use_mk_data':False
    }

    with suppress():
        res2=trial_data(input)
    plotoverhead(res2)
    # plotoverhead_skip(res2)


def plot_critic_polar(  dmin=0.1,
                        dmax=1.0):
    critic=agent_.critic.qf0.cpu()
    nbin=10
    with torch.no_grad():
        env.reset(phi=phi,theta=theta)
        base=env.decision_info.flatten().clone().detach()
        start=base.clone().detach();start[0]=dmin; start[1]=-0.75
        xend=start.clone().detach();yend=start.clone().detach()
        xend[0]=1;yend[1]=0.75
        xdelta=(xend-start)/nbin/2
        ydelta=(yend-start)/nbin
        values=[]
        for xi in range(2*nbin):
            yvalues=[]
            for yi in range(nbin):
                thestate=start+xi*xdelta+yi*ydelta
                value=critic(torch.cat([thestate,agent(thestate).flatten()]))
                yvalues.append(value.item())
            values.append(yvalues)
        values=np.asarray(values)
        values=normalizematrix(values)
        # plt.imshow(im,origin='lower')

    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        axrange=[-0.75 , 0.75, dmin , 1]
        c=ax.imshow(values, origin='lower',extent=axrange, aspect='auto')
        add_colorbar(c)
        ax.set_ylabel('distance')
        ax.set_xlabel('angle')

    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        c=ax.contourf(values, origin='lower',extent=axrange, aspect='auto')
        ax.set_ylabel('distance')
        ax.set_xlabel('angle')
        fig.colorbar(c, ax=ax)

    with initiate_plot(5, 3, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a,d = np.meshgrid(np.linspace(-0.75, 0.75, nbin), np.linspace(dmin, 1, 2*nbin)) 
        fig = plt.figure()
        ax = fig.add_subplot(111,polar='True')
        levels=np.linspace(0, 1, 10)
        c=ax.contourf(a, d, values,levels=levels,cmap="viridis",)
        c.collections[3].set_linewidth(4)  
        c.collections[3].set_color('red') 
        c.collections[3].set_linestyle('dotted')
        ax.set_theta_offset(pi/2)
        ax.grid(False)
        ax.set_thetamin(-43)
        ax.set_thetamax(43)
        fig.colorbar(c, ax=ax)
        maskvalue=values.copy()
        maskvalue[maskvalue>0.45] = None
        c=ax.contourf(a, d, maskvalue,levels=levels,cmap="viridis",)
        fig.tight_layout()


def plot_inverse_trend(inverse_data):
    # trend
    with initiate_plot(30, 2, 300) as fig:
        numparams=len(inverse_data['theta_estimations'][0])
        for n in range(numparams):
            ax=fig.add_subplot(1,numparams,n+1,)
            y=[t[n] for t in inverse_data['theta_estimations']]
            ax.plot(y)
            ax.set_ylim([min(parameter_range[n][0],min(y)[0]),max(parameter_range[n][1],max(y)[0])])
        plt.show()
    # grad
    with initiate_plot(30, 2, 300) as fig:
        numparams=len(inverse_data['grad'][0])
        for n in range(numparams):
            ax=fig.add_subplot(1,numparams,n+1,)
            y=[t[n] for t in inverse_data['grad']]
            ax.plot(y)
            maxvalue=max(abs(min(y)),abs(max(y)))
            # ax.set_ylim(-maxvalue,maxvalue)
            ax.set_ylim(-100,100)
            ax.plot([i for i in range(len(y))],[0]*len(y))
    plt.show()
    # loss
    with initiate_plot(10,5,300) as fig:
        ax=fig.add_subplot(111)
        ax.plot(inverse_data['loss'])
    plt.show()


# make the input hashmap
def input_formatter(**kwargs):
    result={}
    for key, value in kwargs.items():
        # print("{0} = {1}".format(key, value))
        result[key]=value
    return result

# suppress the output
@contextmanager
def suppress(out=True, err=False):
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield
            

def roughbisec(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        midval = a[mid]
        if midval < x:
            lo = mid+1
        elif midval > x: 
            hi = mid
        else:
            return mid
    return mid


def cart2pol(*args):
    if type(args[0])==list:
        x=args[0][0]; y=args[0][1]
    else:
        x=args[0]; y=args[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def xy2pol(*args, rotation=True): # rotated for the task
    if type(args[0])==list or type(args[0])==np.array:
        x=args[0][0]; y=args[0][1]
    else:
        x=args[0]; y=args[1]
    d = np.sqrt(x**2 + y**2)
    a = np.arctan2(y, x)-pi/2 if rotation else  np.arctan2(y, x)
    return d, a


# TODO return n most nearest trials
def similar_trials(ind, tasks, actions):
  indls=[]
  for i in range(len(tasks)):
      if tasks[i][0]>tasks[ind][0]-0.05 \
      and tasks[i][0]<tasks[ind][0]+0.05 \
      and tasks[i][1]>tasks[ind][1]-0.03 \
      and tasks[i][1]<tasks[ind][1]+0.03 \
      and actions[i][0][0]>actions[ind][0][0]-0.1 \
      and actions[i][0][1]<actions[ind][0][1]+0.1:
          indls.append(i)
  return indls


def normalizematrix(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_relative_r_ang(px, py, heading_angle, target_x, target_y):
    heading_angle = np.deg2rad(heading_angle)
    distance_vector = np.vstack([px - target_x, py - target_y])
    relative_r = np.linalg.norm(distance_vector, axis=0)
    
    relative_ang = heading_angle - np.arctan2(distance_vector[1],
                                              distance_vector[0])
    # make the relative angle range [-pi, pi]
    relative_ang = np.remainder(relative_ang, 2 * np.pi)
    relative_ang[relative_ang >= np.pi] -= 2 * np.pi
    return relative_r, relative_ang

        
@contextmanager
def initiate_plot(dimx=24, dimy=9, dpi=100, fontweight='normal'):
    plt.rcParams['figure.figsize'] = (dimx, dimy)
    plt.rcParams['font.weight'] = fontweight
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["font.family"] = 'Arial'
    fig = plt.figure(dpi=dpi)
    yield fig
    plt.show()
    
    
def set_violin_plot(bp, facecolor, edgecolor, linewidth=1, alpha=1, ls='-', hatch=r''):
    plt.setp(bp['bodies'], facecolor=facecolor, edgecolor=edgecolor, 
             linewidth=linewidth, alpha=alpha ,ls=ls, hatch=hatch)
    plt.setp(bp['cmins'], facecolor=facecolor, edgecolor=edgecolor, 
             linewidth=linewidth, alpha=alpha ,ls=ls)
    plt.setp(bp['cmaxes'], facecolor=facecolor, edgecolor=edgecolor, 
             linewidth=linewidth, alpha=alpha ,ls=ls)
    plt.setp(bp['cmedians'], facecolor='k', edgecolor='k', 
             linewidth=linewidth, alpha=alpha ,ls=ls)
    plt.setp(bp['cbars'], facecolor='None', edgecolor='None', 
             linewidth=linewidth, alpha=alpha ,ls=ls)
    
    
def set_box_plot(bp, color, linewidth=1, alpha=0.9, ls='-', unfilled=False):
    if unfilled is True:
        plt.setp(bp['boxes'], facecolor='None', edgecolor=color,
                 linewidth=linewidth, alpha=1,ls=ls)
    else:
        plt.setp(bp['boxes'], facecolor=color, edgecolor=color,
                 linewidth=linewidth, alpha=alpha,ls=ls)
    plt.setp(bp['whiskers'], color='k', linewidth=linewidth, alpha=alpha ,ls=ls)
    plt.setp(bp['caps'], color='k', linewidth=linewidth, alpha=alpha ,ls=ls)
    plt.setp(bp['medians'], color='k', linewidth=linewidth, alpha=alpha ,ls=ls)
    
    
def filter_fliers(data, whis=1.5, return_idx=False):
    filtered_data = []; fliers_ides = []
    for value in data:
        Q1, Q2, Q3 = np.percentile(value, [25, 50, 75])
        lb = Q1 - whis * (Q3 - Q1); ub = Q3 + whis * (Q3 - Q1)
        filtered_data.append(value[(value > lb) & (value < ub)])
        fliers_ides.append(np.where((value > lb) & (value < ub))[0])
    if return_idx:
        return filtered_data, fliers_ides
    else:
        return filtered_data
    
    
def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10**(-precision), precision)


def my_floor(a, precision=0):
    return np.round(a - 0.5 * 10**(-precision), precision)


def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    

def match_targets(df, reference):
    df.reset_index(drop=True, inplace=True)
    reference.reset_index(drop=True, inplace=True)
    df_targets = df.loc[:, ['target_x', 'target_y']].copy()
    reference_targets = reference.loc[:, ['target_x', 'target_y']].copy()

    closest_df_indices = []
    for _, reference_target in reference_targets.iterrows():
        distance = np.linalg.norm(df_targets - reference_target, axis=1)
        closest_df_target = df_targets.iloc[distance.argmin()]
        closest_df_indices.append(closest_df_target.name)
        df_targets.drop(closest_df_target.name, inplace=True)

    matched_df = df.loc[closest_df_indices]
    matched_df.reset_index(drop=True, inplace=True)
    
    return matched_df


def config_colors():
    colors = {'LSTM_c': 'olive', 'EKF_c': 'darkorange', 'monkV_c': 'indianred', 'monkB_c': 'blue',
              'sensory_c': '#29AbE2', 'belief_c': '#C1272D', 'motor_c': '#FF00FF',
              'reward_c': 'C0', 'unreward_c': 'salmon', 
              'gain_colors': ['k', 'C2', 'C3', 'C5', 'C9']}
    return colors


# Heissian polots
def sample_batch(states=None, actions=None, tasks=None, batch_size=20,**kwargs):
    totalsamples=len(tasks)
    sampleind=torch.randint(0,totalsamples,(batch_size,)) # trial inds
    sample_states=[states[i] for i in sampleind]
    sample_actions=[actions[i] for i in sampleind]
    sample_tasks=[tasks[i] for i in sampleind]
    return sample_states, sample_actions, sample_tasks


def compute_H_monkey(env, 
                    agent, 
                    theta_estimation, 
                    phi, 
                    H_dim=11, 
                    num_episodes=1,
                    num_samples=1,
                    action_var=0.1,
                    **kwargs):
    # TODO integrate sample batch
    states, actions, tasks=kwargs['monkeydata']
    totalsamples=len(tasks)
    sampleind=torch.randint(0,totalsamples,(num_episodes,)) # trial inds
    sample_states=[states[i] for i in sampleind]
    sample_actions=[actions[i] for i in sampleind]
    sample_tasks=[tasks[i] for i in sampleind]
    thistheta=torch.nn.Parameter(torch.Tensor(theta_estimation))
    phi=torch.Tensor(phi)
    phi.requires_grad=False
    loss = monkeyloss(agent, sample_actions, sample_tasks, phi, 
                        thistheta, env, 
                        action_var=action_var,
                        num_iteration=1, 
                        states=sample_states, 
                        samples=num_samples,
                        gpu=False)
    print('bp for grad')                    
    grads = torch.autograd.grad(loss, thistheta, create_graph=True,allow_unused=True)[0]
    H = torch.zeros(H_dim,H_dim)
    for i in range(H_dim):
        print('calculate {}th row of H'.format(i))
        H[i] = torch.autograd.grad(grads[i], thistheta, retain_graph=True,allow_unused=True)[0].view(-1)
    return H


def sample_all_tasks(
    na=10, 
    nd=10,
    matrix=True):
    angles=np.linspace(-0.75, 0.75, na)#; angles=torch.tensor(angles)
    distances=np.linspace(0.2, 1, nd)#; distances=torch.tensor(distances)
    if matrix:
        d,a=np.array(np.meshgrid(distances,angles))
        return d,a
    else:
        tasks=[]
        for a in angles:
            for d in distances:
                tasks.append(pol2xy(a,d))
        return tasks


def pol2xy(a,d):
    x=d*np.cos(a)
    y=d*np.sin(a)
    return [x,y]



def convert_2d_response(x, y, z, xmin, xmax, ymin, ymax, num_bins=20, kernel_size=3, isconvolve=True):
    @njit
    def compute(*args):
        x_bins = np.linspace(xmin - 1, xmax + 1, num_bins + 1)
        y_bins = np.linspace(ymin - 1, ymax + 1, num_bins + 1)

        x_indices = np.digitize(x, x_bins) - 1
        y_indices = np.digitize(y, y_bins) - 1

        data = np.zeros((num_bins, num_bins))
        count = data.copy()
        for z_idx, z_value in enumerate(z):
            data[y_indices[z_idx], x_indices[z_idx]] += z_value
            count[y_indices[z_idx], x_indices[z_idx]] += 1

        data /= count
        return x_bins, y_bins, data
    
    x_bins, y_bins, data = compute(x, y, z, xmin, xmax, ymin, ymax, num_bins)
    xx, yy = np.meshgrid(x_bins, y_bins)
    
    if isconvolve:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kernel = np.ones((kernel_size, kernel_size))
            data = convolve(data, kernel, boundary='extend')
    return xx, yy, data


def trial_data(input):
    '''
        trial data.
        based on input, get agent data (w/wo mk data)

        input:(
                agent,                              nn 
                env, gym                            obj
                number of sampled trials,           int
                use_mk_data,                        bool
                task                                list
                mkdata:{
                    trial index                  int
                    task,                        list
                    actions,                     array/list of lists/none
                    states}                      array/list of lists/none
                    
        log (
                monkey trial index,                 int/none
                monkey control,                     
                monkey position (s),
                agent control,
                agent position (s),
                belief mu,
                cov,
                theta,
                )
    '''
    # prepare variables
    result={
    'theta':input['theta'],
    'phi':input['phi'],
    'task':[],
    'num_trials':input['num_trials'],
    'agent_actions':[],
    'agent_beliefs':[],
    'agent_covs':[],
    'agent_states':[],
    'use_mk_data':input['use_mk_data'],
    'mk_trial_index':None,
    'mk_actions':[],
    'mk_states':[],
    }
    # check
    if input['use_mk_data'] and not input['mkdata']: #TODO more careful check
        warnings.warn('need to input monkey data when using monkey data')
    if 'task' not in input:
        warnings.warn('random targets')

    if input['use_mk_data']:
        result['mk_trial_index']=input['mkdata']['trial_index']
        result['task']=[input['mkdata']['task'][i] for i in input['mkdata']['trial_index']]
        result['mk_actions']=[input['mkdata']['actions'][i] for i in input['mkdata']['trial_index']]
        result['mk_states']=[input['mkdata']['states'][i] for i in input['mkdata']['trial_index']]

    else:
        if 'task' not in input:
            for trial_i in range(input['num_trials']):
                distance=torch.ones(1).uniform_(0.3,1)
                angle=torch.ones(1).uniform_(-pi/5,pi/5)
                task=[(torch.cos(angle)*distance).item(),(torch.sin(angle)*distance).item()]
                result['task'].append(task)
        else:
            if len(input['task'])==2:
                result['task']=input['task']*input['num_trials']
            else:
                result['task']=input['task']

    with torch.no_grad():
        for trial_i in range(input['num_trials']):
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
            if input['use_mk_data']: # using mk data
                t=0
                while t<len(result['mk_actions'][trial_i]):
                    action = agent(env.decision_info)[0]
                    if  t+1<result['mk_states'][trial_i].shape[0]:
                        env.step(torch.tensor(result['mk_actions'][trial_i][t]).reshape(1,-1),
                        next_state=result['mk_states'][trial_i][t+1].view(-1,1)) 
                    epactions.append(action)
                    epbliefs.append(env.b)
                    epbcov.append(env.P)
                    epstates.append(env.s)
                    t=t+1
                result['agent_beliefs'].append(torch.stack(epbliefs)) # the predicted belief of monkey
                result['agent_covs'].append(epbcov)   # the predicted uncertainty of monkey
                result['agent_actions'].append(torch.stack(epactions)) # the actions agent try to excute
                result['agent_states'].append(torch.stack(epstates)[:,:,0]) # will be same as mk states
            else: # not using mk data
                done=False
                t=0
                while not done:
                    action = agent(env.decision_info)[0]
                    if 'states' in input['mkdata']:
                        _,_,done,_=env.step(torch.tensor(action).reshape(1,-1),
                        next_state=input['mkdata']['states'][trial_i][t]) 
                    else:
                        _,_,done,_=env.step(torch.tensor(action).reshape(1,-1)) 
                    epactions.append(action)
                    epbliefs.append(env.b)
                    epbcov.append(env.P)
                    epstates.append(env.s)
                    t=+1
                result['agent_actions'].append(torch.stack(epactions))
                result['agent_beliefs'].append(torch.stack(epbliefs)[:,:,0])
                result['agent_covs'].append(epbcov)
                print(torch.stack(epstates))
                result['agent_states'].append(torch.stack(epstates))
    return result


def inverse_trajectory_monkey(theta_trajectory,
                    env=None,
                    agent=None,
                    phi=None, 
                    background_data=None, 
                    background_contour=False,
                    number_pixels=10,
                    background_look='contour',
                    ax=None, 
                    num_episodes=2,
                    loss_sample_size=2, 
                    H=None,
                    action_var=0.1,
                    **kwargs):
    '''    
        plot the inverse trajectory in 2d pc space
        -----------------------------
        input:
        theta trajectory: list of list(theta)
        method: PCA or other projections
        -----------------------------
        output:
        background contour array and figure
    '''
    with torch.no_grad():

        # plot trajectory
        fig = plt.figure(figsize=[3, 3])
        ax = fig.add_subplot()
        data_matrix=column_feature_data(theta_trajectory)
        mu=np.mean(data_matrix,0)
        try:
            score, evectors, evals = pca(data_matrix)
        except np.linalg.LinAlgError:
            score, evectors, evals = pca(data_matrix)


        row_cursor=0
        while row_cursor<score.shape[0]-1:
            row_cursor+=1
            ax.plot(score[row_cursor-1:row_cursor+1,0],
                    score[row_cursor-1:row_cursor+1,1],
                    '-',
                    linewidth=0.1,
                    color='g')
        # plot log likelihood contour
        sample_states, sample_actions, sample_tasks=sample_batch(states=states,actions=actions,tasks=tasks,batch_size=loss_sample_size)
        loss_function=monkey_loss_wrapped(env=env, 
        agent=agent, 
        phi=phi, 
        states=sample_states,
        actions=sample_actions,
        action_var=action_var,
        tasks=sample_tasks,
        num_episodes=num_episodes)
        finaltheta=score[row_cursor, 0], score[row_cursor, 1]
        current_xrange=list(ax.get_xlim())
        current_xrange[0]-=0.1
        current_xrange[1]+=0.1
        current_yrange=list(ax.get_ylim())
        current_yrange[0]-=0.1
        current_yrange[1]+=0.1
        maxaxis=max(abs(current_xrange[0]-finaltheta[0]),abs(current_xrange[1]-finaltheta[0]),abs(current_yrange[0]-finaltheta[1]),abs(current_yrange[1]-finaltheta[1]))
        xyrange=[[-maxaxis,maxaxis],[-maxaxis,maxaxis]]
        print('start background')
        if background_contour:
            background_data=plot_background(
                ax, 
                xyrange,
                mu, 
                evectors, 
                loss_function, 
                number_pixels=number_pixels,
                look=background_look,
                background_data=background_data)

        # plot theta inverse trajectory
        row_cursor=0
        while row_cursor<score.shape[0]-1:
            row_cursor+=1
            ax.plot(score[row_cursor-1:row_cursor+1,0],
                    score[row_cursor-1:row_cursor+1,1],
                    '-',
                    linewidth=0.5,
                    color='w') # line
            if row_cursor%20==0 or row_cursor==1:
                ax.quiver(score[row_cursor-1, 0], score[row_cursor-1, 1],
                        score[row_cursor, 0]-score[row_cursor-1, 0], score[row_cursor, 1]-score[row_cursor-1, 1],
                        angles='xy',color='w',scale=0.2,width=1e-2, scale_units='xy') # arrow
        ax.scatter(score[row_cursor, 0], score[row_cursor, 1], marker=(5, 1), s=200, color=[1, .5, .5])
        ax.set_xlabel('projected parameters')
        ax.set_ylabel('projected parameters')
        ax.set_xticks([-0.3,0,0.3])
        ax.set_yticks([-0.3,0,0.3])
        # plot hessian
        if H is not None:
            cov=theta_cov(H)
            cov_pc=evectors[:,:2].transpose()@np.array(cov)@evectors[:,:2]
            plot_cov_ellipse(cov_pc,pos=score[-1,:2],alpha_factor=0.5,ax=ax)

        return background_data

def plot_background(
    ax, 
    xyrange,
    mu, 
    evectors, 
    loss_function, 
    number_pixels=10, 
    look='contour', 
    alpha=1,
    background_data=None,
    **kwargs):
    with torch.no_grad():
        X,Y=np.meshgrid(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels),np.linspace(xyrange[1][0],xyrange[1][1],number_pixels))
        if background_data is None:
            background_data=np.zeros((number_pixels,number_pixels))
            for i,u in enumerate(np.linspace(xyrange[1][0],xyrange[1][1],number_pixels)):
                for j,v in enumerate(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels)):
                    score=np.array([u,v])
                    reconstructed_theta=score@evectors.transpose()[:2,:]+mu
                    reconstructed_theta=torch.tensor(reconstructed_theta).float()
                    reconstructed_theta.clamp(0.001,3)
                    background_data[i,j]=loss_function(reconstructed_theta)
                    print(background_data[i,j],reconstructed_theta)
        if look=='contour':
            c=ax.contourf(X,Y,background_data.transpose(),alpha=alpha,zorder=1)
            plt.colorbar(c,ax=ax)
        # elif look=='pixel':
        #     im=ax.imshow(X,Y,background_data,alpha=alpha,zorder=1)
        #     add_colorbar(im)
        return background_data


# TODO
def plot_background_par(
    ax, 
    xyrange,
    mu, 
    evectors, 
    loss_function, 
    number_pixels=10, 
    look='contour', 
    alpha=1,
    background_data=None,
    **kwargs):

    def _cal(u,v,i,j):
        with torch.no_grad():
            score=np.array([u,v])
            reconstructed_theta=score@evectors.transpose()[:2,:]+mu
            reconstructed_theta=torch.tensor(reconstructed_theta).float()
            reconstructed_theta.clamp(0.001,3)
            background_data[i,j]=loss_function(reconstructed_theta)

    def _cal(u,v,i,j,background_data):
        print([u,v,i,j])


    X,Y=np.meshgrid(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels),np.linspace(xyrange[1][0],xyrange[1][1],number_pixels))
    if background_data is None:
        background_data=np.zeros((number_pixels,number_pixels))
        pool = multiprocessing.Pool(4)
        jobparam=[(u,v,i,j,background_data) for (i,u),(j,v) in zip(enumerate(np.linspace(xyrange[1][0],xyrange[1][1],number_pixels)),enumerate(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels)))]
        pool.map_async(_cal, (job for job in jobparam)).get(timeout=10)

            
        if look=='contour':
            c=ax.contourf(X,Y,background_data.transpose(),alpha=alpha,zorder=1)
            fig.colorbar(c,ax=ax)
    return background_data


    pool = multiprocessing.Pool(4)
    jobs=list(range(10))
    def _cal(x):
        return x**2
    res=pool.map_async(_cal, (job for job in jobs))
    w = sum(res.get(timeout=10))
    print(w)

    import multiprocessing
    def start_process():
        print ('Starting', multiprocessing.current_process().name)

    pool_size =2
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=start_process,
                                )

    pool_outputs = pool.map(_cal,jobs)
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks


def monkey_loss_wrapped(
                env=None, 
                agent=None, 
                phi=None, 
                actions=None,
                tasks=None,
                states=None,
                num_episodes=10,
                gpu=False,
                action_var=0.001,
                **kwargs):
  new_function= lambda theta_estimation: monkeyloss(
            agent=agent, 
            actions=actions, 
            tasks=tasks, 
            phi=phi, 
            theta=theta_estimation, 
            env=env,
            num_iteration=1, 
            states=states, 
            samples=num_episodes, 
            action_var=action_var,
            gpu=gpu)
    
  return new_function

def plot_background_ev(
    ax, 
    evector,
    loss_function, 
    number_pixels=10, 
    alpha=0.5,
    scale=1.,
    background_data=None,
    **kwargs):
    ev1=evector[:,0].view(-1,1)
    ev2=evector[:,1].view(-1,1)
    if background_data is None:
        background_data=np.zeros((number_pixels,number_pixels))
        with torch.no_grad():
            X=np.linspace(theta-ev1*0.5*scale,theta+ev1*0.5*scale,number_pixels)
            Y=np.linspace(theta-ev2*0.5*scale,theta+ev2*0.5*scale,number_pixels)
            for i in range(number_pixels):
                for j in range(number_pixels):
                        reconstructed_theta=theta+X[i]+Y[j]
                        if torch.all(reconstructed_theta>0):
                            background_data[i,j]=loss_function(reconstructed_theta)
                        else:
                            background_data[i,j]=None
            plt.contourf(background_data,alpha=alpha)
        return background_data


def is_skip(task,state,dthreshold=0.26,sthreshold=None):
    if sthreshold:
        return (dthreshold<torch.norm(torch.tensor(task)-state[:2,-1]) and 
            sthreshold>torch.norm(torch.tensor(task)-state[:2,-1]))
    else:
        return dthreshold<torch.norm(torch.tensor(task)-state[:2,-1])




#-----control curve plots----------------------------------------------------------------

def plotctrl(input,**kwargs):
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if 'ax' in kwargs:
            ax = kwargs['ax'] 
        else:
            ax = fig.add_subplot(111)
        # ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time, dt')
        ax.set_ylabel('control magnitude')
        # plot data
        for trial_i in range(input['num_trials']):
            ax.plot(input['agent_actions'][trial_i][:,0],color=color_settings['v'],alpha=1/input['num_trials']**.3)
            ax.plot(input['agent_actions'][trial_i][:,1],color=color_settings['w'],alpha=1/input['num_trials']**.3)
        # legend and label
        teal_patch = mpatches.Patch(color=color_settings['v'], label='forward')
        orange_patch = mpatches.Patch(color=color_settings['w'], label='angular')
        ax.legend(handles=[teal_patch,orange_patch],loc='upper right',fontsize=6)
        ax.set_xlim(0,40)
        ax.set_ylim(-1,1)
        return  ax

#----overhead----------------------------------------------------------------
def plotoverhead(input,**kwargs):
    '''
        input:
        'theta':            input['theta'],
        'phi':              input['phi'],
        'agent_actions':    [],
        'agent_beliefs':    [],
        'agent_covs':       [],
        'agent_states':     [],
        'use_mk_data':      input['use_mk_data'],
        'mk_trial_index':   None,
        'mk_actions':       [],
        'mk_states':        [],
    '''
    pathcolor='gray' if 'color' not in kwargs else kwargs['color'] 
    fontsize = 9
    
    if 'ax' in kwargs:
        ax = kwargs['ax'] 
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
        
        for trial_i in range(input['num_trials']):
            input['agent_states'][trial_i][:,0]=input['agent_states'][trial_i][:,0]-input['agent_states'][trial_i][0,0]
            input['agent_states'][trial_i][:,1]=input['agent_states'][trial_i][:,1]-input['agent_states'][trial_i][0,1]
            ax.plot(-input['agent_states'][trial_i][:,1]*400,input['agent_states'][trial_i][:,0]*400, 
                c=pathcolor, lw=0.1, ls='-')
            # calculate if rewarded
            dy=input['agent_states'][trial_i][-1,0]*400-input['task'][trial_i][0]*400
            dx=input['agent_states'][trial_i][-1,1]*400-input['task'][trial_i][1]*400
            d2goal=(dx**2+dy**2)**0.5
            if d2goal<=65: # reached goal
                ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                    c='g', marker='.', s=1, lw=1)
            elif d2goal>65 and d2goal< 130:
                ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                    c='k', marker='.', s=1, lw=1)
                # yellow error line
                ax.plot([-input['agent_states'][trial_i][-1,1]*400,-input['task'][trial_i][1]*400],
                            [input['agent_states'][trial_i][-1,0]*400,input['task'][trial_i][0]*400],color='yellow',alpha=0.3, linewidth=1)

            else: # skipped
                ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                    c='r', marker='.', s=1, lw=1)

    else:
        with initiate_plot(1.8, 1.8, 300) as fig:
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
            ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
            x_temp = np.linspace(-235, 235)
            ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
            ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
            ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
            ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
            ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
            ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
            ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)
            fig.tight_layout(pad=0)
            ax.plot(np.linspace(0, 230 + 7),
                    np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
            
            for trial_i in range(input['num_trials']):
                input['agent_states'][trial_i][:,0]=input['agent_states'][trial_i][:,0]-input['agent_states'][trial_i][0,0]
                input['agent_states'][trial_i][:,1]=input['agent_states'][trial_i][:,1]-input['agent_states'][trial_i][0,1]
                ax.plot(-input['agent_states'][trial_i][:,1]*400,input['agent_states'][trial_i][:,0]*400, 
                    c=pathcolor, lw=0.1, ls='-')
                # calculate if rewarded
                dy=input['agent_states'][trial_i][-1,0]*400-input['task'][trial_i][0]*400
                dx=input['agent_states'][trial_i][-1,1]*400-input['task'][trial_i][1]*400
                d2goal=(dx**2+dy**2)**0.5
                if d2goal<=65: # reached goal
                    ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                        c='g', marker='.', s=1, lw=1)
                elif d2goal>65 and d2goal< 130:
                    ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                        c='k', marker='.', s=1, lw=1)
                    # yellow error line
                    ax.plot([-input['agent_states'][trial_i][-1,1]*400,-input['task'][trial_i][1]*400],
                                [input['agent_states'][trial_i][-1,0]*400,input['task'][trial_i][0]*400],color='yellow',alpha=0.3, linewidth=1)

                else: # skipped
                    ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                        c='r', marker='.', s=1, lw=1)
    return ax

def plotoverheadcolor(input,**kwargs):
    linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else 0.1
    alpha=kwargs['alpha']  if 'alpha' in kwargs else 1
    fontsize = 9
    target_idexes = np.arange(1500, 2000)
    if 'ax' in kwargs:
        ax = kwargs['ax'] 
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
        # setting color                
        colors = pl.cm.Set1(np.linspace(0,1,input['num_trials']))
        for trial_i in range(input['num_trials']):
            ax.plot(-input['agent_states'][trial_i][:,1]*400,input['agent_states'][trial_i][:,0]*400, 
                color=colors[trial_i], lw=0.1, ls='-',linewidth=linewidth,alpha=alpha)
            ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                color=colors[trial_i], marker='.', s=1, lw=1)

        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)

    else:
        with initiate_plot(1.8, 1.8, 300) as fig:
            ax = fig.add_subplot(111) 
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
            ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
            
            ax.plot(np.linspace(0, 230 + 7),
                    np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
            # setting color                
            colors = pl.cm.Set1(np.linspace(0,1,input['num_trials']))
            for trial_i in range(input['num_trials']):
                ax.plot(-input['agent_states'][trial_i][:,1]*400,input['agent_states'][trial_i][:,0]*400, 
                    color=colors[trial_i], lw=0.1, ls='-',linewidth=linewidth,alpha=alpha)
                ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                    color=colors[trial_i], marker='.', s=1, lw=1)

            x_temp = np.linspace(-235, 235)
            ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
            ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
            ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
            
            ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
            ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
            ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
            ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)

            fig.tight_layout(pad=0)
    return  ax


def plotoverhead_skip(input):
    '''
        input:
        'theta':            input['theta'],
        'phi':              input['phi'],
        'agent_actions':    [],
        'agent_beliefs':    [],
        'agent_covs':       [],
        'agent_states':     [],
        'use_mk_data':      input['use_mk_data'],
        'mk_trial_index':   None,
        'mk_actions':       [],
        'mk_states':        [],
    '''
    fontsize = 9
    target_idexes = np.arange(1500, 2000)
    with initiate_plot(1.8, 1.8, 300) as fig:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
        
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
        
        for trial_i in range(input['num_trials']):
            ax.plot(-input['agent_states'][trial_i][:,1]*400,input['agent_states'][trial_i][:,0]*400, 
                c='gray', lw=0.1, ls='-')
            # calculate if rewarded
            dx=input['agent_states'][trial_i][-1,1]*400-input['task'][trial_i][1]*400
            dy=input['agent_states'][trial_i][-1,0]*400-input['task'][trial_i][0]*400
            d2goal=(dx**2+dy**2)**0.5
            d2start=(input['agent_states'][trial_i][-1,1]**2+input['agent_states'][trial_i][-1,0]**2)**0.5*400
            if d2goal<65:
                ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                    c='k', marker='.', s=1, lw=1)
            elif d2goal>200: # skipped
                ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                    c='r', marker='.', s=1, lw=1)
            else:
                ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                    c='k', marker='.', s=1, lw=1)
  
        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)

        fig.tight_layout(pad=0)


# change theta in direction of mattered most eigen vector 
def matter_most_eigen_direction():
    theta_mean=torch.tensor([[0.5],
            [1.57],
            [0.5],
            [0.5],
            [0.5],
            [0.5],
            [0.13],
            [0.5],
            [0.5],
            [0.5],
            [0.5],
    ])
    mattermost=evector[0]
    # matterleast=evector[-1]
    diagnose_plot_theta(agent, env, phi, theta_init, theta_mean+0.1*mattermost.view(-1,1),5)
    # diagnose_plot_theta(agent, env, phi, theta_init, theta_mean+3*matterleast.view(-1,1),3)

# describes data df
def histdfcol():
    seen={}
    for each in df.floor_density:
        if each in seen:
            seen[each]+=1
        else:
            seen[each]=1

    def itkey(keys):
        for key in keys:
            yield key

    keys=['gain_v',
    'gain_w',
    'perturb_vpeakmax',
    'perturb_wpeakmax',
    'perturb_sigma',
    'perturb_dur',
    'perturb_vpeak',
    'perturb_wpeak',
    'perturb_start_time',
    'floor_density',
    'pos_r_end',
    'pos_theta_end',
    'target_x',
    'target_y',
    'target_r',
    'target_theta',
    'full_on',
    'rewarded',
    'trial_dur',
    'relative_radius_end',
    'relative_angle_end',
    'category',]

    keygen=itkey(keys)

    key=keygen.__next__()

    plt.hist(df[key])
    plt.title(key)
    plt.show()


def colorgrad_line(x,y,z,ax=None, linewidth=2):
    x,y,z=np.array(x),np.array(y),np.array(z)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors=np.array(z)
    if ax:
        lc = LineCollection(segments, cmap='viridis')
        lc.set_array(colors)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)
        plt.colorbar(line,ax=ax)
    else:
        fig, ax = plt.subplots()
        lc = LineCollection(segments, cmap='viridis')
        lc.set_array(colors)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)
        fig.colorbar(line,ax=ax)
    return ax

def colorgrad_inverse_traj(inverse_data):
        # plot trajectory
        fig = plt.figure(figsize=[3, 3])
        ax = fig.add_subplot()
        data_matrix=column_feature_data(inverse_data['theta_estimations'])
        mu=np.mean(data_matrix,0)
        try:
            score, evectors, evals = pca(data_matrix)
        except np.linalg.LinAlgError:
            score, evectors, evals = pca(data_matrix)
        x=score[:,0]
        y=score[:,1]
        z=inverse_data['loss']
        z=np.array(z)*-1
        z= (z-min(z))/(max(z)-min(z))
        # z=np.log(z)
        ax.set_xlim(min(x),max(x))
        ax.set_ylim(min(y),max(y))
        colorgrad_line(x,y,z,ax=ax)
        ax.set_xlabel('projected parameters')
        ax.set_ylabel('projected parameters')
        ax.set_title('inverse trajectory, color for likelihood')

# inferred obs vs density
def obsvsdensity(noiseparam):
    errbar=False
    x,y=[],[]
    err=[]
    for k,v in noiseparam.items():
        x.append(k)
        y.append(v['param'])
        if 'std' in v:
            err.append(v['std'])
            errbar=True
    y=np.array(y)[:,:,0]
    err=np.array(err)
    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    lables=['pro v','pro w','obs v','obs w']
    if errbar:
        for i in range(4):
            ax.errorbar( [i for i in range(4)],y[:,i],err[:,i],label=lables[i])
    else:
        for i in range(4):
            ax.plot(y[:,i],label=lables[i])
    ax.set_xticks([i for i in range(4)])
    ax.set_xticklabels(["0.0001", "0.0005", "0.001", "0.005"])
    ax.legend(fontsize=16)
    ax.set_ylabel('noise std',fontsize=16)
    ax.set_xlabel('density',fontsize=16)
    ax.set_title('inferred noise level vs ground density',fontsize=20)

    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    ax.plot(y[:,0]/y[:,2],label='forward v')
    ax.plot(y[:,1]/y[:,3],label='angular w')
    # ax.set_xscale('log')
    ax.set_xticks([i for i in range(4)])
    ax.set_xticklabels(["0.0001", "0.0005", "0.001", "0.005"])
    ax.legend(fontsize=16)
    ax.set_ylabel('noise std',fontsize=16)
    ax.set_xlabel('density',fontsize=16)
    ax.set_title('inferred observation reliable degree vs ground density',fontsize=20)

if __name__=='__main__':
    env=ffacc_real.FireFlyPaper(arg)
    agent_=TD3_torch.TD3.load('trained_agent/paper.zip')
    agent=agent_.actor.mu.cpu()


    # load monkey data
    print('loading data')
    note='bnorm4rand'
    with open("C:/Users/24455/Desktop/bruno_normal_downsample",'rb') as f:
            df = pickle.load(f)
    df=datawash(df)
    df=df[df.category=='normal']
    # df=df[df.target_r>250]
    df=df[df.floor_density==0.005]
    # floor density are in [0.0001, 0.0005, 0.001, 0.005]
    df=df[:-100]
    print('process data')
    states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
    print('done process data')
    sample_states, sample_actions, sample_tasks=sample_batch(states=states,actions=actions,tasks=tasks,batch_size=20)
    loss_function=monkey_loss_wrapped(env=env, 
            agent=agent, 
            phi=phi, 
            num_episodes=5,
            action_var=0.001,
            states=sample_states,
            actions=sample_actions,
            tasks=sample_tasks,)



    # load inverse data
    inverse_data=load_inverse_data("testdcont2_16_45")
    theta_trajectory=inverse_data['theta_estimations']
    theta_estimation=theta_trajectory[-1]
    theta=torch.tensor(theta_estimation)
    phi=torch.tensor(inverse_data['phi'])
    H=inverse_data['Hessian'] 
    print(H)
    plot_inverse_trend(inverse_data)
    colorgrad_inverse_traj(inverse_data)


    # vis the inverses of differnt density
    inverse_data=load_inverse_data("testdefstart2_12_43")
    plot_inverse_trend(inverse_data)
    inverse_data=load_inverse_data("test2_12_40")
    plot_inverse_trend(inverse_data)

    inversedensityfiles=[
        'varfixnorm12_0_51',
        'varfixnorm22_0_51',
        'varfixnorm32_0_52',
        'varfixnorm42_0_52',
    ]

    noiseparam={}
    for eachdensity,inversefile in zip([0.0001, 0.0005, 0.001, 0.005],inversedensityfiles):
        noiseparam[eachdensity]={}
        noiseparam[eachdensity]['param']=load_inverse_data(inversefile)['theta_estimations'][-2][2:6]
        H=load_inverse_data(inversefile)['Hessian']
        if H!=[]:
            noiseparam[eachdensity]['std']=stderr(torch.inverse(H))[2:6]
    obsvsdensity(noiseparam)



    with open("C:/Users/24455/Desktop/bruno_normal_downsample",'rb') as f:
        df = pickle.load(f)
        df=datawash(df)
        df=df[df.category=='normal']
        for eachdensity,inversefile in zip([0.0001, 0.0005, 0.001, 0.005],inversedensityfiles):
            tempdf=df[df.floor_density==eachdensity]
            states, actions, tasks=monkey_data_downsampled(tempdf,factor=0.0025)
            sample_states, sample_actions, sample_tasks=sample_batch(states=states,actions=actions,tasks=tasks,batch_size=20)
            theta=torch.tensor(load_inverse_data(inversefile)['theta_estimations'][-2])
            _loss_function=monkey_loss_wrapped(env=env, 
                agent=agent, 
                phi=phi, 
                num_episodes=5,
                action_var=0.001,
                states=sample_states,
                actions=sample_actions,
                tasks=sample_tasks,)
            H=torch.autograd.functional.hessian(_loss_function,theta,strict=True)
            H=H[:,0,:,0]   
            noiseparam[eachdensity]={}  
            noiseparam[eachdensity]['std']=stderr(torch.inverse(H))[2:6]   
            inverse_data=load_inverse_data(inversefile)
            inverse_data['Hessian']=H    
            save_inverse_data(inverse_data)

    for inversefile in  inversedensityfiles:
        theta=torch.tensor(load_inverse_data(inversefile)['theta_estimations'][-2])
        H=inverse_data['Hessian']
        print(stderr(torch.inverse(H)))


    # count reward %
    for eachdensity in [0.0001, 0.0005, 0.001, 0.005]:  
        print('density {}, reward% '.format(eachdensity),len(df[df.floor_density==eachdensity][df.rewarded])/len(df[df.floor_density==eachdensity]))


    # calculate H

    H=torch.autograd.functional.hessian(loss_function,theta,strict=True)
    H=H[:,0,:,0]
    stderr(torch.inverse(H))          
    inverse_data['Hessian']=H
    save_inverse_data(inverse_data)
    ev, evector=torch.eig(H,eigenvectors=True)
    env=ffacc_real.FireFlyPaper(arg)

    # background in eigen axis
    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        sample_states, sample_actions, sample_tasks=sample_batch(states=states,actions=actions,tasks=tasks,batch_size=20)
        bk=plot_background_ev(
        ax, 
        evector,
        loss_function=monkey_loss_wrapped(env=env, 
            agent=agent, 
            phi=phi, 
            num_episodes=50,
            action_var=1e-2,
            states=sample_states,
            actions=sample_actions,
            tasks=sample_tasks,), 
        number_pixels=5, 
        alpha=0.5,
        scale=0.01,
        background_data=None)
        
    # loss vs first eigen
    scale=0.1
    ev1=evector[0].view(-1,1)
    X=np.linspace(theta-ev1*0.5*scale,theta+ev1*0.5*scale,number_pixels)
    logll1d=[]
    for onetheta in X:
        onetheta=torch.tensor(onetheta)
        with torch.no_grad():
            logll1d.append(loss_function(onetheta))
    plt.plot(logll1d)

    # inverse_data['grad']

    # load agent
    agent_=TD3_torch.TD3.load('trained_agent/paper.zip')
    agent=agent_.actor.mu.cpu()
    policygiventheta()
    policygiventhetav2(env.decision_info.shape[-1])
    agentvsmk_skip(55)
    plot_critic_polar()
    ind=torch.randint(low=100,high=5000,size=(1,))
    input={
        'agent':agent,
        'theta':theta,
        'phi':phi,
        'env': env,
        'num_trials':10,
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

    ind=torch.randint(low=100,high=1111,size=(1,))
    indls=similar_trials(ind, tasks, actions)
    indls=indls[:20]
    input={
        'agent':agent,
        'theta':theta,
        'phi':phi,
        'env': env,
        'num_trials':20,
        # 'task':tasks[:20],
        'mkdata':{
                        'trial_index': indls,               
                        'task': tasks,                  
                        'actions': actions,                 
                        'states':states,
                        },                      
        'use_mk_data':True
    }
    with suppress():
        res=trial_data(input)
    plotoverhead(res)
    plotctrl(res)


    number_pixels=5
    background_data=inverse_trajectory_monkey(
                        theta_trajectory,
                        env=env,
                        agent=agent,
                        phi=phi,
                        background_data=None,
                        H=None,
                        background_contour=True,
                        number_pixels=number_pixels,
                        background_look='contour',
                        ax=None,
                        action_var=0.2,
                        loss_sample_size=5,
                        num_episodes=13)
    inverse_data['background_data']={number_pixels:background_data}
    save_inverse_data(inverse_data)


    c=plt.imshow(background_data)
    plt.colorbar(c)

    # 1. warpped up of agents with diff theta in a row
    phi=torch.tensor([[0.5000],
            [1.57],
            [0.01],
            [0.01],
            [0.01],
            [0.01],
            [0.13],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
    ])
    theta_init=torch.tensor([[0.5000],
            [1.57],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
            [0.01],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
    ])
    theta_final=torch.tensor([[0.5000],
            [1.57],
            [0.1],
            [0.1],
            [0.1],
            [0.6],
            [0.2],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
    ])
    env.debug=True
    diagnose_plot_theta(agent, env, phi, theta_init, theta_final,5)





    # 1. showing converge
    # 1.1 theta uncertainty from heissian shows convergence
    # 1.2 log likelihood surface in pc space, with our path and final uncertainty

    #---------------------------------------------------------------------
    # inversed theta with error bar
    # inversed_theta_bar(inverse_data)

    #---------------------------------------------------------------------
    # inversed theta uncertainty by std/mean
    # inversed_uncertainty_bar(inverse_data)
    #---------------------------------------------------------------------


    # 1.4 differnt inverse rans and final thetas in pc space, each with uncertainty

    # 2. reconstruction belief from inferred params



    # 2.1 overhead view, monkey's belief and state in one trial
    ind=torch.randint(low=100,high=5000,size=(1,))
    input={
        'agent':agent,
        'theta':theta,
        'phi':phi,
        'env': env,
        'num_trials':3,
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
    # plot one monkey trial using index
    ind=torch.randint(low=100,high=300,size=(1,))
    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.plot(states[ind][:,0],states[ind][:,1], color='r',alpha=0.5)
        goalcircle = plt.Circle([tasks[ind][0],tasks[ind][1]], 0.13, color='y', alpha=0.5)
        ax.add_patch(goalcircle)
        ax.set_xlim(0,1)
        ax.set_ylim(-0.6,0.6)



    #---------------------------------------------------------------------
    # 3. monkey and inferred agent act similar on similar trials
    ind=torch.randint(low=0,high=888,size=(1,))
    indls=similar_trials(ind, tasks, actions)
    indls=indls[:20]
    with torch.no_grad():
        inputirc={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':20,
            'task':[tasks[i] for i in indls],
            'mkdata':{
                            },                      
            'use_mk_data':False
        }
        with suppress():
            resirc=trial_data(inputirc)

        inputmk={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':20,
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
    ax=plotoverhead(resirc, color='orange')
    # plotctrl(resirc)
    plotoverhead(resmk,ax=ax, color='tab:blue')
    # plotctrl(resmk)
    ax.get_figure()


    #---------------------------------------------------------------------
    # 3. monkey and inferred agent act similar on similar trials, given mk states
    ind=torch.randint(low=0,high=999,size=(1,))
    indls=similar_trials(ind, tasks, actions)
    indls=indls[:20]
    with torch.no_grad():
        inputirc={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':20,
            'task':[tasks[i] for i in indls],
            'mkdata':{           
                            'states':states,
                            },                      
            'use_mk_data':False}
        with suppress():
            resirc=trial_data(inputirc)
        plotoverhead(resirc)
        plotctrl(resirc)
        inputmk={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':20,
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
        plotoverhead(resmk)
        plotctrl(resmk)

    alllogll=[]
    for true, est in zip(resmk['mk_actions'],resmk['agent_actions']):
        eplogll=[0.]
        for t,e in zip(true,est):
            eplogll.append(torch.sum(logll(t,e,std=0.01)))
        alllogll.append(sum(eplogll))
    print(np.mean(alllogll))

    for a in resmk['mk_actions']:
        plt.plot(a[:,0],c='tab:blue',alpha=0.5)
    for a in resmk['agent_actions']:
        plt.plot(a[:,0],c='tab:orange',alpha=0.5)

    for a in resmk['mk_actions']:
        plt.plot(a[:,1],c='royalblue',alpha=0.5)
    for a in resmk['agent_actions']:
        plt.plot(a[:,1],c='orangered',alpha=0.5)
    plt.xlabel('time [dt]')
    plt.ylabel('control v and w')


    plt.plot(actions[indls[-1]])
    plt.plot(resmk['agent_actions'][-1])

    true=actions[indls[-1]]
    est=resmk['agent_actions'][-1]

    sumlogll=0.
    for t,e in zip(true,est):
        sumlogll+=logll(t,e,std=0.001)


    # use irc action
    ind=torch.randint(low=100,high=1111,size=(1,))
    indls=similar_trials(ind, tasks, actions)
    indls=indls[:20]
    maxlen=max([len(actions[i]) for i in indls])
    for ind in indls:
        env.reset(goal_position=tasks[ind],theta=theta,phi=phi)
        vw=[]
        done=False
        with torch.no_grad():
            while not done and env.trial_timer<=maxlen:
                action=agent(env.decision_info)
                vw.append(action)
                _,_,done,_=env.step(action)
        vw=torch.stack(vw)[:,0,:]

        plt.plot(vw,c='orange',alpha=0.5)
        plt.plot(actions[ind],c='tab:blue',alpha=0.5)
    plt.plot(vw,c='orange',alpha=1,label='agent')
    plt.plot(actions[ind],c='tab:blue',alpha=1,label='monkey')
    plt.xlabel('time [dt]')
    plt.ylabel('control v and w')
    plt.legend()


    # use mkaction, same as in inverse
    ind=torch.randint(low=100,high=1111,size=(1,))
    indls=similar_trials(ind, tasks, actions)
    indls=indls[:20]
    for ind in indls:
        env.reset(goal_position=tasks[ind],theta=theta,phi=phi)
        vw=[]
        done=False
        mkactionep=actions[ind][1:]
        with torch.no_grad():
            while not done and env.trial_timer<len(mkactionep):
                action=agent(env.decision_info)
                vw.append(action)
                _,_,done,_=env.step(mkactionep[int(env.trial_timer)])
        vw=torch.stack(vw)[:,0,:]

        plt.plot(vw,c='orange',alpha=0.5)
        plt.plot(actions[ind],c='tab:blue',alpha=0.5)
    plt.plot(vw,c='orange',alpha=1,label='agent')
    plt.plot(actions[ind],c='tab:blue',alpha=1,label='monkey')
    plt.xlabel('time [dt]')
    plt.ylabel('control v and w')
    plt.legend()




    # 3.2 overhead view, similar trials, stopping positions distribution similar

    # 3.3 overhead view, skipped trials boundary similar

    # 3.4 control curves, similar trials, similar randomness

    # 3.5 group similar trials, uncertainty grow along path, vs another theta

    # 4. validation in other (perturbation) trials
    # 4.1 overhead path of agent vs monkey in a perturbation trial, path similar


    #---------------------------------------------------------------
    # value analysis, bar plot, avg trial, skip trial

    critic=agent_.critic.qf0.cpu()
    skipped_task=[0.2,0.9]
    avg_task=[0.,0.7]
    atgoal_task=[0,0]
    with torch.no_grad():
        base=env.decision_info.flatten().clone().detach()
        skipedge=base.clone().detach();skipedge[0:2]=torch.tensor(xy2pol(skipped_task))
        avg=base.clone().detach();avg[0:2]=torch.tensor(xy2pol(avg_task))
        atgoal=base.clone().detach();atgoal[0:2]=torch.tensor(xy2pol(atgoal_task))
        avgvalue=critic(torch.cat([avg,agent(avg).flatten()]))
        skipedgevalues=critic(torch.cat([skipedge,agent(skipedge).flatten()]))
        goalv=critic(torch.cat([atgoal,agent(atgoal).flatten()]))

    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        X=['skipped','avg','goal']
        Y=[skipedgevalues,avgvalue,goalv]
        ax.bar(X,Y)


    #  value analysis, critic heatmap
    plot_critic_polar()


    #---------------------------------------------------------------
    # intial uncertainty x y
    with initiate_plot(2, 2, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot()
        cov=theta_cov(inverse_data['Hessian'])
        theta_stds=stderr(cov)[-2:]
        x_pos = [0,1]
        data=inverse_data['theta_estimations'][-1][-2:]
        data=[d[0] for d in data]
        ax.bar(x_pos, data, yerr=theta_stds, width=0.5, color = 'tab:blue')
        ax.set_xticks(x_pos)
        ax.set_ylabel('inferred parameter')
        ax.set_xticklabels(theta_names[-2:],rotation=45,ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)





    # Heissian polots-----------------------------------------------------------
    H=compute_H_monkey(env, 
                        agent, 
                        theta, 
                        phi, 
                        H_dim=11, 
                        num_episodes=9,
                        num_samples=9,
                        monkeydata=(states, actions, tasks))           
    inverse_data['Hessian']=H
    save_inverse_data(inverse_data)

    # matter_most_eigen_direction()

    stderr(torch.inverse(H))

    p=torch.round(torch.log(torch.sign(H)*H))*torch.sign(H)

    inds=[1,3,5,8,0,2,4,7,6,9,10]
    # covariance
    with initiate_plot(5,5,300) as fig:
        ax=fig.add_subplot(1,1,1)
        cov=theta_cov(H)
        cov=torch.tensor(cov)
        im=plt.imshow(cov[inds].t()[inds].t(),cmap=plt.get_cmap('bwr'),
            vmin=-torch.max(cov),vmax=torch.max(cov))
        ax.set_title('covariance matrix', fontsize=20)
        x_pos = np.arange(len(theta_names))
        plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
        plt.xticks(x_pos, [theta_names[i] for i in inds],rotation=45,ha='right')
        add_colorbar(im)


    # correlation matrix
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



    # eigen vector
    ev, evector=torch.eig(H,eigenvectors=True)
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
        ax.set_yticks([0.1,100,1e4])
        ax.set_xlabel('eigen values, log scale')
        plt.tick_params(axis='y', which='minor')
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        plt.gca().invert_yaxis()



    # pca background

    env=ffacc_real.FireFlyPaper(arg)
    number_pixels=5
    background_data=inverse_trajectory_monkey(
                        theta_trajectory,
                        env=env,
                        agent=agent,
                        phi=phi,
                        background_data=None,
                        H=None,
                        background_contour=True,
                        number_pixels=number_pixels,
                        background_look='contour',
                        ax=None,
                        action_var=0.001,
                        loss_sample_size=20)
    inverse_data['background_data']={number_pixels:background_data}
    save_inverse_data(inverse_data)


    # testing gaussian int
    g=lambda x,var: 1/torch.sqrt(2*pi*torch.ones(1))*torch.exp(-0.5*x**2/torch.ones(1)*var)
    gi=lambda x,var: -(torch.erf(x/torch.sqrt(2*torch.ones(1))/var)-1)/2
    X=np.linspace(-5,5,100)
    Y=[g(x,1) for x in X]
    Z=[gi(x,1) for x in X]
    # Z=[torch.log(torch.ones(1)*x) for x in X]
    # z=1/g(0,1)
    # Z=[z*y for y in Y]
    plt.figure(figsize=(6, 6), dpi=80)
    plt.plot(X,Y)
    plt.plot(X,Z)
    Z[49]


    # background of eigen axis
    env=ffacc_real.FireFlyPaper(arg)
    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        sample_states, sample_actions, sample_tasks=sample_batch(states=states,actions=actions,tasks=tasks,batch_size=5)
        bk=plot_background_ev(
        ax, 
        evector,
        loss_function=monkey_loss_wrapped(env=env, 
            agent=agent, 
            phi=phi, 
            num_episodes=5,
            states=sample_states,
            actions=sample_actions,
            tasks=sample_tasks,), 
        number_pixels=5, 
        alpha=0.5,
        scale=0.2,
        background_data=None)
    plt.imshow(bk)


    #----------------------------------------------------------------
    # function to get value of a given belief
    def getvalues(
        na=11,
        nd=9,
        env=None,
        agent=None,
        belief=None,
        theta=None,
        phi=None,
        iti=5,
        task=None,
        **kwargs
        ):
        # samples tasks
        distances,angles=sample_all_tasks(na=na,nd=nd)
        values=distances*0
        for ai in range(values.shape[0]):
            for di in range(values.shape[1]):
                # collect rollout
                input={
                'agent':agent,
                'theta':phi,
                'phi':phi,
                'env': env,
                'task':torch.tensor(pol2xy(angles[ai,di],distances[ai,di],)).view(1,-1).float(),
                'num_trials':1,
                'mkdata':{},                      
                'use_mk_data':False
                }
                with suppress():
                    res=trial_data(input)
                # calculate anticipated reward probobility
                finalcov=res['agent_covs'][0][-1][:2,:2]
                # finalmu=res['agent_beliefs'][0][-1][:2]
                # goal=torch.tensor(res['task'][0])
                # delta=finalmu-goal
                delta=mu_zero
                R = torch.eye(2)*env.goal_r_hat**2 
                S = R+finalcov
                alpha = -0.5 * delta @ S.inverse() @ delta.t()
                rewardprob = torch.exp(alpha) /2 / pi /torch.sqrt(S.det())
                # normalization
                mu_zero = torch.zeros(1,2)
                alpha_zero = -0.5 * mu_zero @ R.inverse() @ mu_zero.t()
                reward_zero = torch.exp(alpha_zero) /2 / pi /torch.sqrt(R.det())
                # rewardprob = rewardprob/reward_zero
                # calculate value
                # env.trial_timer if env.rewarded() else 60
                value=rewardprob*env.reward/(env.trial_timer+iti)
                values[ai,di]=value #if env.rewarded() else None
                plt.imshow(values)
        return values


    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        theta,rad = np.meshgrid(np.linspace(-0.75,0.75,nd), np.linspace(0.2,1,na)) 
        ax = fig.add_subplot(111,polar='True')
        ax.pcolormesh(theta, rad, values) #X,Y & data2D must all be same dimensions
        ax.set_theta_offset(pi/2)
        ax.grid(False)
        ax.set_thetamin(-43)
        ax.set_thetamax(43)




    norm = np.linalg.norm(values)

    normal_array = values/norm

    np.max(normal_array)


    # density heatmap testing-------------------------------
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy.ndimage.filters import gaussian_filter
    def myplot(x, y, s, bins=1000):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        heatmap = gaussian_filter(heatmap, sigma=s)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        return heatmap.T, extent
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    fig, ax = plt.subplots()
    # img, extent = myplot(x, y, 32)
    # ax.plot(x, y, 'k.', markersize=5)
    # ax.set_title("Scatter plot")
    img, extent = myplot(skipx, skipy, 22)
    # ax.imshow(img, extent=extent, origin='lower', cmap=cm.Greys)
    ax.contourf(img, extent=extent, origin='lower', cmap=cm.Greys)
    ax.set_title("Smoothing with  $\sigma$ = %d" % 32)
    plt.show()




    from scipy.stats import gaussian_kde as kde
    from matplotlib.colors import Normalize
    def makeColours( vals ):
        colours = np.zeros( (len(vals),3) )
        norm = Normalize( vmin=vals.min(), vmax=vals.max() )
        #Can put any colormap you like here.
        colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]
        return colours


    densObj = kde(np.array([skipx,skipy]))
    colours = makeColours( densObj.evaluate( np.array([skipx,skipy]) ) )
    len(colours)
    samples = np.random.multivariate_normal([0,0],env.P[:2,:2],200).T
    samples.shape
    np.array([skipx,skipy])
    plt.scatter( skipx, skipy, color=colours )




    overhead_skip_density(res1)



    # skipping analysis
    '''
        we have a value map and skipping data.
        we want to fit the data to a value curve.
        define,
            v(a,r)=value
            t=threshold of skip
        Loss=L(t(v(a,r)), true skip or not), minimize loss to solve for t.
    '''
    # first, give xy, return value.
    critic=agent_.critic.qf0.cpu()
    nbin=100
    dmin=0.2
    dmax=0.8
    maxangle=0.75
    with torch.no_grad():
        env.reset(phi=phi,theta=theta)
        base=env.decision_info.flatten().clone().detach()
        start=base.clone().detach();start[0]=dmin; start[1]=-maxangle
        xend=start.clone().detach();yend=start.clone().detach()
        xend[0]=dmax;yend[1]=maxangle
        xdelta=(xend-start)/nbin
        ydelta=(yend-start)/nbin
        values=[]
        for xi in range(nbin):
            yvalues=[]
            for yi in range(nbin):
                thestate=start+xi*xdelta+yi*ydelta
                value=critic(torch.cat([thestate,agent(thestate).flatten()]))
                yvalues.append(value.item())
            values.append(yvalues)
        values=np.asarray(values)
        values=normalizematrix(values)
        # values is a distance by angle matrix

    tasksgridd=np.linspace(dmin,dmax,nbin)
    tasksgrida=np.linspace(-maxangle,maxangle,nbin)





    newtasks=[]
    for task in tasks:
        newtasks.append(task[0])
    newtasks=np.array(newtasks)

    plt.scatter(newtasks[:,1],newtasks[:,0])




    taskvalues=[]
    taskskips=[]
    taskds=[]
    for task,state in zip(newtasks,states):
        d,a=xy2pol(task.tolist(),rotation=False)
        di=roughbisec(tasksgridd,d)
        ai=roughbisec(tasksgrida,a)
        taskvalues.append(values[di,ai])
        taskskips.append(is_skip(task,state,dthreshold=0.4))
        taskds.append(d)

    taskvalues=np.array(taskvalues).reshape(-1, 1)
    taskskips=[int(a) for a in taskskips]
    taskds=np.array(taskds).reshape(-1, 1)

    skipvalues=[]
    nonskipvalues=[]
    for v,s in zip(taskvalues,taskskips):
        if s:
            skipvalues.append(v)
        else:
            nonskipvalues.append(v)
    skipvalues=np.array(skipvalues)
    nonskipvalues=np.array(nonskipvalues)

    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.hist(nonskipvalues,alpha=0.5,bins=20,label='non-skipped')
        ax.hist(skipvalues,alpha=0.5,bins=20,label='skipped')
        ax.set_xlabel('value')
        ax.set_ylabel('number of trials')
        ax.legend()

    skipds=[]
    nonskipds=[]
    for v,s in zip(taskds,taskskips):
        if s:
            skipds.append(v)
        else:
            nonskipds.append(v)
    skipds=np.array(skipds)
    nonskipds=np.array(nonskipds)

    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.hist(nonskipds,alpha=0.5,bins=20,label='non-skipped')
        ax.hist(skipds,alpha=0.5,bins=20,label='skipped')
        ax.set_xlabel('distance')
        ax.set_ylabel('number of trials')
        ax.legend()

    from sklearn.linear_model import LogisticRegression
    logisticRegr = LogisticRegression()
    logisticRegr.fit(taskds, taskskips)

    # predictedskip=logisticRegr.predict(newtasks)
    # count=0
    # for ypred,y in zip(predictedskip,taskskips):
    #     if not (ypred^y):
    #         count+=1
    # print(count/len(taskskips))

    from scipy.special import expit
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.scatter(taskds, taskskips, color="black", zorder=20)
    X_test = np.linspace(-5, 10, 300)
    loss = expit(taskds * logisticRegr.coef_ + logisticRegr.intercept_).ravel()
    plt.plot(taskds, loss, color="red", linewidth=3)


    xmin, xmax = -5, 5
    n_samples = 100
    np.random.seed(0)
    X = np.random.normal(size=n_samples)
    y = (X > 0).astype(float)
    X[X > 0] *= 4
    X += 0.3 * np.random.normal(size=n_samples)

    X = X[:, np.newaxis]
    X.shape
    y.shape
    # Fit the classifier
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(X, y)

    # and plot the result
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.scatter(X.ravel(), y, color="black", zorder=20)
    X_test = np.linspace(-5, 10, 300)

    loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
    plt.plot(X_test, loss, color="red", linewidth=3)

    plt.ylabel("y")
    plt.xlabel("X")
    plt.xticks(range(-5, 10))
    plt.yticks([0, 0.5, 1])
    plt.ylim(-0.25, 1.25)
    plt.xlim(-4, 10)
    plt.legend(
        ("Logistic Regression Model", "Linear Regression Model"),
        loc="lower right",
        fontsize="small",
    )
    plt.tight_layout()
    plt.show()




    newtaskspolar=[]
    for task in tasks:
        newtaskspolar.append(xy2pol(task[0],rotation=False))
    newtaskspolar=np.array(newtaskspolar)
    plt.scatter(newtaskspolar[:,1],newtaskspolar[:,0])


    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        axrange=[-maxangle, maxangle, dmin , dmax]
        c=ax.imshow(values, origin='lower',extent=axrange, aspect='auto')
        add_colorbar(c)
        ax.set_ylabel('distance')
        ax.set_xlabel('angle')


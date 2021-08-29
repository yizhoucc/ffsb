import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import torch
import warnings
import random
from contextlib import contextmanager
from astropy.convolution import convolve
from numpy import pi
from numba import njit



'''
forward plots:
trained agents can do the test, and theta alters the behavior.

'''

# 1. overhead view and v and w control of agents with diff theta in a row


# function
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
        agent_dev_costs.append(torch.stack(env.trial_costs))
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
    goalcircle = plt.Circle((etask[0][0],etask[0][1]), 0.13, color='y', alpha=0.5)
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
  # dev and mag costs
    ax2 = fig.add_subplot(6,nplots,n+nplots*2+1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('costs')
    ax2.set_title('costs')
    for i in range(len(data['devcosts'])):
      ax2.plot(data['devcosts'][0], color='green',alpha=0.7)
      # ax2.plot(data['magcosts'][0], color='violet',alpha=0.7)

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
    goalcircle = plt.Circle((etask[0][0],etask[0][1]), 0.13, color='y', alpha=0.5)
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

    # dev and mag costs
    ax2 = fig.add_subplot(6,nplots,n+nplots*5+1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('costs')
    ax2.set_title('costs')
    for i in range(len(data['devcosts'])):
      ax2.plot(data['devcosts'][0], color='green',alpha=0.7)
      # ax2.plot(data['magcosts'][0], color='violet',alpha=0.7)

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
theta_init=torch.tensor([[0.4000],
        [1.57],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.13],
        [0.1],
        [0.0],
        [0.1],
        [0.1],
])
theta_final=torch.tensor([[0.4000],
        [1.57],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.13],
        [0.1],
        [0.3],
        [0.9],
        [0.1],
])
diagnose_plot_theta(agent, env, phi, theta_init, theta_final,5)






'''
inverse plots:
convergence, our stopping point,
reconstruction, we can reconstruct monkey beliefs from inverse result
explain, show that the model's action makes sense
prediction, we can predict monkey's behavior given their internal model in perturbations
'''

# 1. showing converge
# 1.1 theta uncertainty from heissian shows convergence
# 1.2 log likelihood surface in pc space, with our path and final uncertainty




#---------------------------------------------------------------------
# 1.3 inversed theta with error bar

# duymmy: create dataset
inversed_theta = [0.01,0.36,0.14,0.30,0.75,0.14,0.06,0.01,0.09,0.07]
theta_range=[0.9 for x in inversed_theta]
theta_names = ( 'pro gain v',
                'pro gain w',
                'pro noise v',
                'pro noise w',
                'obs noise v',
                'obs noise w',
                'tau',
                'goal radius',
                'mag cost',
                'deriv cost',      
                )
x_pos = np.arange(len(theta_names))
# load data
 
fig, ax = plt.subplots()
# Create bars and choose color
ax.bar(x_pos, inversed_theta, color = 'blue')
 
# title and axis names
# plt.title(' title')
ax.set_xlabel('theta parameters')
ax.set_ylabel('uncertainty std')
plt.xticks(x_pos, theta_names,rotation=45)

# formatting
# plt.subplots_adjust(bottom=0.15)
ax.set_ylim(0,1.)
fig.show()
#---------------------------------------------------------------------



#---------------------------------------------------------------------
# 1.4 differnt inverse rans and final thetas in pc space, each with uncertainty
#---------------------------------------------------------------------




#---------------------------------------------------------------------
# 2. reconstruction belief from inferred params
#---------------------------------------------------------------------



#---------------------------------------------------------------------
# 2.1 overhead view, monkey's belief and state in one trial

# load data

# create plot
fig, ax = plt.subplots()

 
# title and axis names
# plt.title(' title')
ax.set_xlabel('theta parameters')
ax.set_ylabel('uncertainty std')
plt.xticks(x_pos, theta_names,rotation=45)

# formatting
# plt.subplots_adjust(bottom=0.15)
ax.set_ylim(0,1.)
fig.show()

num_trials
agent_actions=[]
agent_beliefs=[]
agent_covs=[]
agent_states=[]
agent_dev_costs=[]
agent_mag_costs=[]
with torch.no_grad():
    for trial_i in range(num_trials):
        env.reset(phi=phi,theta=theta,goal_position=etask[0],initv=initv)
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
        agent_dev_costs.append(torch.stack(env.trial_dev_costs))
        agent_mag_costs.append(torch.stack(env.trial_mag_costs))
        agent_actions.append(torch.stack(epactions))
        agent_beliefs.append(torch.stack(epbliefs))
        agent_covs.append(epbcov)
        agent_states.append(torch.stack(epstates))
        estate=torch.stack(agent_states)[0,:,:,0].t()
delta=(theta_final-theta_init)/(nplots-1)
fig = plt.figure(figsize=[16, 20])
# curved trails
etask=[[0.7,-0.3]]
for n in range(nplots):
    ax1 = fig.add_subplot(6,nplots,n+1)
    theta=(n-1)*delta+theta_init
    ax1.set_xlabel('world x, cm')
    ax1.set_ylabel('world y, cm')
    # ax1.set_title('state plot')
    data=sample_trials(agent, env, theta, phi, etask, num_trials=1, initv=0.)
    ax1.plot(estate[0,:],estate[1,:], color='r',alpha=0.5)
    goalcircle = plt.Circle((etask[0][0],etask[0][1]), 0.13, color='y', alpha=0.5)
    ax1.add_patch(goalcircle)
    ax1.set_xlim([-0.1,1.1])
    ax1.set_ylim([-0.6,0.6])
    for t in range(len(agent_beliefs[0][:,:,0])):
        cov=agent_covs[0][t][:2,:2]
        pos=  [agent_beliefs[0][:,:,0][t,0],
                agent_beliefs[0][:,:,0][t,1]]
        plot_cov_ellipse(cov, pos, nstd=3, color=None, ax=ax1,alpha=0.2)


#---------------------------------------------------------------------



#---------------------------------------------------------------------
# 3. monkey and inferred agent act similar
# 3.1 overhead view, simliar trials, path similar
# 3.2 overhead view, similar trials, stopping positions distribution similar
# 3.3 overhead view, skipped trials boundary similar
# 3.4 control curves, similar trials, similar randomness

# 3.5 group similar trials, uncertainty grow along path, vs another theta


# 4. validation in other (perturbation) trials
# 4.1 overhead path of agent vs monkey in a perturbation trial, path similar
# 4.2 





################################
######### ultils ############
################################
#---------------------------------------------------------------------
# make the input hashmap
def input_formatter(**kwargs):
    result={}
    for key, value in kwargs.items():
        # print("{0} = {1}".format(key, value))
        result[key]=value
    return result


################################
######### work flow ############
################################
#---------------------------------------------------------------------
# load and process monkey's data
with open("C:/Users/24455/Desktop/viktor_normal_trajectory.pkl",'rb') as f:
    df = pickle.load(f)
states, actions, tasks=monkey_trajectory(df,new_dt=0.1, goal_radius=65,factor=0.002)

# example data for testing:
monkeytask=[x[0] for x in tasks[:2]]
monkeystate=states[:2]
monkeyaction=actions[:2]
num_trials=5
phi=phi
theta=theta



#---------------------------------------------------------------------
# collect the agent trial data, give task (optional, mk data)
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
theta=torch.tensor([[0.4000],
        [1.57],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.13],
        [0.1],
        [0.0],
        [0.1],
        [0.1],
])

agent_=TD3_torch.TD3.load('trained_agent/td3_150000_1_29_11_12.zip')
agent=agent_.actor.mu.cpu()
input={
    'agent':agent,
    'theta':theta,
    'phi':phi,
    'env': env,
    'num_trials':10,
    'task':[0.7,-0.2],
    'mkdata':{},
    'use_mk_data':False,
}

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
    else:
        if 'task' not in input:
            warnings.warn('random targets')
    if 'task' in input['mkdata']:
        task=input['mkdata']['task']
    if 'trial_index' in input['mkdata']:
        result['mk_trial_index']=input['mkdata']['trial_index']
        result['mk_actions']=input['mkdata']['actions']
        result['mk_states']=input['mkdata']['states']
    with torch.no_grad():
        for trial_i in range(input['num_trials']):
            if 'task' not in input:
                distance=torch.ones(1).uniform_(0.3,1)
                angle=torch.ones(1).uniform_(-pi/5,pi/5)
                task=[(torch.cos(angle)*distance).item(),(torch.sin(angle)*distance).item()]
                result['task'].append(task)
            else:
                task=input['task']
                result['task'].append(input['task'])
            if input['use_mk_data']:
                env.reset(phi=input['phi'],theta=input['theta'],goal_position=task,vctrl=input['mkdata']['actions'][0], wctrl=input['mkdata']['actions'][1])
            else:
                env.reset(phi=input['phi'],theta=input['theta'],goal_position=task)
            # prepare trial var
            epbliefs=[]
            epbcov=[]
            epactions=[]
            epstates=[]
            if input['use_mk_data']: # using mk data
                t=0
                while t<len(input['mkdata']['actions']):
                    action = agent(env.decision_info)[0]
                    if  t+1<(input['mkdata']['states'].shape[1]):
                        env.step(torch.tensor(input['mkdata']['actions'][t]).reshape(1,-1),task_param=theta,next_state=input['mkdata']['states'][:,t+1].view(-1,1)) 
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
                while not done:
                    action = agent(env.decision_info)[0]
                    _,_,done,_=env.step(torch.tensor(action).reshape(1,-1)) 
                    epactions.append(action)
                    epbliefs.append(env.b)
                    epbcov.append(env.P)
                    epstates.append(env.s)
                result['agent_actions'].append(torch.stack(epactions))
                result['agent_beliefs'].append(torch.stack(epbliefs)[:,:,0])
                result['agent_covs'].append(epbcov)
                result['agent_states'].append(torch.stack(epstates)[:,:,0])
                # astate=torch.stack(agent_states)[0,:,:].t() # used when agent on its own


    return result

res=trial_data(input)
plotctrl(res)
plotoverhead(res)


'''
inverse converge data
2.  inputL(
        theta trajectory(s),
        phi,

        param;{
            background, bool,
            uncertainty ellipse std,
            }   
            )
    
    log:(
        heissian,
        parameter uncertainties,
        background pixel array,
        uncertainty ellipse object(s),
        pca convert function,
        theta trajectory(s) in pc space,
        phi in pc space,
        )

'''

#---------------------------------------------------------------------

'''
ploting function

1. overhead view, a thin line connecting the end to target.
2. control curve
3. pc space theta trajectory approaching
4. 

'''

def plotctrl(input):
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        # ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.setxlabel('time')
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        for trial_i in range(input['num_trials']):
            ax.plot(input['agent_actions'][trial_i][:,0],color='teal',alpha=1/input['num_trials']**.3)
            ax.plot(input['agent_actions'][trial_i][:,1],color='orange',alpha=1/input['num_trials']**.3)
plotctrl(res)

#---------------------------------------------------------------------

def plotoverhead(input):
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
            ax.plot(input['agent_states'][trial_i][:,1]*400,input['agent_states'][trial_i][:,0]*400, 
                c='gray', lw=0.1, ls='-')
            ax.scatter(input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
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
plotoverhead(res)

#---------------------------------------------------------------------
# ruiyi ultils


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


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
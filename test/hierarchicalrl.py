


'''
top level instructor of skipping.

    input, current belief
    output, skip or not

training:

the problem to solve:
given a state, if to skip
given a state, what is the value

first, we get values of a goal.

approach 1, 
use agent.critic, get values,
appoach 2, 
collect rollouts, calculate values by expected reward / (expected T+ITI)
where expected T is a fixed value, we can get it by running the policy.
expected reward is a function of uncertainty, and we can also get it from policy.
here we will use appraoch 2. 

next, we use this value to decide skip or not.
we can normalize the values into 0,1 range, 
thus we can represent the skipping threshold by a parameter value.
'''



import warnings

from torch._C import Value
warnings.filterwarnings('ignore')
from copy import copy
import time
import random
from stable_baselines3 import TD3
seed=0
random.seed(seed)
import torch
torch.manual_seed(seed)
import numpy as np
np.random.seed(int(seed))
from numpy import pi
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from FireflyEnv import ffacc_real
import TD3_torch
from monkey_functions import *
from Config import Config
arg = Config()

arg.presist_phi=False
arg.agent_knows_phi=False
arg.goal_distance_range=[0.4,1]
arg.gains_range =[0.05,1.5,pi/4,pi/1]
arg.goal_radius_range=[0.05,0.3]
arg.std_range = [0.08,0.3,pi/80,pi/80*5]
arg.mag_action_cost_range= [0.0001,0.001]
arg.dev_action_cost_range= [0.0001,0.005]
arg.dev_v_cost_range= [0.1,0.5]
arg.dev_w_cost_range= [0.1,0.5]
arg.TERMINAL_VEL = 0.1
arg.DELTA_T=0.1
arg.EPISODE_LEN=100
arg.agent_knows_phi=False
arg.sample=100
arg.batch = 70
arg.NUM_IT = 1 
arg.NUM_thetas = 1
arg.ADAM_LR = 0.0002
arg.LR_STEP = 20
arg.LR_STOP = 0.5
arg.lr_gamma = 0.95
arg.PI_STD=1
arg.presist_phi=False
arg.cost_scale=1


# loading enviorment
env=ffacc_real.FireFlyReady(arg)

# load agent
agent_=TD3_torch.TD3.load('trained_agent/sk_1.zip')
agent=agent_.actor.mu.cpu()



# run trial
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
        result['task']=[input['mkdata']['task'][i][0] for i in input['mkdata']['trial_index']]
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
                    if  t+1<result['mk_states'][trial_i].shape[1]:
                        env.step(torch.tensor(result['mk_actions'][trial_i][t]).reshape(1,-1),
                        next_state=result['mk_states'][trial_i][:,t+1].view(-1,1)) 
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
    return result


# function to get value of a given belief
def getvalue(
    env=None,
    agent=None,
    belief=None,
    theta=None,
    phi=None,
    iti=5,
    task=None,
    **kwargs
    ):

    # collect rollout
    input={
    'agent':agent,
    'theta':theta,
    'phi':phi,
    'env': env,
    'task':task,
    'num_trials':1,
    'mkdata':{
                    },                      
    'use_mk_data':False
    }

    res=trial_data(input)

    # calculate anticipated reward probobility
    finalcov=res['agent_covs'][0][-1][:2,:2]
    finalmu=res['agent_beliefs'][0][-1][:2]
    goal=torch.tensor(res['task'][0])
    delta=finalmu-goal
    R = torch.eye(2)*env.goal_r_hat**2 
    S = R+finalcov
    alpha = -0.5 * finalmu @ S.inverse() @ finalmu.t()
    rewardprob = torch.exp(alpha) /2 / pi /torch.sqrt(S.det())
    # normalization
    mu_zero = torch.zeros(1,2)
    alpha_zero = -0.5 * mu_zero @ R.inverse() @ mu_zero.t()
    reward_zero = torch.exp(alpha_zero) /2 / pi /torch.sqrt(R.det())
    rewardprob = rewardprob/reward_zero

    # calculate value
    value=rewardprob*env.reward/(env.trial_timer+iti)

    return value






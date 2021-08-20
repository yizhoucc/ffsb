import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import torch
'''
forward plots:
trained agents can do the test, and theta alters the behavior.

'''

# 1. overhead view and v and w control of agents with diff theta in a row


# function
def diagnose_plot_theta(agent, env, phi, theta_init, theta_final,nplots,):
    def sample_trials(agent, env, theta, phi, etask, num_trials=5, initv=0.,initw=0.):
        # print(theta)
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
            plot_cov_ellipse(cov, pos, nstd=3, color=None, ax=ax1,alpha=0.2)

    # v and w
        ax2 = fig.add_subplot(6,nplots,n+nplots+1)
        ax2.set_xlabel('time t')
        ax2.set_ylabel('controls')
        # ax2.set_title('w control')
        agent_actions=data['agent_actions']
        for i in range(len(agent_actions)):
            ax2.plot(agent_actions[i],alpha=0.7)
            ax2.set_ylim([-1.1,1.1])
    blue = mpatches.Patch(color='teal', label='forward control v')
    orange = mpatches.Patch(color='orange', label='angular control w')
    ax2.legend(handles=[blue,orange],loc='lower right' )

phi=pac['phi']
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
        [0.1],
])
diagnose_plot_theta(agent, env, phi, theta_init, theta_final,4)






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


#---------------------------------------------------------------------
# make the input hashmap
def make_input(**kwargs):
    result={}
    for key, value in kwargs.items():
        # print("{0} = {1}".format(key, value))
        result[key]=value
    return result

#---------------------------------------------------------------------
'''
generate data


trial data
1.  input:(
        agent, 
        env, 
        number of sampled trials, 
        arg:{
            use agent action,
            use agent position,
            if use monkey similar trials,
            number of similar trials,
            }
        )

    log (
        monkey trial index,
        monkey control,
        monkey position (s),

        agent control,
        agent position (s),
        belief mu,
        cov,
        theta,
        )
'''


#---------------------------------------------------------------------
# load and process monkey's data



# example data for testing:
monkeytask=[x[0] for x in tasks[:2]]
monkeystate=states[:2]
monkeyaction=actions[:2]
num_trials=5
phi=phi
theta=theta


#---------------------------------------------------------------------
def trial_data_gen(input):

    '''
    input:
        monkey task,
        monkey state,
        monkey action,
    '''
    result={
    agent_actions:[],
    agent_beliefs:[],
    agent_covs:[],
    agent_states:[],
    }
    with torch.no_grad():
        for trial_i in range(input['num_trials']):
            env.reset(phi=phi,theta=theta,goal_position=input['mtask'],initv=input['maction'][0], initw=input['maction'][1])
            epbliefs=[]
            epbcov=[]
            epactions=[]
            epstates=[]
            t=0
            while t<len(eaction):
                action = agent(env.decision_info)[0]
                if estate is None and guided:
                # if t==1: print('no state')
                _,done=env(torch.tensor(eaction[t]).reshape(1,-1),task_param=theta) 
                elif estate is None and not guided:
                # if t==1: print('not guided')
                env.step(torch.tensor(action).reshape(1,-1)) 
                elif estate is not None and guided:
                if  t+1<(estate.shape[1]):
                    # if t==1: 
                    # print('at mk situation',estate[:,t+1].view(-1,1))
                    _,done=env(torch.tensor(eaction[t]).reshape(1,-1),task_param=theta,state=estate[:,t+1].view(-1,1)) 
                epactions.append(action)
                epbliefs.append(env.b)
                # print(env.b, action)
                epbcov.append(env.P)
                epstates.append(env.s)
                t=t+1
            agent_actions.append(torch.stack(epactions))
            agent_beliefs.append(torch.stack(epbliefs))
            agent_covs.append(epbcov)
            agent_states.append(torch.stack(epstates)[:,:,0])
            astate=torch.stack(agent_states)[0,:,:].t() #used when agent on its own
            if estate is None:
            estate=astate     
            return_dict={
            'agent_actions':agent_actions,
            'agent_beliefs':agent_beliefs,
            'agent_covs':agent_covs,
            'estate':estate,
            'astate':agent_states,
            'eaction':eaction,
            'etask':etask,
            'theta':theta,
            'index':index,
            }
    return return_dict




    return result





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

1. overhead view
2. control curve
3. pc space theta trajectory approaching
4. 

'''


#---------------------------------------------------------------------
#---------------------------------------------------------------------
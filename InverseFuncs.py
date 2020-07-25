
# collections of functions for inverse control

import torch
from torch import nn
import numpy as np
from numpy import pi

# ----- single inverse dependencies.
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import grad
from InverseFuncs import *
from collections import deque
import torch
import numpy as np
import time
from torch import nn
from collections import deque
import tqdm


def reset_theta_log(gains_range, std_range, goal_radius_range, Pro_Noise = None, Obs_Noise = None):
    '''
    generate a random theta within arg range.
    '''
    pro_gains = torch.zeros(2)
    obs_gains = torch.zeros(2)

    pro_gains[0] = torch.zeros(1).uniform_(np.exp(gains_range[0]), np.exp(gains_range[1]))  # [proc_gain_vel]
    pro_gains[1] = torch.zeros(1).uniform_(np.exp(gains_range[2]), np.exp(gains_range[3]))  # [proc_gain_ang]
    obs_gains[0] = torch.zeros(1).uniform_(np.exp(gains_range[0]), np.exp(gains_range[1]))  # [obs_gain_vel]
    obs_gains[1] = torch.zeros(1).uniform_(np.exp(gains_range[2]), np.exp(gains_range[3]))  # [obs_gain_ang]
    goal_radius = torch.zeros(1).uniform_(np.exp(goal_radius_range[0]), np.exp(goal_radius_range[1]))

    if Pro_Noise is None:
       pro_noise_stds = torch.zeros(2)
       pro_noise_stds[0] = torch.zeros(1).uniform_(np.exp(std_range[0]), np.exp(std_range[1]))  # [proc_vel_noise]
       pro_noise_stds[1] = torch.zeros(1).uniform_(np.exp(std_range[2]), np.exp(std_range[3]))  # [proc_ang_noise]
    else:
        pro_noise_stds = Pro_Noise


    if Obs_Noise is None:
        obs_noise_stds = torch.zeros(2)
        obs_noise_stds[0] = torch.zeros(1).uniform_(np.exp(std_range[0]), np.exp(std_range[1]))  # [obs_vel_noise]
        obs_noise_stds[1] = torch.zeros(1).uniform_(np.exp(std_range[2]), np.exp(std_range[3]))  # [obs_ang_noise]
    else:
        obs_noise_stds = Obs_Noise

    theta = torch.cat([pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius])
    theta=torch.log(theta)
    return theta


def reset_theta_sig(gains_range= None, std_range= None, goal_radius_range= None, Pro_Noise = None, Obs_Noise = None):
    '''
    generate a random theta within arg range.
    '''
    theta=[]
    for i in range(9):
        theta.append(torch.zeros(1).uniform_(inverse_sigmoid(torch.Tensor([0.001])).item(),inverse_sigmoid(torch.Tensor([0.999])).item()))  # [proc_gain_vel]
    theta=torch.cat(theta)
    return theta


def reset_theta(gains_range, std_range, goal_radius_range, Pro_Noise = None, Obs_Noise = None):
    '''
    generate a random theta within arg range.
    '''
    pro_gains = torch.zeros(2)
    obs_gains = torch.zeros(2)

    pro_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  # [proc_gain_vel]
    pro_gains[1] = torch.zeros(1).uniform_(gains_range[2], gains_range[3])  # [proc_gain_ang]
    obs_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  # [obs_gain_vel]
    obs_gains[1] = torch.zeros(1).uniform_(gains_range[2], gains_range[3])  # [obs_gain_ang]
    goal_radius = torch.zeros(1).uniform_(goal_radius_range[0], goal_radius_range[1])

    if Pro_Noise is None:
       pro_noise_stds = torch.zeros(2)
       pro_noise_stds[0] = torch.zeros(1).uniform_(std_range[0], std_range[1])  # [proc_vel_noise]
       pro_noise_stds[1] = torch.zeros(1).uniform_(std_range[2], std_range[3])  # [proc_ang_noise]
    else:
        pro_noise_stds = Pro_Noise


    if Obs_Noise is None:
        obs_noise_stds = torch.zeros(2)
        obs_noise_stds[0] = torch.zeros(1).uniform_(std_range[0], std_range[1])  # [obs_vel_noise]
        obs_noise_stds[1] = torch.zeros(1).uniform_(std_range[2], std_range[3])  # [obs_ang_noise]
    else:
        obs_noise_stds = Obs_Noise

    theta = torch.cat([pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius])
    return theta


def theta_range_sigmoid(theta):
    theta.data.clamp_(inverse_sigmoid(torch.Tensor([0.001])).item(),inverse_sigmoid(torch.Tensor([0.999])).item())
    return theta
    

def theta_range(theta, gains_range, std_range, goal_radius_range, Pro_Noise = None, Obs_Noise = None):

    '''
    clamp the given theta within the range
    '''

    if type(theta)==tuple:
        print("should use tensor theta not tuple! \n this method will be decraped")
        theta[0][0].data.clamp_(gains_range[0], gains_range[1])
        theta[0][1].data.clamp_(gains_range[2], gains_range[3])  # [proc_gain_ang]

        if Pro_Noise is None:
            theta[1][0].data.clamp_(std_range[0], std_range[1])  # [proc_vel_noise]
            theta[1][1].data.clamp_(std_range[2], std_range[3])  # [proc_ang_noise]
        else:
            theta[2:4].data.copy_(Pro_Noise.data)

        theta[2][0].data.clamp_(gains_range[0], gains_range[1])  # [obs_gain_vel]
        theta[2][1].data.clamp_(gains_range[2], gains_range[3])  # [obs_gain_ang]

        if Obs_Noise is None:
            theta[3][0].data.clamp_(std_range[0], std_range[1])  # [obs_vel_noise]
            theta[3][1].data.clamp_(std_range[2], std_range[3])  # [obs_ang_noise]
        else:
            theta[6:8].data.copy_(Obs_Noise.data)

        theta[4].data.clamp_(goal_radius_range[0], goal_radius_range[1])
        
        return theta
    
    else:
            
        theta[0].data.clamp_(gains_range[0], gains_range[1])
        theta[1].data.clamp_(gains_range[2], gains_range[3])  # [proc_gain_ang]

        if Pro_Noise is None:
            theta[2].data.clamp_(std_range[0], std_range[1])  # [proc_vel_noise]
            theta[3].data.clamp_(std_range[2], std_range[3])  # [proc_ang_noise]
        else:
            theta[2:4].data.copy_(Pro_Noise.data)

        theta[4].data.clamp_(gains_range[0], gains_range[1])  # [obs_gain_vel]
        theta[5].data.clamp_(gains_range[2], gains_range[3])  # [obs_gain_ang]

        if Obs_Noise is None:
            theta[6].data.clamp_(std_range[0], std_range[1])  # [obs_vel_noise]
            theta[7].data.clamp_(std_range[2], std_range[3])  # [obs_ang_noise]
        else:
            theta[6:8].data.copy_(Obs_Noise.data)

        theta[8].data.clamp_(goal_radius_range[0], goal_radius_range[1])


        return theta


def single_inverse(true_theta, phi, arg, env, agent, states, actions, tasks, filename, num_thetas,initial_theta=None):

    # tic = time.time()
    # generate an initial estimation of theta, if not given
    if initial_theta is None:
        theta=env.reset_task_param()
    else:
        theta=initial_theta
    theta = nn.Parameter(theta)
    initial_theta = theta.data.clone()
    print('initial theta: ',initial_theta)
    # prepare the logs
    loss_log = deque(maxlen=arg.NUM_IT)
    loss_act_log = deque(maxlen=arg.NUM_IT)
    loss_obs_log = deque(maxlen=arg.NUM_IT)
    theta_log = deque(maxlen=arg.NUM_IT)
    gradient=deque(maxlen=arg.NUM_IT)
    optT = torch.optim.Adam([theta], lr=arg.ADAM_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optT, step_size=arg.LR_STEP, gamma=arg.lr_gamma) # decreasing learning rate x0.5 every 100steps
    prev_loss = 100000
    loss_diff = deque(maxlen=5)
    # for iterations, do:
        # get loss,
        # apply to estimation of theta.
    for it in range(arg.NUM_IT):
        loss = getLoss(agent, states, actions, tasks, phi, theta, env, arg)
        loss_log.append(loss.data)
        optT.zero_grad() #clears old gradients from the last step
        loss.backward(retain_graph=True) #computes the derivative of the loss w.r.t. the parameters using backpropagation
        gradient.append(theta.grad.data.clone())
        optT.step() # performing single optimize step: this changes theta
        theta_log.append(theta.data.clone())
        if it < arg.LR_STOP:
            scheduler.step()
        loss_diff.append(torch.abs(prev_loss - loss))
        prev_loss = loss.data
            #print("num_theta:{}, num:{}, loss:{}".format(n, it, np.round(loss.data.item(), 6)))
            #print("num:{},theta diff sum:{}".format(it, 1e6 * (true_theta - theta.data.clone()).sum().data))
        print("converged_theta:\n{}".format( theta.data.clone() ) )
        print("true theta:     \n{}".format(true_theta.data.clone()))
            # print('\ngrad     ', theta.grad.data.clone())
    
    
    # loss = getLoss(agent, states, actions, tasks, phi, theta, env, arg)
    # toc = time.time()

    # check heissian
    # grads = grad(loss, theta, create_graph=True)[0]
    # H = torch.zeros(9,9)
    # for i in range(9):
    #     H[i] = grad(grads[i], theta, retain_graph=True)[0]
    # I = H.inverse()
    # stderr = torch.sqrt(I.diag())
    # stderr_ii = 1/torch.sqrt(torch.abs(H.diag()))
    # result = {'true_theta': true_theta,
    #           'initial_theta': ini_theta,
    #           'theta': theta,
    #           'theta_log': theta_log,
    #           'loss_log': loss_log,
    #           'loss_act_log': loss_act_log,
    #           'loss_obs_log': loss_obs_log,
    #           'filename': filename,
    #           'num_theta': n,
    #           'converging_it': it,
    #           'duration': toc-tic,
    #           'arguments': arg,
    #           'stderr': stderr,
    #           'stderr_ii': stderr_ii,
    #           'grad':gradient
    #           }
    # return result


def trajectory(agent, phi, theta, env, NUM_EP):
    with torch.no_grad():
        #------prepare saving vars-----
        states = [] # true location
        actions = [] # action
        tasks=[] # vars that define an episode

        #---------reset the env, using theta and phi. theta not necessarily equals phi
        theta=nn.Parameter(theta)
        env.reset(phi=phi,theta=theta)

        episode = 1
        while episode <= NUM_EP:
            episode +=1 # 1 index
            states_ep = []
            actions_ep = []
            task_ep=[[env.goalx, env.goaly], env.phi]
            done=False

            # loop for one trial
            while not done:
                action = agent(env.decision_info)[0]
                actions_ep.append(action) # a at t0
                states_ep.append(env.s)     # s at t0
                _,done=env(action,task_param=theta) 
                # now s, o,b P are at  t+1
            # now we have t0-tT's states, action chose from these states.

            # end of one trial
            states.append(states_ep)
            actions.append(actions_ep)
            tasks.append(task_ep)

            # reset to a new trial, with same param but diff goalx,y
            env.reset(phi=phi,theta=theta)

    return states, actions, tasks


def trajectory_(agent, theta, env, arg, gains_range, std_range, goal_radius_range, NUM_EP):
    '''
    get NUM_EP=500 trials of (x,o,b,a|phi,theta=phi)
    '''
    pro_gains, pro_noise_stds, obs_gains, obs_noise_stds,  goal_radius = torch.split(theta.view(-1), 2)

    x_traj = [] # true location
    obs_traj =[] # observation
    a_traj = [] # action
    b_traj = []

    env.pro_gains = pro_gains
    env.pro_noise_stds = pro_noise_stds
    env.goal_radius = goal_radius
    env.obs_gains=obs_gains
    env.obs_noise_stds=obs_noise_stds

    env.reset()     # apply the true theta to env

    env.belief = env.Breshape(theta=theta)
    episode = 0
    tot_t = 0

    #while tot_t <= TOT_T:
    while episode <= NUM_EP: #defalt 500
        episode +=1
        t = torch.zeros(1)
        x_traj_ep = []
        obs_traj_ep = []
        a_traj_ep = []
        b_traj_ep = []

        while t < arg.EPISODE_LEN: # for a single FF

            action = agent(env.belief)[0] # TODO action rounded, not precise
            # action=agent.predict(env.belief_state)[0]
            a_traj_ep.append(action)
            x_traj_ep.append(env.x)
            # env(action)
            env(action.view(-1)) # so env.x env.belief updated by action and next o
            obs_traj_ep.append(env.o) # the same next obs used to update belief

            # check time limit
            TimeEnd = (t+1 == arg.EPISODE_LEN) # if the monkey can't catch the firefly in EPISODE_LEN, reset the game.
            mask = torch.tensor([1 - float(TimeEnd)]) # mask = 0: episode is over

            tot_t += 1.
            t += 1
            if env.stop:
                pass
            if env.stop or TimeEnd:  # if the monkey stops or pass the time limit, start the new firefly
                
                env.pro_gains = pro_gains
                env.pro_noise_stds = pro_noise_stds
                env.goal_radius = goal_radius
                env.obs_gains=obs_gains
                env.obs_noise_stds=obs_noise_stds

                env.reset()

                break
        x_traj.append(x_traj_ep)
        obs_traj.append(obs_traj_ep)
        a_traj.append(a_traj_ep)
        b_traj.append(b_traj_ep)
    return x_traj, obs_traj, a_traj, b_traj


def getLoss(agent, states, actions, tasks, phi, theta, env, arg):
    
    '''
    from the trajectories, compute the loss for current estimation of theta
    the theta input is the estimation of theta.
    we dont need true theta, as we are not supposed to know the true theta.
    phi, if given phi, make sure the env.phi is phi
    '''

    # initialize the loss var
    logPr = torch.zeros(1) #torch.FloatTensor([])
    # we dont really need to take out the vars
    # pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = torch.split(theta.view(-1), 2)

    # what is the num samples here? epoch likely
    for num_it in range(arg.NUM_SAMPLES): 
        # repeat for number of episodes in the trajectories
        for ep, x_traj_ep in enumerate(states): 
            # initialize the loss var
            logPr_ep = torch.zeros(1)
            x = x_traj_ep[0].view(-1)
            pos=tasks[ep][0]
            # take out the teacher trajectories for one episode
            a_traj_ep = actions[ep]  
            # reset monkey's internal model, using the task parameters, estimation of theta
            env.reset(theta=theta, phi=phi, goal_position=pos)  
            # repeat for steps in episode
            for it, next_x in enumerate(x_traj_ep[1:]): 
                action = agent(env.decision_info)[0] 
                decision_info,_=env(a_traj_ep[it],task_param=theta) 
                # env.s=next_x # offset the noise during state transition, is it necessary?
                action_loss = mse_loss(action , a_traj_ep[it] )
                # obs_loss = 5*torch.ones(2)+torch.log(np.sqrt(2* pi)*obs_noise_stds) +(next_ox_ - next_ox).view(-1) ** 2/2/(obs_noise_stds**2)
                # print(action_loss)
                logPr_ep = logPr_ep + action_loss.sum()
            logPr += logPr_ep
        # print('loss is: ',logPr)

    return logPr


def mse_loss(true,estimate):
    loss=(true - estimate ) ** 2
    return loss

def getLoss_(agent, x_traj,obs_traj, a_traj, theta, env, gains_range, std_range, PI_STD, NUM_SAMPLES):
    ''
    logPr = torch.zeros(1) #torch.FloatTensor([])
    logPr_act = torch.zeros(1)
    logPr_obs = torch.zeros(1)

    pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = torch.split(theta.view(-1), 2)

    for num_it in range(NUM_SAMPLES): # what is the numsample/particle here? default is 50

        for ep, x_traj_ep in enumerate(x_traj): # repeat for 501 episodes
            a_traj_ep = a_traj[ep]
            obs_traj_ep=obs_traj[ep]
            logPr_ep = torch.zeros(1)
            logPr_act_ep = torch.zeros(1)
            logPr_obs_ep = torch.zeros(1)

            t = torch.zeros(1)
            x = x_traj_ep[0].view(-1)

            env.pro_gains = pro_gains
            env.pro_noise_stds = pro_noise_stds
            env.goal_radius = goal_radius
            env.obs_gains=obs_gains
            env.obs_noise_stds=obs_noise_stds
            env.reset()  # reset monkey's internal model

            env.x=x # assign the same x
            env.b=x,env.P # assign same b
            env.belief = env.Breshape(b=env.b, time=t, theta=theta)

            for it, next_x in enumerate(x_traj_ep[1:]): # repeat for steps in episode
                action = agent(env.belief)[0] # simulated acton

                next_ox = env.observations_mean(next_x) # multiplied by observation gain, no noise
                # next_ox_ = env.observations(next_x)  # simulated observation (with noise)

                action_loss =5*torch.ones(2)+np.log(np.sqrt(2* pi)*PI_STD) + (action - a_traj_ep[it] ) ** 2 / 2 /(PI_STD**2)
                # obs_loss = 5*torch.ones(2)+torch.log(np.sqrt(2* pi)*obs_noise_stds) +(next_ox_ - next_ox).view(-1) ** 2/2/(obs_noise_stds**2)

                logPr_act_ep = logPr_act_ep + action_loss.sum()
                logPr_obs_ep = logPr_obs_ep #+ obs_loss.sum()
                logPr_ep = logPr_ep + logPr_act_ep #+ logPr_obs_ep

                next_b, info = env.belief_step(env.b, next_ox, a_traj_ep[it], env.box)  # no change to internal var
                env.b=next_b
                next_state = env.Breshape(b=next_b, time=t, theta=theta)
                env.belief=next_state
                t += 1



            logPr_act += logPr_act_ep
            logPr_obs += logPr_obs_ep

            logPr += logPr_ep
            #logPr = torch.cat([logPr, logPr_ep])

    return logPr, logPr_act, logPr_obs #logPr.sum()


def init_theta(phi,arg,purt=None):
    '''
    create a initial theta by adding random to phi
    '''
    rndsgn = torch.sign(torch.randn(1,len(phi))).view(-1)
    if purt is None:
        purt= torch.Tensor([0.5,0.5,0.1,0.1,0.5,0.5,0.1,0.1,0.1])
    theta = phi.data.clone()+rndsgn*purt
    theta = theta_range(theta, arg.gains_range, arg.std_range, arg.goal_radius_range)  # keep inside of trained range
    return theta


def unpack_theta(theta):
    'unpack the 1x9 tensor theta into p gain/noise, obs gain/noise, r'
    pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = torch.split(theta.view(-1), 2)
    return pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius


def norm_parameter(parameter,param_range, given_range=[0.001,0.999]):
    'normalize the paramter range to a range'
    k=(max(given_range)-min(given_range))/(max(param_range)-min(param_range))
    c=k*min(param_range)-min(given_range)
    parameter=parameter*k-c
    return parameter


def denorm_parameter(parameter,param_range, given_range=[0.001,0.999]):
    'denormalize the paramter range to a range'
    k=(max(given_range)-min(given_range))/(max(param_range)-min(param_range))
    c=k*min(param_range)-min(given_range)
    parameter=(parameter+c)/k
    return parameter


def inverse_sigmoid(parameter):
    return torch.log(parameter/(1-parameter))


# def theta_init(agent, env, arg):
#     '''
#     return (true theta, true loss, x trajectory, a trajectory)
#     '''

#     # true theta
#     true_theta = reset_theta(arg.gains_range, arg.std_range, arg.goal_radius_range)
#     #true_theta_log.append(true_theta.data.clone())
#     x_traj, _, a_traj, _ = trajectory(agent, true_theta, env, arg, arg.gains_range, arg.std_range,
#                                       arg.goal_radius_range, arg.NUM_EP)  # generate true trajectory
#     true_loss, _, _ = getLoss(agent, x_traj, a_traj, true_theta, env, arg.gains_range, arg.std_range, arg.PI_STD,
#                         arg.NUM_SAMPLES)  # this is the lower bound of loss?

#     init_result = {'true_theta_log': true_theta,
#                    'true_loss_log': true_loss,
#                    'x_traj_log': x_traj,
#                    'a_traj_log': a_traj
#                    }
#     return init_result
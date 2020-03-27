
# collections of functions for inverse control

import torch
import numpy as np
from numpy import pi

def trajectory(agent, theta, env, arg, gains_range, std_range, goal_radius_range, NUM_EP):
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

    env.reset()
    x=env.state
    #ox = agent.Bstep.observations(x)  # observation
    state=env.belief_state


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

            action = agent(state)[0]

            env( action.view(-1)) #track true next_x of monkey

            next_ox = env.observations(env.state)  # observation

            next_state, _,_,info = env.step(action)#b, next_ox, action, env.box) # belief next state, info['stop']=terminal # reward only depends on belief

            # check time limit
            TimeEnd = (t+1 == arg.EPISODE_LEN) # if the monkey can't catch the firefly in EPISODE_LEN, reset the game.
            mask = torch.tensor([1 - float(TimeEnd)]) # mask = 0: episode is over

            x_traj_ep.append(x)
            obs_traj_ep.append(next_ox)
            a_traj_ep.append(action)
            # b_traj_ep.append(b)

            x = env.state
            state = next_state
            # b = next_b
            #ox = next_ox
            tot_t += 1.
            t += 1

            if info['stop'] or TimeEnd:  # if the monkey stops or pass the time limit, start the new firefly
                
                env.pro_gains = pro_gains
                env.pro_noise_stds = pro_noise_stds
                env.goal_radius = goal_radius
                env.obs_gains=obs_gains
                env.obs_noise_stds=obs_noise_stds

                env.reset()
                x=env.state
                #ox = agent.Bstep.observations(x)  # observation
                state=env.belief_state
                break
        x_traj.append(x_traj_ep)
        obs_traj.append(obs_traj_ep)
        a_traj.append(a_traj_ep)
        b_traj.append(b_traj_ep)
    return x_traj, obs_traj, a_traj, b_traj



# MCEM based approach 
def getLoss(agent, x_traj, a_traj, theta, env, gains_range, std_range, PI_STD, NUM_SAMPLES):

    logPr = torch.zeros(1) #torch.FloatTensor([])
    logPr_act = torch.zeros(1)
    logPr_obs = torch.zeros(1)

    pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = torch.split(theta.view(-1), 2)



    for num_it in range(NUM_SAMPLES): # what is the numsample/particle here? default is 50

        for ep, x_traj_ep in enumerate(x_traj): # repeat for 501 episodes
            a_traj_ep = a_traj[ep]
            logPr_ep = torch.zeros(1)
            logPr_act_ep = torch.zeros(1)
            logPr_obs_ep = torch.zeros(1)

            t = torch.zeros(1)
            x = x_traj_ep[0].view(-1)

            env.pro_gains = pro_gains
            env.pro_noise_stds = pro_noise_stds
            env.goal_radius = goal_radius
            env.reset()  # reset monkey's internal model
            env.x=x
            state= env.belief_state
            b=x,env.P
            for it, next_x in enumerate(x_traj_ep): # repeat for steps in episode
                action = agent(state)[0] # simulated acton



                #next_ox = env.input(next_x, obs_gains) # multiplied by observation gain, no noise
                next_ox = env.observations_mean(next_x) # multiplied by observation gain, no noise
                next_ox_ = env.observations(next_x)  # simulated observation (with noise)
                # this is the only thing changes action-trajectory action, part of the action loss



                action_loss =5*torch.ones(2)+np.log(np.sqrt(2* pi)*PI_STD) + (action - a_traj_ep[it] ) ** 2 / 2 /(PI_STD**2)
                obs_loss = 5*torch.ones(2)+torch.log(np.sqrt(2* pi)*obs_noise_stds) +(next_ox_ - next_ox).view(-1) ** 2/2/(obs_noise_stds**2)

                logPr_act_ep = logPr_act_ep + action_loss.sum()
                logPr_obs_ep = logPr_obs_ep + obs_loss.sum()



                logPr_ep = logPr_ep + logPr_act_ep + logPr_obs_ep
                #logPr_ep = logPr_ep +   (obs_loss).sum()

                next_b, info = env.belief_step(b, next_ox_, a_traj_ep[it], env.box)  # action: use real data
                env.b=next_b
                next_state = env.Breshape(next_b, t, (pro_gains, pro_noise_stds, obs_gains, obs_noise_stds,
                                                              goal_radius))
                env.belief=next_state
                t += 1
                state = next_state
                b = next_b

            logPr_act += logPr_act_ep
            logPr_obs += logPr_obs_ep

            logPr += logPr_ep
            #logPr = torch.cat([logPr, logPr_ep])

    return logPr, logPr_act, logPr_obs #logPr.sum()




def reset_theta(gains_range, std_range, goal_radius_range, Pro_Noise = None, Obs_Noise = None):
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


def theta_range(theta, gains_range, std_range, goal_radius_range, Pro_Noise = None, Obs_Noise = None):

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


def theta_init(agent, env, arg):
    '''
    return (true theta, true loss, x trajectory, a trajectory)
    '''

    # true theta
    true_theta = reset_theta(arg.gains_range, arg.std_range, arg.goal_radius_range)
    #true_theta_log.append(true_theta.data.clone())
    x_traj, _, a_traj, _ = trajectory(agent, true_theta, env, arg, arg.gains_range, arg.std_range,
                                      arg.goal_radius_range, arg.NUM_EP)  # generate true trajectory
    true_loss, _, _ = getLoss(agent, x_traj, a_traj, true_theta, env, arg.gains_range, arg.std_range, arg.PI_STD,
                        arg.NUM_SAMPLES)  # this is the lower bound of loss?

    init_result = {'true_theta_log': true_theta,
                   'true_loss_log': true_loss,
                   'x_traj_log': x_traj,
                   'a_traj_log': a_traj
                   }
    return init_result

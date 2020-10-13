"""
reward.py
This file describes reward function which is the “expected reward” for the belief distribution over Gaussian reward distribution.
rew_std: standard deviation of Gaussian distribution for reward [std_x, std_y]

b(s)= 1/sqrt(2*pi*det(P)) * exp(-0.5* ((s-x)^T*P^-1*(s-x)) : 
Gaussian distribution with mean x, covariance P

r(s) = scale * exp(-0.5* s^T* R^-1 * s): 
reward gaussian distribution with mean zeros, covariance R

invS = invR +invP
R(b) = \int b(s)*r(s) ds = c *sqrt(det(S)/det(P))* exp(-0.5* mu^T*(invP - invP*S*invP)*mu)

R(b) =  \int b(s)*r(s) ds = 1/sqrt(det(2 pi (P+R)) * exp(-0.5*mu^t(R+P)^-1*mu)
"""
import torch
import numpy as np
from numpy import pi
from FireflyEnv.env_utils import is_pos_def
import torch
'''
the reward is computed for every time step.
'''

class Reward(object):
    
    def __init__(self,reward_function=None):
        pass


def mixing_reward(self, reward_source, reward_target,ratio_target):
        return reward_source*(1-ratio_target)+reward_target*ratio_target


def actual_task_reward(stop, reached_target, reward=10, belief=None, goal_radius=None):
    '''
    return discrete rewrad based on real location and stop.
    this is the actual task setting.
    '''
    if stop and reached_target:
        return reward
    else:
        return 0.


def belief_gaussian_reward(agent_stops, reached_target, b,P, goal_radius, REWARD,goalx,goaly,time=0):
    '''
    gaussian reward based on belief uncertainty overlapping with goal radius reward distribution.
    that is, at center the reward is higher. 
    '''
    def rewardFunc(rew_std, x, P, scale,goalx,goaly):
        mu = torch.Tensor([goalx,goaly])-x[:2]  # pos
        
        R = torch.eye(2) * rew_std**2 # reward function is gaussian
        P = P[:2, :2] # cov
        S = R+P
        if not is_pos_def(S):
            print('R+P is not positive definite!')
        alpha = -0.5 * mu @ S.inverse() @ mu.t()
        reward = torch.exp(alpha) /2 / np.pi /torch.sqrt(S.det())
        print(reward)
        # normalization -> to make max reward as 1
        mu_zero = torch.zeros(1,2)
        alpha_zero = -0.5 * mu_zero @ R.inverse() @ mu_zero.t()
        reward_zero = torch.exp(alpha_zero) /2 / np.pi /torch.sqrt(R.det())
        reward = reward/reward_zero

        reward = scale * reward  # adjustment for reward per timestep
        if reward > scale:
            print('reward is wrong!', reward)
            print('mu', mu)
            print('P', P)
            print('R', R)
        return reward.view(-1)

    def get_reward(b,P, goal_radus, REWARD,time,goalx,goaly):
        rew_std = goal_radus / 2  # std of reward function --> 2*std (=goal radius) = reward distribution
        reward = rewardFunc(rew_std, b.view(-1), P, REWARD,goalx,goaly)
        return reward

    reward = get_reward(b,P, goal_radius, REWARD,time,goalx,goaly)
    # if not agent_stops or not reached_target:
    #     return 0.
    return reward.item()


def reward_LQC_test(agent_stops, reached_target, b, goal_radius, REWARD,time=0, episode=0,finetuning = 0):
    x, P = b
    position=x[:2]
    r=torch.norm(position)
    reward = REWARD*(1-r)*0.9**time  # reward currently only depends on belief not action
    return reward


def actual_task_reward(agent_stops, reached_target, b,P, goal_radius, REWARD,goalx,goaly,time=0):
    if reached_target and agent_stops:
        return REWARD
    else:
        return 0.


def state_gaussian_reward(agent_stops, reached_target, b,P, goal_radius, REWARD,goalx,goaly,time=0):
    d=torch.Tensor([goalx,goaly])-b[0,:2]
    r=torch.norm(d)
    if agent_stops:
        std=goal_radius/2
        gaussian_reward=(1/std/np.sqrt(2*np.pi))* (np.exp(-0.5*(r/std)**2))
        reward=min(gaussian_reward,torch.ones(1))*REWARD
        return reward.item()
    else:
        return 0.


def return_reward_swcase(agent_stops, reached_target, b, goal_radius, REWARD,time=0, episode=0,finetuning = 0):
    reward = get_reward(b, goal_radius, REWARD,time)
    if agent_stops:
        if reached_target:
            pass  
        else:
            reward=0.5* torch.ones(1)
    else:
        reward=reward*0.1

    return reward


def return_reward_LQC_test(agent_stops, reached_target, b, goal_radius, REWARD,time=0, episode=0,finetuning = 0):
    x, P = b
    position=x[:2]
    r=torch.norm(position)
    reward = REWARD*(1-r)*0.9**time  # reward currently only depends on belief not action
    return reward


def return_reward_location(agent_stops, reached_target, b,P, goal_radius, REWARD,goalx,goaly,time=0):
    if reached_target:
        return REWARD
    else:
        return 0.


def belief_reward_bin_intergration(agent_stops, reached_target, b,P, goal_radius, REWARD,goalx,goaly,bins=80,time=0,discount=0.95):   
    
    cov=P[:2,:2]
    mu = torch.Tensor([goalx,goaly])-b[:2,0]  # pos
    # first a squre
    xrange=[mu[0]-goal_radius, mu[0]+goal_radius]
    yrange=[mu[1]-goal_radius, mu[1]+goal_radius]
    # select the circle (actual goal range)
    P=0
    xs=np.linspace(xrange[0],xrange[1],bins)
    ys=np.linspace(yrange[0],yrange[1],bins)

    for i in range(bins):
        for j in range(bins):
            if (xs[i]-mu[0])**2+(ys[j]-mu[1])**2<=goal_radius**2:                
                expectation=( (1/2/pi/np.sqrt(np.linalg.det(cov)))
                * np.exp(-1/2
                    * (np.array([xs[i],ys[j]]).reshape(1,2)@np.linalg.inv(cov)@np.array([xs[i],ys[j]]).reshape(2,1)) ))
                P=P+torch.Tensor(expectation)*4*goal_radius*goal_radius/bins/bins

    if P>= 1:
        print('error, prob is greater than or equal 1')
    reward= P*REWARD

    if time != 0:
        reward=reward*discount**time
        # print('the reward with discounting',reward)
    if type(reward)==torch.Tensor:
        # print('reward as tensor',reward)
        reward=reward.item()
        # print('reward as float',reward)
    return reward

    # def get_reward(b,P, goal_radus, REWARD,time,goalx,goaly):
    #     rew_std = goal_radus / 2  # std of reward function --> 2*std (=goal radius) = reward distribution
    #     reward = rewardFunc(rew_std, b.view(-1), P, REWARD,goalx,goaly)
    #     return reward

    # reward = get_reward(b,P, goal_radius, REWARD,time,goalx,goaly)
    # # if not agent_stops or not reached_target:
    # #     return 0.
    # return reward.item()   


def belief_reward_mc(agent_stops, reached_target, b,P, goal_radius, REWARD,goalx,goaly,nsamples=800,time=0,discount=0.95, verbose=True):
    
    cov=P[:2,:2]
    mu = torch.Tensor([goalx,goaly])-b[:2,0]  # pos
    # construct ellipse object and sample
    xs, ys = np.random.multivariate_normal(mu, cov, nsamples).T
    check=[]
    # select the dots in circle (actual goal range)
    for i in range(nsamples):
        if (xs[i])**2+(ys[i])**2<=goal_radius**2:                
            check.append(1)
        else:
            check.append(0)

    P=np.mean(check)
    if verbose:
        print('reward p ',P, ' time t', time)
    reward= P*REWARD*discount**time

    if type(reward)==torch.Tensor:
        # print('reward as tensor',reward)
        reward=reward.item()
        # print('reward as float',reward)
    return reward


def action_cost_magnitude(action):
    # sum up the squared norm action magnitude
    cost=np.sum(np.linalg.norm(action)**2)
    return cost


def action_cost_dev(action, previous_action):
    # sum up the norm squre for delta action
    cost=(np.linalg.norm(action-previous_action))**2
    return cost


def action_cost_wrapper(action, previous_action,mag_scalar, dev_scalar):
    mag_cost=action_cost_magnitude(action)
    dev_cost=action_cost_dev(action, previous_action)
    total_cost=mag_scalar*mag_cost+dev_scalar*dev_cost
    return total_cost, mag_cost, dev_cost


# def reward_wrapper_ac(agent_stops, reached_target, b,P, goal_radius, REWARD,goalx,goaly,nsamples=800,time=0,discount=0.95, actions=None, action_cost_param=[0.05,0.05]):
#     reward=belief_reward_mc(agent_stops, reached_target, b,P, goal_radius, REWARD,goalx,goaly,nsamples=nsamples,time=time,discount=discount)
#     cost=action_cost_param[0]*action_cost_magnitude(actions)+action_cost_param[1]*action_cost_dev(actions)    
#     final_reward=reward-cost
#     return final_reward
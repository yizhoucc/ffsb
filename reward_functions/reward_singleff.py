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

def actual_task_reward(agent_stops, reached_target, a,b,c, REWARD,d,e):
    if reached_target and agent_stops:
        return REWARD
    else:
        return 0.

def state_gaussian_reward(agent_stops, reached_target, b,P, goal_radius, REWARD,goalx,goaly,time=0):
    if reached_target and agent_stops:
        return REWARD
    elif agent_stops:
        std=goal_radius/3
        gaussian_reward=(1/std/np.sqrt(2*np.pi))* (np.exp(-0.5*(r/std)**2))
        reward=max(gaussian_reward,REWARD)
        return reward
    else:
        return 0.



def return_reward_swcase(agent_stops, reached_target, b, goal_radius, REWARD,time=0, episode=0,finetuning = 0):
    reward = get_reward(b, goal_radius, REWARD,time)
    if agent_stops:
        if reached_target:
            pass  
        else:
            reward=0.5* torch.ones(1)
        # receive reward if monkey stops. position does not matters
        # if finetuning == 0:
        #     reward = get_reward(b, goal_radius, REWARD,time)
        #     # if reached_target == 1:
        #     #     # print("Ep {}: Good Job!!, reward= {:0.3f}".format(episode, reward[-1]))
        #     # else:
        #     #     pass #print("reward= %.3f" % reward.view(-1)) #pass
        # # elif finetuning == 1 and reached_target == 1:
        # #     reward = REWARD * torch.ones(1)
        # #     print("Ep {}: Good Job!!, FIXED reward= {:0.3f}".format(episode, reward[-1]))
        # else:  # finetuning == 1 and reached_target == 0
        #     reward = -0 * torch.ones(1)
    else:
        reward=reward*0.1
        # reward = -0 * torch.ones(1)
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
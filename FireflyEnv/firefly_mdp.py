import gym
from gym import spaces
from numpy import pi
import numpy as np
from FireflyEnv.env_utils import *
from FireflyEnv.firefly_acc import FireflyAcc
from reward_functions import reward_singleff

def get_vm_(tau, d=2, T=7):
    vm=d/2/tau *(1/(np.log(np.cosh(T/2/tau))))
    return vm

def compute_tau_range(vm_range, d=2, T=7, tau=1.8):
    vm=get_vm_(tau, d=d, T=T)
    while vm > vm_range[0]:
        tau=tau-0.1
        vm=get_vm_(tau)
    lowertau=max(tau,0.1)
    while vm < vm_range[-1]:
        tau=tau+0.1
        vm=get_vm_(tau)
    uppertau=tau
    # take 2 place after .
    lowertau=lowertau//0.01*0.01
    uppertau=uppertau//0.01*0.01
    return [lowertau, uppertau]

class FireflyMDP(FireflyAcc): 

    def observations(self, s, task_param=None): 

        '''
        get observation of the state

        inpute:
            state 
            task param, default is theta
        output:
            observation of the state
        '''

        
        task_param=self.theta if task_param is None else task_param
        # sample some noise
        on = torch.distributions.Normal(0,torch.ones([2,1])).sample()*task_param[6:8]*0 # on is observation noise
        vel, ang_vel = torch.split(s.view(-1),1)[-2:] # 1,5 to vector and take last two

        ovel = task_param[4,0] * vel + on[0] # observe velocity
        oang_vel = task_param[5,0]* ang_vel + on[1]
        observation = torch.stack((ovel, oang_vel)) # observed x
        return observation.view(-1,1)


    def observations_mean(self, state, task_param=None): # apply external noise and internal noise, to get observation

        '''
        mean observation of the state, only gain no noise

        inpute:
            state 
            task param, default is theta
        output:
            observation of the state
        '''        
        
        task_param=self.theta if task_param is None else task_param

        vel, ang_vel = torch.split(state.view(-1),1)[-2:] # 1,5 to vector and take last two

        ovel = self.theta[4,0] * vel  
        oang_vel = self.theta[5,0]* ang_vel 
        observation = torch.stack((ovel, oang_vel)) 

        return observation.view(-1,1)



    
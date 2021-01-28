import gym
from gym import spaces

from numpy import pi
import numpy as np

from FireflyEnv.env_utils import *

from FireflyEnv.ffenv_new_cord import FireflyAgentCenter
from reward_functions import reward_singleff

# change log
# obs space is inf
# representation

# done, make max action cost lower than possible reward
# lowest possible reward, a function, now value is reward_scale*gamma**T <2.5
# using 2.5 as max cost.
# we first random a alpha, then random a beta in proper range.


class FireflyActionCost(FireflyAgentCenter): 

    def __init__(self,arg=None,kwargs=None):
        '''
        state-observation-blief-action-next state
        arg:
            gains_range, 4 element list

        step:  action -> new state-observation-blief    
        reset: init new state
        '''
        super(FireflyAgentCenter, self).__init__(arg=arg,kwargs=kwargs)
        
        # we have modified observation space
        low=-np.inf
        high=np.inf
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,31),dtype=np.float32)

        self.cost_function=reward_singleff.action_cost_wrapper
        self.cost_scale=1


    def _apply_param_range(self,gains_range=None,std_range=None,goal_radius_range=None,mag_action_cost_range=None,dev_action_cost_range=None):

        if goal_radius_range is None:
            self.goal_radius_range =     self.arg.goal_radius_range
        if gains_range is None:
            self.gains_range=            self.arg.gains_range
        if std_range is None:
            self.std_range =             self.arg.std_range
        if mag_action_cost_range is None:
            self.mag_action_cost_range =     self.arg.mag_action_cost_range
        if dev_action_cost_range is None:
            self.dev_action_cost_range =     self.arg.dev_action_cost_range

    def reset_decision_info(self):
        self.episode_time = 0   # int
        self.stop=False         # bool
        self.decision_info = self.wrap_decision_info(b=self.b, time=self.episode_time, task_param=self.theta)
        self.decision_info = row_vector(self.decision_info)
        self.previous_action=np.zeros(2)


    # save previous control to buffer
    def state_step(self,a, s, task_param=None):

        '''
        # acc control, vt+1 = a*vt + b*ut
        '''
        
        # STEP 1: use the v,w to move forward
        task_param=self.phi if task_param is None else task_param
        next_s = self.update_state(s, task_param=task_param)

        # STEP 2: use the action to assign v and w
        px, py, heading, vel, ang_vel = torch.split(next_s.view(-1), 1)
        a_v = a[0]  # action for velocity
        a_w = a[1]  # action for angular velocity
        # sample noise value and apply to v, w together with gains
        w=torch.distributions.Normal(0,torch.ones([2,1])).sample()*task_param[2:4]     
        vel = task_param[0,0] *a_v + w[0]
        ang_vel = task_param[1,0] * a_w + w[1]
        next_s = torch.stack((px, py, heading, vel, ang_vel)).view(1,-1)
        self.previous_action=a
        return next_s.view(-1,1)
        

    def constrained_cost_factor_range(self, alpha_param, max_cost=2.5,total_steps=40):
        # mag cost, assuming v is always max (which is true)
        # assuming 40 max steps
        mag_cost=alpha_param*total_steps
        max_dev_cost=max_cost-mag_cost
        max_beta_param=max_dev_cost/3 # theortically, 0 to 1/-1 then to opposite, has the min dev cost
        beta_param_range= [0.001, max_beta_param]
        return beta_param_range


    def reset_task_param(self,                
                pro_gains = None, 
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                goal_radius=None,
                mag_action_cost_factor=None,
                dev_action_cost_factor=None
                ):
        
        _pro_gains = torch.zeros(2)
        _pro_gains[0] = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1]) 
        _pro_gains[1] = torch.zeros(1).uniform_(self.gains_range[2], self.gains_range[3]) 
        
        _pro_noise_stds=torch.zeros(2)
        _pro_noise_stds[0]=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _pro_noise_stds[1]=torch.zeros(1).uniform_(self.std_range[2],self.std_range[3])
        
        _obs_gains = torch.zeros(2)
        _obs_gains[0] = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1]) 
        _obs_gains[1] = torch.zeros(1).uniform_(self.gains_range[2], self.gains_range[3])  
        
        _obs_noise_stds=torch.zeros(2)
        _obs_noise_stds[0]=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _obs_noise_stds[1]=torch.zeros(1).uniform_(self.std_range[2],self.std_range[3])
        
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])

        _mag_action_cost_factor = torch.zeros(1).uniform_(self.mag_action_cost_range[0], self.mag_action_cost_range[1])
        # _dev_action_cost_factor = torch.zeros(1).uniform_(self.dev_action_cost_range[0], self.dev_action_cost_range[1])
        beta_range=self.constrained_cost_factor_range(_mag_action_cost_factor)
        _dev_action_cost_factor = torch.zeros(1).uniform_(float(beta_range[0]),float(beta_range[1]))

        phi=torch.cat([_pro_gains,_pro_noise_stds,_obs_gains,_obs_noise_stds,_goal_radius,_mag_action_cost_factor,_dev_action_cost_factor])

        phi[0:2]=pro_gains if pro_gains is not None else phi[0:2]
        phi[2:4]=pro_noise_stds if pro_noise_stds is not None else phi[2:4]
        
        phi[4:6]=obs_gains if obs_gains is not None else phi[4:6]
        phi[6:8]=obs_noise_stds if obs_noise_stds is not None else phi[6:8]
        
        phi[8]=goal_radius if goal_radius is not None else phi[8]
        
        phi[9]=mag_action_cost_factor if mag_action_cost_factor is not None else phi[9]
        phi[10]=dev_action_cost_factor if dev_action_cost_factor is not None else phi[10]

        return col_vector(phi)


    def caculate_reward(self):
        '''
        calculate the reward, when agent stop (made the decision)

        input:
        output:
            reward as float, or tensor?
        '''
        if self.stop:
            reward=self.reward_function(self.stop, self.reached_goal(),
                self.b,self.P,self.phi[8,0],self.reward,
                self.goalx,self.goaly,time=self.episode_time)
        else:
            reward=0.
        cost, mag, dev=self.cost_function(self.a, self.previous_action,mag_scalar=self.phi[-2], dev_scalar=self.phi[-1])
        return reward-self.cost_scale*cost
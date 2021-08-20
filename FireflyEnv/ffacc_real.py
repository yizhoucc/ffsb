import gym
from gym import spaces
from gym.core import ObservationWrapper, RewardWrapper
from matplotlib.colors import rgb2hex
from numpy import pi
import numpy as np
from FireflyEnv.env_utils import *



class SmoothAction1d(gym.Env):
    # only the dev action cost, testing training for smooth actions

    def __init__(self) -> None:
        super().__init__()
        self.max_distance=0.9 
        self.min_distance=0.7
        # self.max_ctrl=1. 
        self.dt = 0.1
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Box(low=-1., high=1.,shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(5,),dtype=np.float32)
        self.gains_range=[1.,1.001]
        self.goal_radius_range=[0.1,0.1001]
        self.dev_action_cost_range=[0.,1.]
        self.reward_amout=100.
        self.trial_len=70
        self.trial_timer=0
        self.cost_scale=1.
        

    def step(self,action,debug={}):
        action=torch.tensor(action)
        self.trial_timer+=1
        self.s[0]-=self.s[1]*self.dt*self.phi[0]
        reward,stop=self.calculate_reward(action)
        self.trial_reward+=reward
        self.s[1]=action
        done=False
        if stop or self.trial_timer>self.trial_len:
            done=True
            print(self.trial_reward,bool((self.distance()-self.phi[1])<=0),self.is_skip())
        return self.wrap_decision_info(),reward,done,debug
    
    def if_stop(self,):
        stop=False
        self.if_start()
        if self.start and (abs(self.s[1])<0.05):
            stop=True
        return stop

    def if_start(self,):
        if not self.start:
            if self.s[1]>0.05:
                self.start=True
    
    def is_skip(self,):
        skip=False
        if self.trial_timer<4:
            skip=True
        return skip
    
    def calculate_reward(self,action):
        stop=self.if_stop()
        cost=self.calculate_cost(action)*self.cost_scale
        reward=0.
        if stop and self.distance()<self.phi[1]:
            reward=self.reward_amout
        return reward-cost,stop
    
    def calculate_cost(self,action):
        return (action-self.s[1])**2*self.phi[2]*200

    def reset(self,                
                pro_gains = None, 
                goal_radius=None,
                dev_action_cost_factor=None,
                d=None):
        d = torch.zeros(1).uniform_(self.min_distance, self.max_distance) if not d else d
        # self.previous_vctrl=torch.zeros(1).uniform_(0., self.max_ctrl)
        self.s=torch.tensor([d,0.])
        self.phi=self.reset_task_param(pro_gains=pro_gains, 
                            goal_radius=goal_radius,
                            dev_action_cost_factor=dev_action_cost_factor,
                            )
        self.trial_timer=0
        self.trial_reward=0.
        self.start=False
        return self.wrap_decision_info()
    
    def reset_task_param(self,                
                pro_gains = None, 
                goal_radius=None,
                dev_action_cost_factor=None,
                ):
        _pro_gains = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])
        _dev_action_cost_factor = torch.zeros(1).uniform_(self.dev_action_cost_range[0], self.dev_action_cost_range[1])
        phi=torch.cat([_pro_gains,
            _goal_radius,
            _dev_action_cost_factor,
        ])
        phi[0]=pro_gains if pro_gains is not None else phi[0]
        phi[1]=goal_radius if goal_radius is not None else phi[1]
        phi[2]=dev_action_cost_factor if dev_action_cost_factor is not None else phi[2]
        return phi

    def wrap_decision_info(self,):
        return torch.cat([self.s,self.phi,torch.ones(1)*self.trial_timer])

    def distance(self,):
        return abs(self.s[0])

class SmoothAction1dgamma(SmoothAction1d):

    def __init__(self) -> None:
        super().__init__()
        self.gamma_range=[0,1]
        low=-np.inf
        high=np.inf
        self.observation_space = spaces.Box(low=low, high=high,shape=(7,),dtype=np.float32)

    def calculate_reward(self,action):
        stop=self.if_stop()
        cost=self.calculate_cost(action)*self.cost_scale
        reward=0.
        if stop and self.s[0]<self.phi[1]:
            reward=self.reward_amout*self.phi[3]**self.trial_timer
        return reward-cost,stop
        
    def reset_task_param(self,                
                pro_gains = None, 
                goal_radius=None,
                dev_action_cost_factor=None,
                gamma=None,
                ):
        _pro_gains = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])
        _dev_action_cost_factor = torch.zeros(1).uniform_(self.dev_action_cost_range[0], self.dev_action_cost_range[1])
        _gamma = 1-0.07*torch.zeros(1).uniform_(self.gamma_range[0], self.gamma_range[1])
        phi=torch.cat([_pro_gains,
            _goal_radius,
            _dev_action_cost_factor,
            _gamma,
        ])
        phi[0]=pro_gains if pro_gains is not None else phi[0]
        phi[1]=goal_radius if goal_radius is not None else phi[1]
        phi[2]=dev_action_cost_factor if dev_action_cost_factor is not None else phi[2]
        phi[3]=gamma if gamma is not None else phi[3]
        return phi

    def reset(self,                
                pro_gains = None, 
                goal_radius=None,
                dev_action_cost_factor=None,
                gamma=None,
                ):
        self.distance = torch.zeros(1).uniform_(self.min_distance, self.max_distance)
        # self.previous_vctrl=torch.zeros(1).uniform_(0., self.max_ctrl)
        self.s=torch.tensor([self.distance,0.])
        self.phi=self.reset_task_param(pro_gains=pro_gains, 
                            goal_radius=goal_radius,
                            dev_action_cost_factor=dev_action_cost_factor,
                            gamma=gamma,
                            )
        self.trial_timer=0
        self.trial_reward=0.
        self.start=False
        return self.wrap_decision_info()

class SmoothAction1dRewardRate(SmoothAction1d):

    def __init__(self) -> None:
        super().__init__()
        low=-np.inf
        high=np.inf
        self.observation_space = spaces.Box(low=low, high=high,shape=(6,),dtype=np.float32)
        self.stop_punish=0.1

    def step(self,action,debug={}):
        self.prev_d=self.distance().item()
        action=torch.tensor(action)
        self.trial_timer+=1
        self.s[0]-=self.s[1]*self.dt*self.phi[0]
        reward,cost,stop=self.calculate_reward(action)
        self.trial_reward+=reward
        self.trial_cost+=cost
        self.s[1]=action
        done=False
        if (stop and self.distance()<=self.phi[1]) or self.trial_timer>self.trial_len:
            done=True 
            print((self.trial_reward-self.trial_cost)/self.trial_timer,
            bool((self.distance()-self.phi[1])<=0),self.is_skip())
        return  self.wrap_decision_info(),\
                (self.trial_reward-self.trial_cost)/self.trial_timer,\
                done,debug

    def reset(self,                
                pro_gains = None, 
                goal_radius=None,
                dev_action_cost_factor=None,
                d=None):
        d = torch.zeros(1).uniform_(self.min_distance, self.max_distance) if not d else d
        self.s=torch.tensor([d,0.])
        self.phi=self.reset_task_param(pro_gains=pro_gains, 
                            goal_radius=goal_radius,
                            dev_action_cost_factor=dev_action_cost_factor,
                            )
        self.trial_timer=0
        self.trial_reward=0.
        self.trial_cost=0.
        self.start=False
        return self.wrap_decision_info()

    def calculate_reward(self,action):
        stop=self.if_stop()
        cost=self.calculate_cost(action)*self.cost_scale
        reward=0.
        if stop and self.s[0]<self.phi[1]:
            reward=self.reward_amout
        else:
            reward=self.aux1()+self.aux2()
        return reward,cost,stop

    def aux1(self,): # reward when reducing distance
        return self.prev_d-self.distance()

    def aux2(self,): # punish when not doing anything
        scalar=0
        if self.if_stop(): # start then stop
            scalar=-1
        return scalar*self.stop_punish

class Smooth1dCrossFade(SmoothAction1dRewardRate):

    def __init__(self) -> None:
        super().__init__()
        low=-np.inf
        high=np.inf
        self.observation_space = spaces.Box(low=low, high=high,shape=(6,),dtype=np.float32)
        self.stop_punish=0.1
        self.rewardfun_ratio=1
    
    def rewardfun(self, action):
        reward,cost,stop=self.calculate_reward(action)
        self.trial_reward+=reward
        self.trial_cost+=cost
        reward_rate=(self.trial_reward-self.trial_cost)/(self.trial_timer+5)
        if stop or self.trial_timer>self.trial_len: # end of ep
            signal=self.rewardfun_ratio*(reward-cost)+(1-self.rewardfun_ratio)*reward_rate 
            done=True
            print(
            self.rewardfun_ratio,
            self.s[0],
            (self.trial_reward-self.trial_cost)/self.trial_timer,
            self.trial_reward,
            self.trial_cost,
            self.trial_timer,
            bool((self.distance()-self.phi[1])<0),self.is_skip())
        else:
            r=self.rewardfun_ratio*reward
            c=self.rewardfun_ratio*cost
            signal=r-c
            done=False

        return signal,done

    def step(self,action,debug={}):
        self.prev_d=abs(self.s[0].item())
        self.prev_a=self.s[1].item()
        action=torch.tensor(action)
        self.trial_timer+=1
        self.s[0]-=self.s[1]*self.dt*self.phi[0]
        self.s[1]=action

        signal,done=self.rewardfun(action)
        return self.wrap_decision_info(),signal,done,debug


        # if stop or self.trial_timer>self.trial_len:
        #     r=(self.trial_reward-self.trial_cost)/self.trial_timer*15
        #     print((
        #     self.s[0],
        #     self.trial_reward,
        #     self.trial_cost,
        #     r),
        #     self.trial_timer,
        #     bool((self.distance()-self.phi[1])<0),self.is_skip())
        #     return self.wrap_decision_info(),r,True,debug
        # else:
        #     return self.wrap_decision_info(),0.,False,debug
        # if stop or self.trial_timer>self.trial_len:
        #     r=(self.trial_reward-self.trial_cost)/self.trial_timer
        #     print((self.s[0],self.trial_reward,self.trial_cost),r,self.trial_timer,
        #     bool((self.distance()-self.phi[1])<0),self.is_skip())
        #     return self.wrap_decision_info(),\
        #         self.trial_reward-self.trial_cost,(stop or self.trial_timer>self.trial_len),debug
        # else:
        #     return self.wrap_decision_info(),0.,False,debug


    def calculate_reward(self,action):
        stop=self.if_stop()
        cost=self.calculate_cost(action)*self.cost_scale
        reward=0.
        if stop and self.distance()<=self.phi[1]:
            reward=self.reward_amout
        elif stop and self.distance()>self.phi[1]:
            reward=self.aux3()
        else:
            reward=self.aux1()+self.aux2()
        return reward,cost,stop

    def realtime_reward(self,action):
        return self.calculate_reward(action)
        
    def reward_rate(self,action):
        reward,cost,stop=self.calculate_reward(action)
        if stop or self.trial_timer>self.trial_len:
            r=(self.trial_reward-self.trial_cost)/self.trial_timer*15
            print((
            self.s[0],
            self.trial_reward,
            self.trial_cost,
            r),
            self.trial_timer,
            bool((self.distance()-self.phi[1])<0),self.is_skip())
            return self.wrap_decision_info(),r,True,debug
        else:
            return self.wrap_decision_info(),0.,False,debug
    
    def calculate_cost(self,action):
        return (action-self.prev_a)**2*self.phi[2]*200
        
    def aux1(self,): # reward when reducing distance
        return self.prev_d-abs(self.s[0])
    def aux2(self,): # punish when not doing anything
        scalar=0
        if self.if_stop(): # start then stop
            scalar=-1
        return scalar*self.stop_punish
    def aux3(self,):
        return self.phi[1]/(self.distance()+self.phi[1])



class Simple1d(gym.Env, torch.nn.Module): 

    def __init__(self,arg=None):
        super(Simple1d, self).__init__()
        self.arg=arg
        self.min_distance = 0.9
        self.max_distance = 1.
        self.terminal_vel = 0.05
        self.episode_len = 100
        self.dt = 0.1
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Box(low=-1., high=1.,shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,14),dtype=np.float32)
        self.cost_scale = arg.cost_scale 
        self.reward=self.arg.REWARD
        self.goal_radius_range =        self.arg.goal_radius_range
        self.gains_range=               self.arg.gains_range
        self.std_range =                self.arg.std_range
        self.recent100reward=[]
        self.recentreward=[]
        self.trial_counter=0
        self.session_len=500
        self.mag_action_cost_range =     self.arg.mag_action_cost_range
        self.dev_action_cost_range =     self.arg.dev_action_cost_range
        self.previous_v_range=[0.,1.]
        self.trial_base=arg.trial_base
        # self.inital_uncertainty_range=[0.,0.3]

    def reset(self,
                pro_gains = None, 
                pro_noise_stds = None,
                obs_noise_stds = None,
                goal_position=None,
                initv=None,
                new_session=True,
                ): 
        if new_session:
            # print('new sess')
            self.session_time=0
            self.session_trial_counter=0
            self.session_rewarded_trial=0
            self.session_reward=0.
            self.phi=self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_noise_stds=obs_noise_stds)
            self.theta=self.phi
            self.unpack_theta()
            self.Q = torch.zeros(2,2)
            self.Q[1,1]=self.pro_noise_hat**2
            self.R = self.obs_noise**2
            self.A = self.transition_matrix(task_param=self.theta) 
            if self.R < 1e-8:
                self.R=self.R+1e-8
        self.trial_sum_cost=0
        self.trial_mag=0
        self.trial_dev=0
        self.episode_time = torch.zeros(1) 
        self.reset_state(goal_position=goal_position,initv=initv)
        self.reset_obs()
        self.reset_belief()
        self.reset_decision_info()
        self.stop=False    
        return self.decision_info.view(1,-1)

    def unpack_theta(self):
        self.pro_gain=self.phi[0]
        self.pro_noise=self.phi[1]
        self.goal_r=self.phi[3]
        self.mag_cost=self.phi[4]
        self.dev_cost=self.phi[5]
        self.pro_gain_hat=self.theta[0]
        self.pro_noise_hat=self.theta[1]
        self.obs_noise=self.theta[2]
        self.goal_r_hat=self.theta[3]
        # self.inital_uncertainty=self.theta[6]

    def reset_task_param(self,                
                pro_gains = None, 
                pro_noise_stds = None,
                obs_noise_stds = None,
                goal_radius=None,
                mag_action_cost_factor=None,
                dev_action_cost_factor=None,
                inital_uncertainty=None
                ):
        _pro_gains = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  
        _pro_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _obs_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])
        _mag_action_cost_factor = torch.zeros(1).uniform_(self.mag_action_cost_range[0], self.mag_action_cost_range[1]) 
        _dev_action_cost_factor = torch.zeros(1).uniform_(self.dev_action_cost_range[0], self.dev_action_cost_range[1])
        # _inital_uncertainty = torch.zeros(1).uniform_(self.inital_uncertainty_range[0], self.inital_uncertainty_range[1])
        phi=torch.cat([_pro_gains,
            _pro_noise_stds,
            _obs_noise_stds,
            _goal_radius,
            _mag_action_cost_factor,
            _dev_action_cost_factor,
            # _inital_uncertainty
        ])
        phi[0]=pro_gains if pro_gains is not None else phi[0]
        phi[1]=pro_noise_stds if pro_noise_stds is not None else phi[1]
        phi[2]=obs_noise_stds if obs_noise_stds is not None else phi[2]
        phi[3]=goal_radius if goal_radius is not None else phi[3]
        phi[4]=mag_action_cost_factor if mag_action_cost_factor is not None else phi[4]
        phi[5]=dev_action_cost_factor if dev_action_cost_factor is not None else phi[5]
        # phi[6]=inital_uncertainty if inital_uncertainty is not None else phi[6]
        return phi

    def reset_state(self,goal_position=None,initv=None):
        self.goalx = torch.zeros(1).uniform_(self.min_distance, self.max_distance)  # max d to the goal is world boundary.
        if goal_position is not None:
            self.goalx = goal_position*torch.tensor(1.0)
        if initv is not None:
            vctrl=initv
        else:
            vctrl = torch.zeros(1).uniform_(self.previous_v_range[0], self.previous_v_range[1]) 
        self.previous_action=torch.tensor([[vctrl]])
        self.s = torch.tensor(
            # [[torch.distributions.Normal(0,torch.ones(1)).sample()*0.06*self.inital_uncertainty],
            [[0.],
            [vctrl*self.pro_gain]])

    def reset_belief(self): 
        # self.P = torch.tensor([[0.06*self.inital_uncertainty,0.],[0.,self.obs_noise**2]])
        self.P = torch.eye(2)*1e-8
        self.b = torch.tensor(
            [[0.],
            [1.]])*self.o

    def transition_matrix(self, task_param=None): 
        A = torch.zeros(2,2)
        A[0, 0] = 1
        # partial dev with v
        A[0, 1] = self.dt
        return A

    def reset_obs(self):
        self.o=self.observations(self.s)

    def reset_decision_info(self):
        self.decision_info = self.wrap_decision_info(b=self.b, time=self.episode_time, task_param=self.theta)
        self.decision_info = row_vector(self.decision_info)

    def step(self, action, onetrial=False):
        action=torch.tensor(action).reshape(1,-1)
        self.action=action
        self.episode_time=self.episode_time+1
        self.session_time+=1
        self.sys_vel=action
        self.stop=True if self.if_agent_stop(sys_vel=self.sys_vel) else False 
        # dynamic
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b, self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        # eval
        # if self.no_skip:
        #     end_current_ep=self.episode_time>=self.episode_len
        #     _,d= self.get_distance(state=self.b)
        #     if d<=self.theta[3] and self.stop:
        #         end_current_ep=True
        # else:
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
        # self.episode_reward = self.caculate_reward()
        self.episode_reward, cost, mag, dev = self.caculate_reward()
        self.trial_sum_cost += cost
        self.trial_mag += mag
        self.trial_dev += dev
        # if self.stop and end_current_ep:
        #     print('distance, ', self.get_distance()[1], 'goal', self.goal_r,'sysvel',self.sys_vel, 'time', self.episode_time)
        #     print('reward: {}, cost: {}, mag{}, dev{}'.format(self.episode_reward-self.trial_sum_cost, self.trial_sum_cost, self.trial_mag, self.trial_dev))
        #     # return self.decision_info, self.episode_reward-self.trial_sum_cost, end_current_ep, {}
        self.previous_action=action
        if len(self.recentreward)>=1000:
            self.recentreward.pop(0)
        self.recentreward.append(self.episode_reward-self.trial_sum_cost)
        # if len(self.recentreward)==1000 and self.session_time%100==0:
            # print('recent 1000 dt rewards',sum(self.recentreward))
        if end_current_ep:
            self.trial_counter+=1
            self.session_trial_counter+=1
            # skip=abs(self.goalx-self.s[0,0])>0.3
            # if skip:
            #     self.episode_reward=-20.
            self.session_reward+=self.episode_reward-self.trial_sum_cost
            if self.episode_reward>0.:
                self.session_rewarded_trial+=1
            # print(self.trial_counter)
            # if len(self.recent100reward)<100:
            #     pass
            # else:
            #     self.recent100reward.pop(0)
            # self.recent100reward.append(self.episode_reward)
            # if len(self.recent100reward)==100 and self.trial_counter%50==0:
                # print('recent mean rewards',sum(self.recent100reward)/100)
            if onetrial or self.trial_base:
                print(self.episode_reward-cost)
            else:
                self.reset(new_session=False)
        end_session=self.session_time>self.session_len
        if end_session:
            print('session reward', self.session_reward, 'percentagge:',self.session_rewarded_trial/self.session_trial_counter)
        if onetrial or self.trial_base:
            return self.decision_info, self.episode_reward-cost, end_current_ep, {}
        else:
            return self.decision_info, self.episode_reward-cost, end_session, {}
        
    def if_agent_stop(self,state=None,sys_vel=None):
        terminal_vel = self.terminal_vel
        state=self.s if state is None else state
        px, vel = torch.split(state.view(-1), 1)
        stop = (abs(vel) <= terminal_vel)
        if sys_vel is not None:
            stop=(abs(sys_vel) <= terminal_vel)
        return stop

    def caculate_reward(self):
        dev=(self.previous_action-self.action)**2*5*self.mag_cost
        mag=self.action**2*self.dev_cost*5000
        if self.stop:
            d=abs(self.goalx-self.s[0,0])
            r=self.goal_r
            if d<=r:
                reward=self.reward
            else:
                reward= 0.
        else:
            reward= 0.
        return reward, dev+mag, mag, dev

    def forward(self, action,task_param,state=None):

        if not self.if_agent_stop() and not self.agent_start:
            self.agent_start=True

        if self.agent_start:
            self.stop=True if self.if_agent_stop() else False 
        else:
            self.stop=False

        self.a=action.clone().detach()
        if state is None:
            self.s=self.state_step(action,self.s)
        else:
            self.s=state

        self.o=self.observations(self.s)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action,task_param=task_param)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=task_param)
        self.episode_time=self.episode_time+1
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
             
        return self.decision_info, end_current_ep

    def state_step(self,a, s):
        next_s = self.update_state(s)
        px, vel = torch.split(next_s.view(-1), 1)
        w=torch.distributions.Normal(0,torch.ones([1,1])).sample()*self.pro_noise  
        vel = self.pro_gain * (a[0] + w)
        next_s = torch.cat((px, vel[0])).view(1,-1)
        return next_s.view(-1,1)

    def update_state(self, state):
        px, vel = torch.split(state.view(-1), 1)
        px = px + vel * self.dt
        px = torch.clamp(px, -1., 1.) 
        next_s = torch.stack((px, vel))
        return next_s.view(-1,1)

    def observations(self, s): 
        on = torch.distributions.Normal(0,torch.tensor(1.)).sample()*self.theta[2]
        vel = s.view(-1)[-1] # 1,5 to vector and take last two
        ovel =  vel + on
        return ovel.view(-1,1)

    def apply_action(self,state,action,latent=False):
        px, vel = torch.split(state.view(-1), 1)
        if latent:
            vel=action[0]*self.pro_gain_hat
        else:
            vel=action[0]*self.pro_gain
        next_s = torch.stack((px, vel))
        return next_s.view(-1,1)

    def wrap_decision_info(self,b=None,P=None,time=None, task_param=None):
        task_param=self.theta if task_param is None else task_param
        b=self.b if b is None else b
        P=self.P if P is None else P
        px, vel = torch.split(b.view(-1), 1) # unpack state x
        r = (self.goalx-px).view(-1)
        vecL = P.view(-1)
        decision_info = torch.cat([r, vel, 
                        self.episode_time, self.previous_action[0],
                        vecL, task_param.view(-1)])
        return decision_info.view(1, -1)
        # only this is a row vector. everything else is col vector

    def belief_step(self, previous_b,previous_P, o, a, task_param=None):
        task_param = self.theta if task_param is None else task_param
        I = torch.eye(2)
        H = torch.zeros(1,2)
        H[-1,-1] = 1
        # prediction
        predicted_b = self.update_state(previous_b)
        predicted_b = self.apply_action(predicted_b,a,latent=True)
        predicted_P = self.A@(previous_P)@(self.A.t())+self.Q 
        if not is_pos_def(predicted_P):
            print('theta: ', task_param)
            print("predicted_P:", predicted_P)
            print('Q:',self.Q)
            print("previous_P:", previous_P)
            print("A:", A)
            APA = self.A@(previous_P)@(self.A.t())
            print("APA:", APA)
            print("APA +:", is_pos_def(APA))
        error = o - H@predicted_b 
        S = H@(predicted_P)@(H.t()) + self.R 
        K = predicted_P@(H.t())@(torch.inverse(S)) 
        b = predicted_b + K@(error)
        I_KH = I - K@(H)
        P = I_KH@(predicted_P)
        if not is_pos_def(P): 
            print("after update not pos def")
            print("updated P:", P)
            print('Q : ', self.Q)
            print('R : ', self.R)
            # print("K:", K)
            # print("H:", H)
            print("I - KH : ", I_KH)
            print("error : ", error)
            print('task parameter : ',task_param)
            P = (P + P.t()) / 2 + 1e-6 * I  # make symmetric to avoid computational overflows
        return b, P


class FireflyTrue1d_real(gym.Env, torch.nn.Module): 

    def __init__(self,arg=None,kwargs=None):
        super(FireflyTrue1d_real, self).__init__()
        self.arg=arg
        self.min_distance = 0.1
        self.max_distance = 1 
        self.terminal_vel = 0.05
        self.episode_len = 100
        self.dt = 0.1
        low=-np.inf
        high=np.inf
        # self.control_scalar=100
        self.action_space = spaces.Box(-np.ones(1), np.ones(1), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,15),dtype=np.float32)
        self.cost_function=self.action_cost_wrapper
        self.cost_scale=1
        self.presist_phi=           self.arg.presist_phi
        self.agent_knows_phi=       self.arg.agent_knows_phi
        self.reward=self.arg.REWARD
        # self.reward_function=reward_singleff.belief_reward_mc
        self.phi=None
        self.goal_radius_range =     self.arg.goal_radius_range
        self.gains_range=            self.arg.gains_range
        self.std_range =             self.arg.std_range
        self.tau_range = self.arg.tau_range
        self.mag_action_cost_range =     self.arg.mag_action_cost_range
        self.dev_action_cost_range =     self.arg.dev_action_cost_range
        self.session_len=2000

    def a_(self, tau, dt=None):
        dt=self.dt if dt is None else dt
        return torch.exp(-dt/tau)
    
    def vm_(self, tau, x=400, T=8.5):
        vm=x/2/tau*(1/(torch.log(torch.cosh(T/2/tau))))
        if vm==0: # too much velocity control rather than acc control
            vm=x/T*torch.ones(1)
        return vm

    def b_(self, tau, dt=None):
        dt=self.dt if dt is None else dt
        return self.vm_(tau)*(1-self.a_(tau))

    def reset(self,
                pro_gains = None, 
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                phi=None,
                theta=None,
                goal_radius_range=None,
                gains_range = None,
                std_range=None,
                goal_position=None,
                obs_traj=None,
                pro_traj=None,
                new_session=True,
                initv=None
                ): 
        if new_session:
            if phi is not None:
                self.phi=phi
            elif self.presist_phi and self.phi is not None:
                pass
            else: # when either not presist, or no phi.
                self.phi=self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_gains=obs_gains,obs_noise_stds=obs_noise_stds)
            if theta is not None:
                self.theta=theta
            else:
                self.theta=self.phi if self.agent_knows_phi else self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_gains=obs_gains,obs_noise_stds=obs_noise_stds)
            self.unpack_theta()
            self.session_timer=0
        self.reset_state(goal_position=goal_position,initv=initv)
        self.reset_belief()
        self.reset_obs()
        self.reset_decision_info()
        self.actions=[]
        self.obs_traj=obs_traj
        self.pro_traj=pro_traj
        self.trial_sum_cost=torch.zeros(1)
        self.trial_dev_costs=[]
        self.trial_mag_costs=[]
        self.stop=False
        self.agent_start=False
        self.trial_actions=[]
        return self.decision_info.view(1,-1)

    def unpack_theta(self):
        self.pro_gain=self.phi[0]
        self.pro_noise=self.phi[1]
        self.goal_r=self.phi[3]
        self.tau=self.phi[4]
        self.mag_cost=self.phi[5]
        self.dev_cost=self.phi[6]
        self.tau_a=self.a_(self.tau)
        self.tau_b=self.b_(self.tau)

        self.pro_gain_hat=self.theta[0]
        self.pro_noise_hat=self.theta[1]
        self.obs_noise=self.theta[2]
        self.goal_r_hat=self.theta[3]
        self.tau_hat=self.theta[4]
        self.tau_a_hat=self.a_(self.tau)
        self.tau_b_hat=self.b_(self.tau)


        self.Q = torch.zeros(2,2)
        self.Q[1,1]=(self.tau_b*self.pro_gain*self.pro_noise_hat)**2
        self.R = self.obs_noise**2
        self.A = self.transition_matrix(task_param=self.theta) 
        if self.R < 1e-8:
            self.R=self.R+1e-8

        # self.vm=self.vm_(self.tau)

    def reset_task_param(self,                
                pro_gains = None, 
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                goal_radius=None,
                tau=None,
                mag_action_cost_factor=None,
                dev_action_cost_factor=None,
                ):
        
        _pro_gains = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  
        _pro_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _obs_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])
        _tau = torch.zeros(1).uniform_(self.tau_range[0], self.tau_range[1])
        _mag_action_cost_factor = torch.zeros(1).uniform_(self.mag_action_cost_range[0], self.mag_action_cost_range[1]) 
        _dev_action_cost_factor = torch.zeros(1).uniform_(self.dev_action_cost_range[0], self.dev_action_cost_range[1])

        phi=torch.cat([_pro_gains,_pro_noise_stds,
            _obs_noise_stds,
                _goal_radius,_tau,_mag_action_cost_factor,_dev_action_cost_factor])

        phi[0]=pro_gains if pro_gains is not None else phi[0]
        phi[1]=pro_noise_stds if pro_noise_stds is not None else phi[1]
        phi[2]=obs_noise_stds if obs_noise_stds is not None else phi[2]
        phi[3]=goal_radius if goal_radius is not None else phi[3]
        phi[4]=tau if tau is not None else phi[4]
        phi[5]=mag_action_cost_factor if mag_action_cost_factor is not None else phi[5]
        phi[6]=dev_action_cost_factor if dev_action_cost_factor is not None else phi[6]

        return col_vector(phi)

    def reset_state(self,goal_position=None,initv=None):
        self.goalx = torch.zeros(1).uniform_(self.min_distance, self.max_distance)  # max d to the goal is world boundary.
        initv_ctrl = torch.zeros(1).uniform_(0.,1.) if initv is None else initv
        self.previous_action=initv_ctrl
        self.sys_vel=initv_ctrl
        if goal_position is not None:
            self.goalx = goal_position*torch.tensor(1.0)
        self.s = torch.cat([torch.zeros(1), initv_ctrl*self.theta[0]])
        self.s = self.s.view(-1,1) # column vector
        self.agent_start=False

    def reset_belief(self): 

        self.P = torch.eye(2)  * 1e-8 
        self.b = self.s.view(-1,1)  # column vector

    def reset_obs(self):
        self.o=torch.zeros(1).view(-1,1)       # column vector

    def reset_decision_info(self):
        self.episode_time = torch.zeros(1)  # int
        self.stop=False         # bool
        self.decision_info = self.wrap_decision_info(b=self.b, time=self.episode_time, task_param=self.theta)
        self.decision_info = row_vector(self.decision_info)
        self.previous_action=torch.zeros(1)
        self.trial_sum_cost=0
        self.trial_mag=0
        self.trial_dev=0

    def step(self, action, onetrial=False):
        action=torch.tensor(action).reshape(1)
        self.a=action
        self.actions.append(np.sign(action)[0])
        self.episode_time=self.episode_time+1
        self.sys_vel=self.sys_vel*self.tau_a+action[0]*self.pro_gain*self.tau_b
        if not self.if_agent_stop(sys_vel=self.sys_vel) and not self.agent_start:
            self.agent_start=True
        if self.agent_start:
            self.stop=True if self.if_agent_stop(sys_vel=self.sys_vel) else False 
        else:
            self.stop=False
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
        end_session=(self.session_timer>=self.session_len)
        self.episode_reward, cost,mag,dev=self.caculate_reward()
        self.previous_action=action
        self.trial_actions.append(action)
        self.trial_sum_cost+=cost
        self.trial_dev_costs.append(dev)
        self.trial_mag_costs.append(mag)
        # self.trial_mag+=mag
        # self.trial_dev+=dev
        if end_current_ep:
            reward_rate=(self.episode_reward-self.trial_sum_cost)/(self.episode_time+5)
            print('reward_rate, ', reward_rate, 'reward, ',self.episode_reward)
            if onetrial:
                print(self.episode_time)
                return self.decision_info, torch.zeros(1), end_current_ep, {}
            self.reset(new_session=False)
            return self.decision_info, reward_rate, end_session, {}
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
            # print('distance, ', self.get_distance(), 'goal', self.goal_r)
            # print('reward: {}, cost: {}, mag{}, dev{}'.format(self.episode_reward, self.trial_sum_cost, self.trial_mag, self.trial_dev))
        return self.decision_info, torch.zeros(1), end_session, {}

    def if_agent_stop(self,action=None,state=None,sys_vel=None):
        terminal=False
        terminal_vel = self.terminal_vel
        state=self.s if state is None else state
        px,  vel = torch.split(state.view(-1), 1)
        stop = (abs(vel) <= terminal_vel) and sum(self.actions)<self.episode_time
        if stop:
            terminal= True
        if sys_vel is not None:
            stop=(abs(sys_vel) <= terminal_vel)

        return terminal

    def forward(self, action,task_param,state=None, giving_reward=None):

        if not self.if_agent_stop() and not self.agent_start:
            self.agent_start=True

        if self.agent_start:
            self.stop=True if self.if_agent_stop() else False 
        else:
            self.stop=False
        self.a=action.clone().detach()
        if state is None:
            self.s=self.state_step(action,self.s)
        else:
            self.s=state

        self.o=self.observations(self.s)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action,task_param=task_param)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=task_param)
        self.episode_time=self.episode_time+1
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
             
        return self.decision_info, end_current_ep

    def state_step(self,a, s):

        next_s = self.update_state(s)
        px, vel = torch.split(next_s.view(-1), 1)
        if self.pro_traj is None:
            w=torch.distributions.Normal(0,torch.ones([1,1])).sample()*self.pro_noise  
        else:
            w=self.pro_traj[int(self.episode_time.item())]#*self.pro_noise  
        vel = self.tau_a * vel + self.tau_b*self.pro_gain * (torch.tensor(1.0)*a[0] + w)
        next_s = torch.cat((px, vel[0])).view(1,-1)
        self.previous_action=a
        return next_s.view(-1,1)

    def update_state(self, state):

        px, vel = torch.split(state.view(-1), 1)
        px = px + vel * self.dt
        px = torch.clamp(px, -self.max_distance, self.max_distance) 
        next_s = torch.stack((px, vel))
        return next_s.view(-1,1)

    def apply_action(self,state,action,latent=False):

        px, vel = torch.split(state.view(-1), 1)
        if latent:
            vel=vel*self.tau_a+action[0]*self.pro_gain_hat*self.tau_b_hat
        else:
            vel=vel*self.tau_a+action[0]*self.pro_gain*self.tau_b

        next_s = torch.stack((px, vel))
        return next_s.view(-1,1)

    def belief_step(self, previous_b,previous_P, o, a, task_param=None):
        
        task_param = self.theta if task_param is None else task_param
        I = torch.eye(2)
        H = torch.zeros(1,2)
        H[-1,-1] = 1

        # prediction
        predicted_b = self.update_state(previous_b)
        predicted_b = self.apply_action(predicted_b,a,latent=True)
        predicted_P = self.A@(previous_P)@(self.A.t())+self.Q 

        if not is_pos_def(predicted_P):
            print('theta: ', task_param)
            print("predicted_P:", predicted_P)
            print('Q:',self.Q)
            print("previous_P:", previous_P)
            print("A:", A)
            APA = self.A@(previous_P)@(self.A.t())
            print("APA:", APA)
            print("APA +:", is_pos_def(APA))

        error = o - H@predicted_b 
        S = H@(predicted_P)@(H.t()) + self.R 
        K = predicted_P@(H.t())@(torch.inverse(S)) 
        
        b = predicted_b + K@(error)
        I_KH = I - K@(H)
        P = I_KH@(predicted_P)

        if not is_pos_def(P): 
            print("after update not pos def")
            print("updated P:", P)
            print('Q : ', self.Q)
            print('R : ', self.R)
            # print("K:", K)
            # print("H:", H)
            print("I - KH : ", I_KH)
            print("error : ", error)
            print('task parameter : ',task_param)
            P = (P + P.t()) / 2 + 1e-6 * I  # make symmetric to avoid computational overflows
        
        return b, P

    def wrap_decision_info(self,b=None,P=None,time=None, task_param=None):

        task_param=self.theta if task_param is None else task_param
        b=self.b if b is None else b
        P=self.P if P is None else P
        px, vel = torch.split(b.view(-1), 1) # unpack state x
        r = (self.goalx-px).view(-1)
        vecL = P.view(-1)
        decision_info = torch.cat([r, vel, 
                        self.episode_time, self.previous_action,
                        vecL, task_param.view(-1)])
        return decision_info.view(1, -1)
        # only this is a row vector. everything else is col vector

    def caculate_reward(self):
        if self.stop:
            _,d= self.get_distance()
            if abs(d)<=self.phi[3,0]:
                reward=self.reward
            else:
                reward=0.
            #     reward = (self.max_distance-abs(d))/self.max_distance*self.reward/2
        else:
            # neg_velocity=self.s[1] if self.s[1]<0 else 0 # punish the backward
            # reward = neg_velocity
            reward=0.
        cost, mag, dev=self.cost_function(self.a, self.previous_action,mag_scalar=self.phi[5,0], dev_scalar=self.phi[6,0])
        return reward, self.cost_scale*cost, mag, dev

    def transition_matrix(self, task_param=None): 

        task_param = self.theta if task_param is None else task_param
        
        A = torch.zeros(2,2)
        A[0,0] = 1
        # partial dev with v
        A[0, 1] = self.dt
        A[1, 1] = self.tau_a

        return A

    def reached_goal(self):
        # use real location
        _,distance=self.get_distance(state=self.s)
        reached_bool= (distance<=self.phi[3,0])
        return reached_bool

    def observations(self, s): 

        if self.obs_traj is None:
            on = torch.distributions.Normal(0,torch.tensor(1.)).sample()*self.obs_noise
        else:
            on = self.obs_traj[int(self.episode_time.item())]#*self.obs_noise
        vel = s.view(-1)[-1] # 1,5 to vector and take last two
        ovel =  vel + on
        return ovel.view(-1,1)

    def observations_mean(self, s): # apply external noise and internal noise, to get observation
        # sample some noise
        vel = s.view(-1)[-1] # 1,5 to vector and take last two
        ovel = vel
        return ovel.view(-1,1)

    def get_distance(self, state=None): 

        state=self.s if state is None else state
        position = state[0]
        distance = self.goalx-state[0]
        return position, distance

    def action_cost_dev(self, action, previous_action):
        # sum up the norm squre for delta action
        cost=(action-previous_action)**2*100
        return cost

    def action_cost_magnitude(self,action):
        return (action)**2*20

    def action_cost_wrapper(self,action, previous_action,mag_scalar, dev_scalar):
        mag_cost=self.action_cost_magnitude(action)
        dev_cost=self.action_cost_dev(action, previous_action)
        total_cost=mag_scalar*mag_cost+dev_scalar*dev_cost
        return total_cost, mag_cost, dev_cost


class FireflyTrue1d(gym.Env, torch.nn.Module): 

    def __init__(self,arg=None,kwargs=None,debug=False,seed=None):
        super(FireflyTrue1d, self).__init__()
        self.arg=arg
        self.min_distance = 0.1
        self.max_distance = 1 
        self.terminal_vel = 0.05
        self.episode_len = 100
        self.dt = 0.1
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Box(-np.ones(1), np.ones(1), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,14),dtype=np.float32)
        self.cost_scale=1
        self.reward=                 self.arg.REWARD
        self.goal_radius_range =     self.arg.goal_radius_range
        self.gains_range=            self.arg.gains_range
        self.std_range =             self.arg.std_range
        self.mag_action_cost_range = self.arg.mag_action_cost_range
        self.dev_action_cost_range = self.arg.dev_action_cost_range
        self.session_len=1000
        self.debug=debug
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

    def reset(self,
                pro_gains = None, 
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                phi=None,
                theta=None,
                goal_position=None,
                obs_traj=None,
                pro_traj=None,
                new_session=True,
                initv=None
                ): 
        if new_session:
            if phi is not None:
                self.phi=phi
            if theta is not None:
                self.theta=theta
            if phi is None and theta is None:
                self.phi=self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_noise_stds=obs_noise_stds)
                self.theta=self.phi
            self.session_timer=0
        self.episode_timer = torch.zeros(1)  # int
        self.reset_state(goal_position=goal_position,initv=initv)
        self.reset_belief()
        self.reset_obs()
        self.reset_decision_info()
        self.actions=[]
        self.obs_traj=obs_traj
        self.pro_traj=pro_traj
        self.trial_sum_cost=torch.zeros(1)
        self.trial_dev_costs=[]
        self.trial_mag_costs=[]
        self.trial_actions=[]
        self.A=self.transition_matrix()
        self.Q = torch.zeros(2,2)
        self.Q[1,1]=(self.theta[0]*self.theta[1])**2
        self.R = self.theta[2]**2
        self.A = self.transition_matrix() 
        if self.R < 1e-8:
            self.R=self.R+1e-8
        return self.decision_info.view(1,-1)

    def reset_task_param(self,                
                pro_gains = None, 
                pro_noise_stds = None,
                obs_noise_stds = None,
                goal_radius=None,
                mag_action_cost_factor=None,
                dev_action_cost_factor=None,
                ):
        
        _pro_gains = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  
        _pro_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _obs_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])
        _mag_action_cost_factor = torch.zeros(1).uniform_(self.mag_action_cost_range[0], self.mag_action_cost_range[1]) 
        _dev_action_cost_factor = torch.zeros(1).uniform_(self.dev_action_cost_range[0], self.dev_action_cost_range[1])
        phi=torch.cat([_pro_gains,_pro_noise_stds,
            _obs_noise_stds,
                _goal_radius,_mag_action_cost_factor,_dev_action_cost_factor])
        phi[0]=pro_gains if pro_gains is not None else phi[0]
        phi[1]=pro_noise_stds if pro_noise_stds is not None else phi[1]
        phi[2]=obs_noise_stds if obs_noise_stds is not None else phi[2]
        phi[3]=goal_radius if goal_radius is not None else phi[3]
        phi[4]=mag_action_cost_factor if mag_action_cost_factor is not None else phi[4]
        phi[5]=dev_action_cost_factor if dev_action_cost_factor is not None else phi[5]
        return col_vector(phi)

    def reset_state(self,goal_position=None,initv=None):
        self.goalx = torch.zeros(1).uniform_(self.min_distance, self.max_distance)  # max d to the goal is world boundary.
        initv_ctrl = torch.zeros(1).uniform_(0.,1.) if initv is None else initv
        self.previous_action=initv_ctrl
        if goal_position is not None:
            self.goalx = goal_position*torch.tensor(1.0)
        self.s = torch.cat([torch.zeros(1), initv_ctrl*self.theta[0]])
        self.s = self.s.view(-1,1)

    def reset_belief(self): 
        self.P = torch.eye(2)  * 1e-8 
        self.b = self.s.view(-1,1)  # column vector

    def reset_obs(self):
        self.o=torch.zeros(1).view(-1,1)       # column vector

    def reset_decision_info(self):
        self.decision_info = self.wrap_decision_info(b=self.b, time=self.episode_timer, task_param=self.theta)
        self.decision_info = row_vector(self.decision_info)

    def step(self, action, onetrial=False):
        action=torch.tensor(action).reshape(1)
        self.a=action
        self.episode_timer+=1
        self.session_timer+=1
        end_current_ep=(action<self.terminal_vel or self.episode_timer>=self.episode_len)
        end_session=(self.session_timer>=self.session_len)
        self.episode_reward, cost,mag,dev=self.caculate_reward()
        self.previous_action=action
        self.trial_sum_cost+=cost
        if self.episode_reward != 0:
            print('reward-cost, ',self.episode_reward-self.trial_sum_cost)
        if self.debug:
            self.trial_actions.append(action)
            self.trial_dev_costs.append(dev)
            self.trial_mag_costs.append(mag)
        if end_current_ep:
            reward_rate=(self.episode_reward-self.trial_sum_cost)/(self.episode_timer+5)
            # print('reward_rate, ', reward_rate, 'reward, ',self.episode_reward)
            if onetrial:
                # print(self.episode_timer)
                return self.decision_info, 0., end_current_ep, {}
            self.reset(new_session=False)
            return self.decision_info, float(reward_rate), end_session, {}
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_timer,task_param=self.theta)
        return self.decision_info, 0., end_session, {}

    def forward(self, action,task_param,state=None):
        if not self.if_agent_stop() and not self.agent_start:
            self.agent_start=True
        if self.agent_start:
            self.stop=True if self.if_agent_stop() else False 
        else:
            self.stop=False
        self.a=action.clone().detach()
        if state is None:
            self.s=self.state_step(action,self.s)
        else:
            self.s=state

        self.o=self.observations(self.s)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action,task_param=task_param)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=task_param)
        self.episode_time=self.episode_time+1
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
             
        return self.decision_info, end_current_ep

    def state_step(self,a, s):
        next_s = self.update_state(s)
        px, vel = torch.split(next_s.view(-1), 1)
        if self.pro_traj is None:
            w=torch.distributions.Normal(0,torch.ones([1,1])).sample()*self.phi[1]  
        else:
            w=self.pro_traj[int(self.episode_time.item())]#*self.pro_noise  
        vel =self.phi[0] * (torch.tensor(1.0)*a[0] + w)
        next_s = torch.cat((px, vel[0])).view(1,-1)
        self.previous_action=a
        return next_s.view(-1,1)

    def update_state(self, state):
        px, vel = torch.split(state.view(-1), 1)
        px = px + vel * self.dt
        px = torch.clamp(px, -self.max_distance, self.max_distance) 
        next_s = torch.stack((px, vel))
        return next_s.view(-1,1)

    def apply_action(self,state,action,latent=False):
        px, vel = torch.split(state.view(-1), 1)
        if latent:
            vel=action[0]*self.theta[0]
        else:
            vel=action[0]*self.phi[0]

        next_s = torch.stack((px, vel))
        return next_s.view(-1,1)

    def belief_step(self, previous_b,previous_P, o, a, task_param=None):

        task_param = self.theta if task_param is None else task_param
        I = torch.eye(2)
        H = torch.zeros(1,2)
        H[-1,-1] = 1
        # prediction
        predicted_b = self.update_state(previous_b)
        predicted_b = self.apply_action(predicted_b,a,latent=True)
        predicted_P = self.A@(previous_P)@(self.A.t())+self.Q 
        if not is_pos_def(predicted_P):
            print('theta: ', task_param)
            print("predicted_P:", predicted_P)
            print('Q:',self.Q)
            print("previous_P:", previous_P)
            print("A:", A)
            APA = self.A@(previous_P)@(self.A.t())
            print("APA:", APA)
            print("APA +:", is_pos_def(APA))
        error = o - H@predicted_b 
        S = H@(predicted_P)@(H.t()) + self.R 
        K = predicted_P@(H.t())@(torch.inverse(S)) 
        b = predicted_b + K@(error)
        I_KH = I - K@(H)
        P = I_KH@(predicted_P)
        if not is_pos_def(P): 
            print("after update not pos def")
            print("updated P:", P)
            print('Q : ', self.Q)
            print('R : ', self.R)
            # print("K:", K)
            # print("H:", H)
            print("I - KH : ", I_KH)
            print("error : ", error)
            print('task parameter : ',task_param)
            P = (P + P.t()) / 2 + 1e-6 * I  # make symmetric to avoid computational overflows
        return b, P

    def wrap_decision_info(self,b=None,P=None,time=None, task_param=None):

        task_param=self.theta if task_param is None else task_param
        b=self.b if b is None else b
        P=self.P if P is None else P
        px, vel = torch.split(b.view(-1), 1) # unpack state x
        r = (self.goalx-px).view(-1)
        vecL = P.view(-1)
        decision_info = torch.cat([r, vel, 
                        self.episode_timer, self.previous_action,
                        vecL, task_param.view(-1)])
        return decision_info.view(1, -1)
        # only this is a row vector. everything else is col vector

    def caculate_reward(self):
        reward=0. 
        if self.a<self.terminal_vel:
            _,d= self.get_distance(state=self.b)
            if d<=self.phi[3,0]:
                reward=self.reward
        cost, mag, dev=self.action_cost_wrapper(self.a, self.previous_action,mag_scalar=self.phi[4,0], dev_scalar=self.phi[5,0])
        return reward, cost, mag, dev

    def transition_matrix(self, task_param=None): 
        A = torch.zeros(2,2)
        A[0, 0] = 1
        # partial dev with v
        A[0, 1] = self.dt
        return A

    def observations(self, s): 
        if self.obs_traj is None:
            on = torch.distributions.Normal(0,torch.tensor(1.)).sample()*self.theta[2]
        else:
            on = self.obs_traj[int(self.episode_time.item())]#*self.obs_noise
        vel = s.view(-1)[-1] # 1,5 to vector and take last two
        ovel =  vel + on
        return ovel.view(-1,1)

    def observations_mean(self, s): # apply external noise and internal noise, to get observation
        # sample some noise
        vel = s.view(-1)[-1] # 1,5 to vector and take last two
        ovel = vel
        return ovel.view(-1,1)

    def get_distance(self, state=None): 
        state=self.s if state is None else state
        position = state[0]
        distance = abs(self.goalx-state[0])
        return position, distance

    def action_cost_dev(self, action, previous_action):
        cost=(action-previous_action)**2*800
        return cost

    def action_cost_magnitude(self,action):
        return (action)**2*30

    def action_cost_wrapper(self,action, previous_action,mag_scalar, dev_scalar):
        mag_cost=self.action_cost_magnitude(action)*self.cost_scale
        dev_cost=self.action_cost_dev(action, previous_action)*self.cost_scale
        total_cost=mag_scalar*mag_cost+dev_scalar*dev_cost
        return total_cost, mag_cost, dev_cost


class Firefly_real_vel(gym.Env, torch.nn.Module): 

    def __init__(self,arg=None,kwargs=None):

        super(Firefly_real_vel, self).__init__()
        self.arg=arg
        self.min_distance = arg.goal_distance_range[0]
        self.max_distance = arg.goal_distance_range[1] 
        self.min_angle = -pi/4
        self.max_angle = pi/4
        self.terminal_vel = arg.TERMINAL_VEL 
        self.episode_len = arg.EPISODE_LEN
        self.dt = arg.DELTA_T
        self.reward=arg.REWARD
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Box(low=-1., high=1.,shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,39),dtype=np.float32)
        self.cost_function = self.action_cost_wrapper
        self.cost_scale = arg.cost_scale
        self.presist_phi=           arg.presist_phi
        self.agent_knows_phi=       arg.agent_knows_phi
        self.phi=None
        self.goal_radius_range =     arg.goal_radius_range
        self.gains_range=            arg.gains_range
        self.std_range =             arg.std_range
        self.mag_action_cost_range =     arg.mag_action_cost_range
        self.dev_action_cost_range =     arg.dev_action_cost_range


    def reset(self,
                pro_gains = None, 
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                phi=None,
                theta=None,
                goal_radius_range=None,
                gains_range = None,
                std_range=None,
                goal_position=None,
                initv=None,
                obs_traj=None,
                pro_traj=None,
                ): 
        if phi is not None:
            self.phi=phi
        elif self.presist_phi and self.phi is not None:
            pass
        else: # when either not presist, or no phi.
            self.phi=self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_gains=obs_gains,obs_noise_stds=obs_noise_stds)
        if theta is not None:
            self.theta=theta
        else:
            self.theta=self.phi if self.agent_knows_phi else self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_gains=obs_gains,obs_noise_stds=obs_noise_stds)
        
        self.unpack_theta()
        self.sys_vel=torch.zeros(1)
        self.obs_traj=obs_traj
        self.pro_traj=pro_traj
        self.stop=False 
        self.agent_start=False 
        self.episode_time = torch.zeros(1)
        self.previous_action=torch.tensor([[0.,0.]])
        self.trial_sum_cost=0
        self.trial_mag=0
        self.trial_dev=0
        self.reset_state(goal_position=goal_position,initv=initv)
        self.reset_belief()
        self.reset_obs()
        self.reset_decision_info()
        return self.decision_info.view(1,-1)


    def unpack_theta(self):
        self.pro_gainv=self.phi[0]
        self.pro_gainw=self.phi[1]
        self.pro_noisev=self.phi[2]
        self.pro_noisew=self.phi[3]
        self.goal_r=self.phi[6]
        self.mag_cost=self.phi[7]
        self.dev_cost=self.phi[8]

        self.pro_gainv_hat=self.theta[0]
        self.pro_gainw_hat=self.theta[1]
        self.pro_noisev_hat=self.theta[2]
        self.pro_noisew_hat=self.theta[3]
        self.obs_noisev=self.theta[4]
        self.obs_noisew=self.theta[5]
        self.goal_r_hat=self.theta[6]

        self.Q = torch.zeros(5,5)
        self.Q[3,3]=self.pro_noisev_hat**2 if self.pro_noisev_hat**2 > 1e-4 else self.pro_noisev_hat**2+1e-4
        self.Q[4,4]=self.pro_noisew_hat**2 if self.pro_noisew_hat**2 > 1e-4 else self.pro_noisew_hat**2+1e-4
        self.R = torch.zeros(2,2)
        self.R[0,0] = self.obs_noisev**2 if self.obs_noisev**2 > 1e-4 else self.obs_noisev**2+1e-4
        self.R[1,1] = self.obs_noisew**2 if self.obs_noisew**2 > 1e-4 else self.obs_noisew**2+1e-4


    def reset_task_param(self,                
                pro_gains = None,
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                goal_radius=None,
                mag_action_cost_factor=None,
                dev_action_cost_factor=None,
                ):
        _prov_gains = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  
        _prow_gains = torch.zeros(1).uniform_(self.gains_range[2], self.gains_range[3])  
        _prov_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _prow_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _obsv_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _obsw_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])
        _mag_action_cost_factor = torch.zeros(1).uniform_(self.mag_action_cost_range[0], self.mag_action_cost_range[1]) 
        _dev_action_cost_factor = torch.zeros(1).uniform_(self.dev_action_cost_range[0], self.dev_action_cost_range[1])

        phi=torch.cat([_prov_gains,_prow_gains,_prov_noise_stds,_prow_noise_stds,
            _obsv_noise_stds,_obsw_noise_stds,
                _goal_radius,_mag_action_cost_factor,_dev_action_cost_factor])

        phi[0]=pro_gains[0] if pro_gains is not None else phi[0]
        phi[1]=pro_gains[1] if pro_gains is not None else phi[1]
        phi[2]=pro_noise_stds[0] if pro_noise_stds is not None else phi[2]
        phi[3]=pro_noise_stds[1] if pro_noise_stds is not None else phi[3]
        phi[4]=obs_noise_stds[0] if obs_noise_stds is not None else phi[4]
        phi[5]=obs_noise_stds[1] if obs_noise_stds is not None else phi[5]
        phi[6]=goal_radius if goal_radius is not None else phi[6]
        phi[7]=mag_action_cost_factor if mag_action_cost_factor is not None else phi[7]
        phi[8]=dev_action_cost_factor if dev_action_cost_factor is not None else phi[8]

        return col_vector(phi)


    def reset_state(self,goal_position=None,initv=None):
        if goal_position is not None:
            self.goalx = goal_position[0]*torch.tensor(1.0)
            self.goaly = goal_position[1]*torch.tensor(1.0)
        else:
            distance = torch.zeros(1).uniform_(self.min_distance, self.max_distance)  
            angle = torch.zeros(1).uniform_(self.min_angle, self.max_angle) 
            self.goalx = torch.cos(angle)*distance
            self.goaly = torch.sin(angle)*distance
        vel = torch.zeros(1) 
        self.s = torch.tensor([[0.],[0.],[0.],[0.],[0.]])


    def reset_belief(self): 
        self.P = torch.eye(5) * 1e-8 
        self.b = self.s


    def reset_obs(self):
        self.o=torch.tensor([[0],[0]])


    def reset_decision_info(self):
        self.decision_info = self.wrap_decision_info(b=self.b, time=self.episode_time, task_param=self.theta)


    def step(self, action):
        action=torch.tensor(action).reshape(1,-1)
        self.a=action
        self.episode_time=self.episode_time+1
        self.sys_vel=torch.norm(action)
        if not self.if_agent_stop(sys_vel=self.sys_vel) and not self.agent_start:
            self.agent_start=True
        if self.agent_start:
            self.stop=True if self.if_agent_stop(sys_vel=self.sys_vel) else False 
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
        # dynamic
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b, self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        # eval
        self.episode_reward, cost, mag, dev = self.caculate_reward()
        if self.episode_reward>self.reward or self.episode_reward<0:
            raise RuntimeError('wrong reward function')
        self.trial_sum_cost += cost
        self.trial_mag += mag
        self.trial_dev += dev
        if self.stop:
            print('distance, ', self.get_distance()[1], 'goal', self.goal_r,'sysvel',self.sys_vel)
            print('reward: {}, cost: {}, mag{}, dev{}'.format(self.episode_reward-self.trial_sum_cost, self.trial_sum_cost, self.trial_mag, self.trial_dev))
        self.previous_action=action
        return self.decision_info, self.episode_reward-cost, end_current_ep, {}


    def if_agent_stop(self,sys_vel=None):
        stop=(sys_vel <= self.terminal_vel)
        return stop


    def forward(self, action,task_param,state=None, giving_reward=None):
        action=torch.tensor(action).reshape(1,-1)
        self.a=action
        self.episode_time=self.episode_time+1
        self.sys_vel=torch.norm(action)
        if not self.if_agent_stop(sys_vel=self.sys_vel) and not self.agent_start:
            self.agent_start=True
        if self.agent_start:
            self.stop=True if self.if_agent_stop(sys_vel=self.sys_vel) else False 
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
        # dynamic
        if state is None:
            self.s=self.state_step(action,self.s)
        else:
            self.s=state
            self.previous_action=action
        self.o=self.observations(self.s)
        self.o_mu=self.observations_mean(self.s)
        self.b, self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        return self.decision_info, end_current_ep


    def state_step(self,a, s):
        next_s = self.update_state(s)
        px, py, angle, v, w = torch.split(next_s.view(-1), 1)
        if self.pro_traj is None:
            noisev=torch.distributions.Normal(0,torch.ones(1)).sample()*self.pro_noisev  
            noisew=torch.distributions.Normal(0,torch.ones(1)).sample()*self.pro_noisew
        else:
            noisev=self.pro_traj[int(self.episode_time.item())]#*self.pro_noise  
            noisew=self.pro_traj[int(self.episode_time.item())]#*self.pro_noise  
        v = self.pro_gainv * torch.tensor(1.0)*a[0,0] + noisev
        w = self.pro_gainw * torch.tensor(1.0)*a[0,1] + noisew
        next_s = torch.cat((px, py, angle, v, w)).view(1,-1)
        return next_s.view(-1,1)


    def update_state(self, state): # run state dyanmic, changes x, y, theta
        px, py, angle, v, w = torch.split(state.view(-1), 1)
        px = px + v * self.dt * torch.cos(angle)
        py = py + v * self.dt * torch.sin(angle)
        angle = angle + w* self.dt
        px = torch.clamp(px, -self.max_distance, self.max_distance) 
        py = torch.clamp(py, -self.max_distance, self.max_distance) 
        next_s = torch.stack((px, py, angle, v, w ))
        return next_s.view(-1,1)


    def apply_action(self,state,action):
        px, py, angle, v, w = torch.split(state.view(-1), 1)
        v=action[0,0]*self.pro_gainv_hat
        w=action[0,1]*self.pro_gainw_hat
        next_s = torch.stack((px, py, angle, v, w))
        return next_s.view(-1,1)


    def belief_step(self, previous_b,previous_P, o, a, task_param=None):
        task_param = self.theta if task_param is None else task_param
        I = torch.eye(5)
        H = torch.tensor([[0.,0.,0.,1,0.],[0.,0.,0.,0.,1]])
        self.A = self.transition_matrix(previous_b, a) 
        # prediction
        predicted_b = self.update_state(previous_b)
        predicted_b = self.apply_action(predicted_b,a)
        predicted_P = self.A@(previous_P)@(self.A.t())+self.Q 
        if not is_pos_def(predicted_P):
            print('predicted not pos def')
            print('action ,', a)
            print('theta: ', task_param)
            print("predicted_P:", predicted_P)
            print('Q:',self.Q)
            print("previous_P:", previous_P)
            print("A:", self.A)
            APA = self.A@(previous_P)@(self.A.t())
            print("APA:", APA)
            print("APA +:", is_pos_def(APA))

        error = o - H@predicted_b
        S = H@(predicted_P)@(H.t()) + self.R 
        K = predicted_P@(H.t())@(torch.inverse(S)) 
        
        b = predicted_b + K@(error)
        I_KH = I - K@(H)
        P = I_KH@(predicted_P)

        if not is_pos_def(P): 
            print("after update not pos def")
            print("updated P:", P)
            print('Q : ', self.Q)
            print('R : ', self.R)
            # print("K:", K)
            # print("H:", H)
            print("I - KH : ", I_KH)
            print("error : ", error)
            print('task parameter : ',task_param)
            P = (P + P.t()) / 2 + 1e-6 * I  # make symmetric to avoid computational overflows
        
        return b, P


    def wrap_decision_info(self,b=None,P=None,time=None, task_param=None):

        task_param=self.theta if task_param is None else task_param
        b=self.b if b is None else b
        P=self.P if P is None else P
        px, py, angle, v, w = torch.split(b.view(-1), 1)
        relative_distance = torch.sqrt((self.goalx-px)**2+(self.goaly-py)**2).view(-1)
        relative_angle = torch.atan((self.goaly-py)/(self.goalx-px)).view(-1)-angle
        relative_angle = torch.clamp(relative_angle,-pi,pi)
        vecL = P.view(-1)
        decision_info = torch.cat([relative_distance, relative_angle, v, w,
                        self.episode_time, 
                        vecL, task_param.view(-1)])
        return decision_info.view(1, -1)
        # only this is a row vector. everything else is col vector


    def caculate_reward(self):
        if self.stop:
            _,d= self.get_distance()
            if d<=self.goal_r:
                reward=torch.tensor([1.])*self.reward
            else:
                reward = (self.goal_r/d)**3*self.reward
                if reward>self.reward:
                    raise RuntimeError
                if reward<0:
                    raise RuntimeError
                    # reward=torch.tensor([0.])
        else:
            reward=torch.tensor([0.])

        cost, mag, dev=self.cost_function(self.a, self.previous_action,mag_scalar=self.phi[7,0], dev_scalar=self.phi[8,0])
        return reward, self.cost_scale*cost, mag, dev


    def transition_matrix(self,b, a): 
        angle=b[2,0]
        vel=b[3,0]
        A = torch.zeros(5, 5)
        A[:3, :3] = torch.eye(3)
        # partial dev with theta
        A[0, 2] = - vel * torch.sin(angle) * self.dt*self.pro_gainv_hat
        A[1, 2] = vel * torch.cos(angle) * self.dt*self.pro_gainv_hat
        # partial dev with v
        A[0, 3] =   torch.cos(angle) * self.dt*self.pro_gainv_hat
        A[1, 3] =  torch.sin(angle) * self.dt*self.pro_gainv_hat
        # partial dev with w
        A[2, 4] =  self.dt*self.pro_gainw_hat
        return A


    def reached_goal(self):
        # use real location
        _,distance=self.get_distance(state=self.s)
        reached_bool= (distance<=self.goal_r)
        return reached_bool


    def observations(self, state): 
        s=state.clone()
        if self.obs_traj is None:
            noisev = torch.distributions.Normal(0,torch.tensor(1.)).sample()*self.obs_noisev
            noisew = torch.distributions.Normal(0,torch.tensor(1.)).sample()*self.obs_noisew
        else:
            noise = self.obs_traj[int(self.episode_time.item())]#*self.obs_noise
        vw = s.view(-1)[-2:] # 1,5 to vector and take last two
        vw[0] =  vw[0] + noisev
        vw[1] =  vw[1] + noisew
        return vw.view(-1,1)


    def observations_mean(self, s):
        vw = s.view(-1)[-2:]
        return vw.view(-1,1)


    def get_distance(self, state=None): 
        state=self.s if state is None else state
        px, py, angle, v, w = torch.split(state.view(-1), 1)
        position = torch.stack([px,py])
        relative_distance = torch.sqrt((self.goalx-px)**2+(self.goaly-py)**2).view(-1)
        return position, relative_distance


    def action_cost_dev(self, action, previous_action):
        cost=torch.norm(torch.tensor(1.0)*action-previous_action)**2
        return cost

    def action_cost_magnitude(self,action):
        return torch.norm(action)


    def action_cost_wrapper(self,action, previous_action,mag_scalar, dev_scalar):
        mag_cost=mag_scalar*self.action_cost_magnitude(action)
        dev_cost=dev_scalar*self.action_cost_dev(action, previous_action)
        total_cost=mag_cost+dev_cost
        return total_cost, mag_cost, dev_cost


class Firefly_2cost(Firefly_real_vel): 

    def action_cost_dev(self, action, previous_action):
        cost=(torch.norm(torch.tensor(1.0)*action-previous_action)*10)**2
        return cost

    def action_cost_magnitude(self,action):
        return torch.norm(action)

    def action_cost_wrapper(self,action, previous_action,mag_scalar, dev_scalar):
        mag_cost=mag_scalar*self.action_cost_magnitude(action)
        dev_cost=dev_scalar*self.action_cost_dev(action, previous_action)
        total_cost=mag_cost+dev_cost
        return total_cost, mag_cost, dev_cost


class Firefly_2devcost(Firefly_real_vel): 

    
    def __init__(self,arg=None,kwargs=None):

        super(Firefly_real_vel, self).__init__()
        self.arg=arg
        self.min_distance = arg.goal_distance_range[0]
        self.max_distance = arg.goal_distance_range[1] 
        self.min_angle = -pi/4
        self.max_angle = pi/4
        self.terminal_vel = arg.TERMINAL_VEL 
        self.episode_len = arg.EPISODE_LEN
        self.dt = arg.DELTA_T
        self.reward=arg.REWARD
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Box(low=-1., high=1.,shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,40),dtype=np.float32)
        self.cost_function = self.action_cost_wrapper
        self.cost_scale = arg.cost_scale
        self.presist_phi=           arg.presist_phi
        self.agent_knows_phi=       arg.agent_knows_phi
        self.phi=None
        self.goal_radius_range =     arg.goal_radius_range
        self.gains_range=            arg.gains_range
        self.std_range =             arg.std_range
        self.mag_action_cost_range =     arg.mag_action_cost_range
        self.dev_action_cost_range =     arg.dev_action_cost_range
        self.dev_vw_ratio_range =arg.dev_vw_ratio_range
    
    def reset_task_param(self,                
                pro_gains = None,
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                goal_radius=None,
                mag_action_cost_factor=None,
                dev_action_cost_factor=None,
                dev_vw_ratio=None
                ):
        _prov_gains = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  
        _prow_gains = torch.zeros(1).uniform_(self.gains_range[2], self.gains_range[3])  
        _prov_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _prow_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _obsv_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _obsw_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])
        _mag_action_cost_factor = torch.zeros(1).uniform_(self.mag_action_cost_range[0], self.mag_action_cost_range[1]) 
        _dev_action_cost_factor = torch.zeros(1).uniform_(self.dev_action_cost_range[0], self.dev_action_cost_range[1])
        _dev_vw_ratio = torch.zeros(1).uniform_(self.dev_vw_ratio_range[0], self.dev_vw_ratio_range[1])

        phi=torch.cat([_prov_gains,_prow_gains,_prov_noise_stds,_prow_noise_stds,
            _obsv_noise_stds,_obsw_noise_stds,
                _goal_radius,_mag_action_cost_factor,_dev_action_cost_factor,_dev_vw_ratio])

        phi[0]=pro_gains[0] if pro_gains is not None else phi[0]
        phi[1]=pro_gains[1] if pro_gains is not None else phi[1]
        phi[2]=pro_noise_stds[0] if pro_noise_stds is not None else phi[2]
        phi[3]=pro_noise_stds[1] if pro_noise_stds is not None else phi[3]
        phi[4]=obs_noise_stds[0] if obs_noise_stds is not None else phi[4]
        phi[5]=obs_noise_stds[1] if obs_noise_stds is not None else phi[5]
        phi[6]=goal_radius if goal_radius is not None else phi[6]
        phi[7]=mag_action_cost_factor if mag_action_cost_factor is not None else phi[7]
        phi[8]=dev_action_cost_factor if dev_action_cost_factor is not None else phi[8]
        phi[9]=dev_vw_ratio if dev_vw_ratio is not None else phi[9]
        return col_vector(phi)

    def action_cost_dev(self, action, previous_action):
        # action is row vector
        vcost=(action[0,0]-previous_action[0,0])*10*self.theta[9]
        wcost=(action[0,1]-previous_action[0,1])*10/self.theta[9]
        cost=(vcost+wcost)**2
        return cost


class FireflySepdev(Firefly_real_vel): 

    
    def __init__(self,arg=None,kwargs=None):

        super(Firefly_real_vel, self).__init__()
        self.arg=arg
        self.min_distance = arg.goal_distance_range[0]
        self.max_distance = arg.goal_distance_range[1] 
        self.min_angle = -pi/4
        self.max_angle = pi/4
        self.terminal_vel = arg.TERMINAL_VEL 
        self.episode_len = arg.EPISODE_LEN
        self.dt = arg.DELTA_T
        self.reward=arg.REWARD
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Box(low=-1., high=1.,shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,42),dtype=np.float32)
        self.cost_function = self.action_cost_wrapper
        self.cost_scale = arg.cost_scale
        self.presist_phi=           arg.presist_phi
        self.agent_knows_phi=       arg.agent_knows_phi
        self.phi=None
        self.goal_radius_range =     arg.goal_radius_range
        self.gains_range=            arg.gains_range
        self.std_range =             arg.std_range
        self.mag_action_cost_range =     arg.mag_action_cost_range
        self.dev_v_cost_range =     arg.dev_v_cost_range
        self.dev_w_cost_range =    arg.dev_w_cost_range    
    
    def reset_task_param(self,                
                pro_gains = None,
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                goal_radius=None,
                mag_action_cost_factor=None,
                dev_v_cost_factor=None,
                dev_w_cost_factor=None
                ):
        _prov_gains = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  
        _prow_gains = torch.zeros(1).uniform_(self.gains_range[2], self.gains_range[3])  
        _prov_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _prow_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _obsv_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _obsw_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])
        _mag_action_cost_factor = torch.zeros(1).uniform_(self.mag_action_cost_range[0], self.mag_action_cost_range[1]) 
        _dev_v_cost_factor = torch.zeros(1).uniform_(self.dev_v_cost_range[0], self.dev_v_cost_range[1])
        _dev_w_cost_factor = torch.zeros(1).uniform_(self.dev_w_cost_range[0], self.dev_w_cost_range[1])

        phi=torch.cat([_prov_gains,_prow_gains,_prov_noise_stds,_prow_noise_stds,
            _obsv_noise_stds,_obsw_noise_stds,
                _goal_radius,_mag_action_cost_factor,_dev_v_cost_factor,_dev_w_cost_factor])

        phi[0]=pro_gains[0] if pro_gains is not None else phi[0]
        phi[1]=pro_gains[1] if pro_gains is not None else phi[1]
        phi[2]=pro_noise_stds[0] if pro_noise_stds is not None else phi[2]
        phi[3]=pro_noise_stds[1] if pro_noise_stds is not None else phi[3]
        phi[4]=obs_noise_stds[0] if obs_noise_stds is not None else phi[4]
        phi[5]=obs_noise_stds[1] if obs_noise_stds is not None else phi[5]
        phi[6]=goal_radius if goal_radius is not None else phi[6]
        phi[7]=mag_action_cost_factor if mag_action_cost_factor is not None else phi[7]
        phi[8]=dev_v_cost_factor if dev_v_cost_factor is not None else phi[8]
        phi[9]=dev_w_cost_factor if dev_w_cost_factor is not None else phi[9]
        return col_vector(phi)

    def action_cost_dev(self, action, previous_action):
        # action is row vector
        vcost=(action[0,0]-previous_action[0,0])*10*self.theta[8]
        wcost=(action[0,1]-previous_action[0,1])*10*self.theta[9]
        cost=vcost**2+wcost**2
        return cost


    def wrap_decision_info(self,b=None,P=None,time=None, task_param=None):

        task_param=self.theta if task_param is None else task_param
        b=self.b if b is None else b
        P=self.P if P is None else P
        prevv, prevw = torch.split(self.previous_action.view(-1), 1)
        px, py, angle, v, w = torch.split(b.view(-1), 1)
        relative_distance = torch.sqrt((self.goalx-px)**2+(self.goaly-py)**2).view(-1)
        relative_angle = torch.atan((self.goaly-py)/(self.goalx-px)).view(-1)-angle
        relative_angle = torch.clamp(relative_angle,-pi,pi)
        vecL = P.view(-1)
        decision_info = torch.cat([relative_distance, relative_angle, v, w,
                        self.episode_time, prevv,prevw,
                        vecL, task_param.view(-1)])
        return decision_info.view(1, -1)
        # only this is a row vector. everything else is col vector


class FireflyFinal(Firefly_real_vel): 

    
    def __init__(self,arg=None,kwargs=None):

        super(Firefly_real_vel, self).__init__()
        self.arg=arg
        self.min_distance = arg.goal_distance_range[0]
        self.max_distance = arg.goal_distance_range[1] 
        self.min_angle = -pi/4
        self.max_angle = pi/4
        self.terminal_vel = arg.TERMINAL_VEL 
        self.episode_len = arg.EPISODE_LEN
        self.dt = arg.DELTA_T
        self.reward=arg.REWARD
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Box(low=-1., high=1.,shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,42),dtype=np.float32)
        self.cost_function = self.action_cost_wrapper
        self.cost_scale = arg.cost_scale
        self.presist_phi=           arg.presist_phi
        self.agent_knows_phi=       arg.agent_knows_phi
        self.phi=None
        self.goal_radius_range =     arg.goal_radius_range
        self.gains_range=            arg.gains_range
        self.std_range =             arg.std_range
        self.mag_action_cost_range =     arg.mag_action_cost_range
        self.dev_v_cost_range =     arg.dev_v_cost_range
        self.dev_w_cost_range =     arg.dev_w_cost_range  
        self.previous_v_range=[0.,1.]  

    def reset_state(self,goal_position=None,initv=None):
        if goal_position is not None:
            self.goalx = goal_position[0]*torch.tensor(1.0)
            self.goaly = goal_position[1]*torch.tensor(1.0)
        else:
            distance = torch.zeros(1).uniform_(self.min_distance, self.max_distance)  
            angle = torch.zeros(1).uniform_(self.min_angle, self.max_angle) 
            self.goalx = torch.cos(angle)*distance
            self.goaly = torch.sin(angle)*distance
        if initv is not None:
            vel=initv
        else:
            vel = torch.zeros(1).uniform_(self.previous_v_range[0], self.previous_v_range[1]) 
        self.previous_action=torch.tensor([[vel,0.]])
        self.s = torch.tensor([[0.],[0.],[0.],[vel],[0.]])
        
    def reset_task_param(self,                
                pro_gains = None,
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                goal_radius=None,
                mag_action_cost_factor=None,
                dev_v_cost_factor=None,
                dev_w_cost_factor=None
                ):
        _prov_gains = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  
        _prow_gains = torch.zeros(1).uniform_(self.gains_range[2], self.gains_range[3])  
        _prov_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _prow_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _obsv_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _obsw_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])
        _mag_action_cost_factor = torch.zeros(1).uniform_(self.mag_action_cost_range[0], self.mag_action_cost_range[1]) 
        _dev_v_cost_factor = torch.zeros(1).uniform_(self.dev_v_cost_range[0], self.dev_v_cost_range[1])
        _dev_w_cost_factor = torch.zeros(1).uniform_(self.dev_w_cost_range[0], self.dev_w_cost_range[1])

        phi=torch.cat([_prov_gains,_prow_gains,_prov_noise_stds,_prow_noise_stds,
            _obsv_noise_stds,_obsw_noise_stds,
                _goal_radius,_mag_action_cost_factor,_dev_v_cost_factor,_dev_w_cost_factor])

        phi[0]=pro_gains[0] if pro_gains is not None else phi[0]
        phi[1]=pro_gains[1] if pro_gains is not None else phi[1]
        phi[2]=pro_noise_stds[0] if pro_noise_stds is not None else phi[2]
        phi[3]=pro_noise_stds[1] if pro_noise_stds is not None else phi[3]
        phi[4]=obs_noise_stds[0] if obs_noise_stds is not None else phi[4]
        phi[5]=obs_noise_stds[1] if obs_noise_stds is not None else phi[5]
        phi[6]=goal_radius if goal_radius is not None else phi[6]
        phi[7]=mag_action_cost_factor if mag_action_cost_factor is not None else phi[7]
        phi[8]=dev_v_cost_factor if dev_v_cost_factor is not None else phi[8]
        phi[9]=dev_w_cost_factor if dev_w_cost_factor is not None else phi[9]
        return col_vector(phi)

    def step(self, action):
        action=torch.tensor(action).reshape(1,-1)
        self.a=action
        self.episode_time=self.episode_time+1
        self.sys_vel=torch.norm(action)
        if not self.if_agent_stop(sys_vel=self.sys_vel) and not self.agent_start:
            self.agent_start=True
        if self.agent_start:
            self.stop=True if self.if_agent_stop(sys_vel=self.sys_vel) else False 
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
        # dynamic
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b, self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        # eval
        self.episode_reward, cost, mag, dev = self.caculate_reward()
        self.trial_sum_cost += cost
        self.trial_mag += mag
        self.trial_dev += dev
        if end_current_ep:
            print('distance, ', self.get_distance()[1], 'goal', self.goal_r,'sysvel',self.sys_vel)
            print('reward: {}, cost: {}, mag{}, dev{}'.format(self.episode_reward-self.trial_sum_cost, self.trial_sum_cost, self.trial_mag, self.trial_dev))
            # return self.decision_info, self.episode_reward-self.trial_sum_cost, end_current_ep, {}
        self.previous_action=action
        return self.decision_info, self.episode_reward-cost, end_current_ep, {}

    def update_state(self, state): # run state dyanmic, changes x, y, theta
        px, py, angle, v, w = torch.split(state.view(-1), 1)
        if self.episode_time==1:
            return state
        else:
            if v==0:
                pass
            elif w==0:
                px = px + v*self.dt * torch.cos(angle)
                py = py + v*self.dt * torch.sin(angle)
            else:
                px = px-torch.sin(angle)*(v/w-(v*torch.cos(w*self.dt)/w))+torch.cos(angle)*((v*torch.sin(w*self.dt)/w))
                py = py+torch.cos(angle)*(v/w-(v*torch.cos(w*self.dt)/w))+torch.sin(angle)*((v*torch.sin(w*self.dt)/w))
            angle = angle + w* self.dt
            px = torch.clamp(px, -self.max_distance, self.max_distance) 
            py = torch.clamp(py, -self.max_distance, self.max_distance) 
            next_s = torch.stack((px, py, angle, v, w ))
            return next_s.view(-1,1)

    # def transition_matrix(self,b, a): 
    #     angle=b[2,0]
    #     v=b[3,0]
    #     w=b[4,0]
    #     A = torch.zeros(5, 5)
    #     A[:3, :3] = torch.eye(3)

    #     if w !=0:
    #         # partial dev with theta
    #         A[0, 2] = torch.cos(angle)*(-v*self.pro_gainv_hat/w/self.pro_gainw_hat + v*self.pro_gainv_hat*torch.cos(w*self.dt*self.pro_gainw_hat)/w/self.pro_gainw_hat) - v*self.pro_gainv_hat*torch.sin(angle)*torch.sin(w*self.dt*self.pro_gainw_hat)/w/self.pro_gainw_hat
    #         A[1, 2] = -(v*self.pro_gainv_hat/w/self.pro_gainw_hat-v*self.pro_gainv_hat*torch.cos(w*self.dt*self.pro_gainw_hat)/w/self.pro_gainw_hat)*torch.sin(angle) + v*self.pro_gainv_hat*torch.cos(angle)*torch.sin(w*self.dt*self.pro_gainw_hat)/w/self.pro_gainw_hat
    #         # partial dev with v
    #         A[0, 3] = (-1/w/self.dt/self.pro_gainw_hat+torch.cos(w*self.dt*self.pro_gainw_hat)/w/self.dt/self.pro_gainw_hat)*torch.sin(angle) + torch.cos(angle)*torch.sin(w*self.dt*self.pro_gainw_hat)/w/self.dt/self.pro_gainw_hat
    #         A[1, 3] =  torch.cos(angle)*(1/w/self.dt/self.pro_gainw_hat-torch.cos(w*self.dt*self.pro_gainw_hat)/w/self.dt/self.pro_gainw_hat)+torch.sin(angle)*torch.sin(w*self.dt*self.pro_gainw_hat)/w/self.dt/self.pro_gainw_hat
    #         # partial dev with w
    #         A[0, 4] = v*self.pro_gainv_hat*torch.cos(angle)*torch.cos(w*self.dt*self.pro_gainw_hat)/w/self.pro_gainw_hat - v*self.dt*self.pro_gainv_hat*torch.cos(angle)*torch.sin(w*self.dt*self.pro_gainw_hat)/((w*self.dt*self.pro_gainw_hat)**2) + torch.sin(angle)*(v*self.dt*self.pro_gainv_hat/((w*self.dt*self.pro_gainw_hat)**2)-v*self.dt*self.pro_gainv_hat*torch.cos(w*self.dt*self.pro_gainw_hat)/((w*self.dt*self.pro_gainw_hat)**2)-v*self.pro_gainv_hat*torch.sin(w*self.dt*self.pro_gainw_hat)/w/self.pro_gainw_hat)
    #         A[1, 4] = v*self.pro_gainv_hat*torch.cos(w*self.dt*self.pro_gainw_hat)*torch.sin(angle)/w/self.pro_gainw_hat - v*self.dt*self.pro_gainv_hat*torch.sin(angle)*torch.sin(w*self.dt*self.pro_gainw_hat)/((w*self.dt*self.pro_gainw_hat)**2) + torch.cos(angle)*(-v*self.dt*self.pro_gainv_hat/((w*self.dt*self.pro_gainw_hat)**2)+v*self.dt*self.pro_gainv_hat*torch.cos(w*self.dt*self.pro_gainw_hat)/((w*self.dt*self.pro_gainw_hat)**2)+v*self.pro_gainv_hat*torch.sin(w*self.dt*self.pro_gainw_hat)/w/self.pro_gainw_hat)
    #         A[2, 4] =  self.dt*self.pro_gainw_hat
    #     else:
    #         # partial dev with theta
    #         A[0, 2] = -v*self.dt*self.pro_gainv_hat*torch.sin(angle)
    #         A[1, 2] = v*self.dt*self.pro_gainv_hat*torch.cos(angle)
    #         # partial dev with v
    #         A[0, 3] = torch.cos(angle)
    #         A[1, 3] = torch.sin(angle)
    #         # partial dev with w
    #         A[2, 4] =  self.dt*self.pro_gainw_hat
    #     return A


    def action_cost_dev(self, action, previous_action):
        # action is row vector
        action[0,0]=1.0 if action[0,0]>1.0 else action[0,0]
        action[0,1]=1.0 if action[0,1]>1.0 else action[0,1]
        action[0,1]=-1.0 if action[0,1]<-1.0 else action[0,1]

        vcost=(action[0,0]-previous_action[0,0])*self.theta[8]
        wcost=(action[0,1]-previous_action[0,1])*self.theta[9]
        cost=vcost**2+wcost**2
        cost=cost*10
        return cost

    def wrap_decision_info(self,b=None,P=None,time=None, task_param=None):
        task_param=self.theta if task_param is None else task_param
        b=self.b if b is None else b
        P=self.P if P is None else P
        prevv, prevw = torch.split(self.previous_action.view(-1), 1)
        px, py, angle, v, w = torch.split(b.view(-1), 1)
        relative_distance = torch.sqrt((self.goalx-px)**2+(self.goaly-py)**2).view(-1)
        relative_angle = torch.atan((self.goaly-py)/(self.goalx-px)).view(-1)-angle
        relative_angle = torch.clamp(relative_angle,-pi,pi)
        vecL = P.view(-1)
        decision_info = torch.cat([
            relative_distance, 
            relative_angle, 
            v, 
            w,
            self.episode_time, 
            prevv,
            prevw,
            vecL, 
            task_param.view(-1)])
        return decision_info.view(1, -1)
        # only this is a row vector. everything else is col vector

    def action_cost_wrapper(self,action, previous_action):
        mag_cost=self.action_cost_magnitude(action)*self.theta[7]
        dev_cost=self.action_cost_dev(action, previous_action)
        total_cost=mag_cost+dev_cost
        return total_cost, mag_cost, dev_cost

    def caculate_reward(self):
        if self.stop:
            _,d= self.get_distance()
            if d<=self.goal_r:
                reward=torch.tensor([1.])*self.reward
            else:
                if self.goal_r/d>1/3:
                    reward = (self.goal_r/d)**1*self.reward*0.8
                else: reward=torch.tensor([0.])
                if reward>self.reward:
                    raise RuntimeError
                if reward<0:
                    raise RuntimeError
                    # reward=torch.tensor([0.])
        else:
            # _,d= self.get_distance()
            # reward=(self.goal_r/d)**2
            reward=torch.tensor([0.])
        cost, mag, dev=self.cost_function(self.a, self.previous_action)
        print('reward',reward, 'dev',dev, 'actions',self.previous_action,self.a)
        reward=reward
        return reward, self.cost_scale*cost, mag, dev


class FireflyFinal2(FireflyFinal): 

    def __init__(self,arg=None,kwargs=None):
        super(FireflyFinal2, self).__init__(arg=arg)
        self.inital_cov_range=[0.,1]  # max==goal r
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Box(low=-1., high=1.,shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,27),dtype=np.float32)
        self.no_skip=False
        self.session_len=60*10*5 # 5 min session
        self.prev_control_range=[0.,1.]  
        # self.time_cost_range=[0.,1.]

    def reset_task_param(self,                
                pro_gains = None,
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                goal_radius=None,
                mag_action_cost_factor=None,
                dev_v_cost_factor=None,
                dev_w_cost_factor=None,
                inital_x_std=None,
                inital_y_std=None,
                # time_cost=None
                ):
        _prov_gains = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  
        _prow_gains = torch.zeros(1).uniform_(self.gains_range[2], self.gains_range[3])  
        _prov_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _prow_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _obsv_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _obsw_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])
        _mag_action_cost_factor = torch.zeros(1).uniform_(self.mag_action_cost_range[0], self.mag_action_cost_range[1]) 
        _dev_v_cost_factor = torch.zeros(1).uniform_(self.dev_v_cost_range[0], self.dev_v_cost_range[1])
        _dev_w_cost_factor = torch.zeros(1).uniform_(self.dev_w_cost_range[0], self.dev_w_cost_range[1])
        _inital_x_std = torch.zeros(1).uniform_(self.inital_cov_range[0], self.inital_cov_range[1])
        _inital_y_std = torch.zeros(1).uniform_(self.inital_cov_range[0], self.inital_cov_range[1])
        # _time_cost = torch.zeros(1).uniform_(self.time_cost_range[0], self.time_cost_range[1])

        phi=torch.cat([_prov_gains,_prow_gains,_prov_noise_stds,_prow_noise_stds,
            _obsv_noise_stds,_obsw_noise_stds,
                _goal_radius,_mag_action_cost_factor,_dev_v_cost_factor,_dev_w_cost_factor,
                _inital_x_std,_inital_y_std,
                # _time_cost
                ])

        phi[0]=pro_gains[0] if pro_gains is not None else phi[0]
        phi[1]=pro_gains[1] if pro_gains is not None else phi[1]
        phi[2]=pro_noise_stds[0] if pro_noise_stds is not None else phi[2]
        phi[3]=pro_noise_stds[1] if pro_noise_stds is not None else phi[3]
        phi[4]=obs_noise_stds[0] if obs_noise_stds is not None else phi[4]
        phi[5]=obs_noise_stds[1] if obs_noise_stds is not None else phi[5]
        phi[6]=goal_radius if goal_radius is not None else phi[6]
        phi[7]=mag_action_cost_factor if mag_action_cost_factor is not None else phi[7]
        phi[8]=dev_v_cost_factor if dev_v_cost_factor is not None else phi[8]
        phi[9]=dev_w_cost_factor if dev_w_cost_factor is not None else phi[9]
        phi[10]=inital_x_std if inital_x_std is not None else phi[10]
        phi[11]=inital_y_std if inital_y_std is not None else phi[11]
        # phi[12]=time_cost if time_cost is not None else phi[12]

        return col_vector(phi)

    def reset(self,
                pro_gains = None, 
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                phi=None,
                theta=None,
                goal_radius_range=None,
                gains_range = None,
                std_range=None,
                goal_position=None,
                initv=None,
                initw=None,
                obs_traj=None,
                pro_traj=None,
                new_session=True,
                ): 
        if new_session:
            self.session_time = torch.zeros(1)
            if phi is not None:
                self.phi=phi
            elif self.presist_phi and self.phi is not None:
                pass
            else: # when either not presist, or no phi.
                self.phi=self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_gains=obs_gains,obs_noise_stds=obs_noise_stds)
            if theta is not None:
                self.theta=theta
            else:
                self.theta=self.phi if self.agent_knows_phi else self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_gains=obs_gains,obs_noise_stds=obs_noise_stds)
            self.unpack_theta()
            print('new_session')
        self.sys_vel=torch.zeros(1)
        self.obs_traj=obs_traj
        self.pro_traj=pro_traj
        self.stop=False 
        self.agent_start=False 
        self.episode_time = torch.zeros(1)
        self.episode_reward=torch.zeros(1)
        self.previous_action=torch.tensor([[0.,0.]])
        self.trial_sum_cost=torch.zeros(1)
        self.trial_mag_costs=[]
        self.trial_dev_costs=[]
        self.trial_actions=[]
        self.reset_state(goal_position=goal_position,initv=initv,initw=initw)
        self.reset_belief()
        self.reset_obs()
        self.reset_decision_info()
        return self.decision_info.view(1,-1)

    def reset_state(self,goal_position=None,initv=None, initw=None):
        if goal_position is not None:
            self.goalx = goal_position[0]*torch.tensor(1.0)
            self.goaly = goal_position[1]*torch.tensor(1.0)
        else:
            distance = torch.zeros(1).uniform_(self.min_distance, self.max_distance)  
            angle = torch.zeros(1).uniform_(self.min_angle, self.max_angle) 
            self.goalx = torch.cos(angle)*distance
            self.goaly = torch.sin(angle)*distance
        if initv is not None:
            v_ctrl=initv
        else:
            v_ctrl = torch.zeros(1).uniform_(self.prev_control_range[0], self.prev_control_range[1]) 
        if initw is not None:
            w_ctrl=initw
        else:
            w_ctrl = torch.zeros(1).uniform_(-self.prev_control_range[1]/2, self.prev_control_range[1]/2) 
        self.previous_action=torch.tensor([[v_ctrl,w_ctrl]])
        self.s = torch.tensor(
            [[torch.distributions.Normal(0,torch.ones(1)).sample()*0.06*self.theta[10]],
            [torch.distributions.Normal(0,torch.ones(1)).sample()*0.06*self.theta[11]],
            [0.],
            [v_ctrl*self.phi[0]],
            [w_ctrl*self.phi[1]]])

    def reset_obs(self):
        self.o=self.observations(self.s)

    def reset_belief(self): 
        self.P = torch.eye(5) * 1e-8 
        self.P[0,0]=(self.theta[10]*0.06)**2 # sigma xx
        self.P[1,1]=(self.theta[11]*0.06)**2 # sigma yy
        self.b = torch.tensor(
            [[0.],
            [0.],
            [0.],
            [1],
            [1]])
        self.b=self.b*self.s

    def caculate_reward(self, action):
        if self.stop: # only evaluate reward when stop.
            rew_std = self.phi[6]/2 
            mu = torch.Tensor([self.goalx,self.goaly])-self.b[:2,0]
            R = torch.eye(2)*rew_std**2 
            P = self.P[:2,:2]
            S = R+P
            if not is_pos_def(S):
                print('R+P is not positive definite!')
            alpha = -0.5 * mu @ S.inverse() @ mu.t()
            reward = torch.exp(alpha) /2 / pi /torch.sqrt(S.det())
            # normalization -> to make max reward as 1
            mu_zero = torch.zeros(1,2)
            alpha_zero = -0.5 * mu_zero @ R.inverse() @ mu_zero.t()
            reward_zero = torch.exp(alpha_zero) /2 / pi /torch.sqrt(R.det())
            reward = reward/reward_zero
            if reward > 1.:
                print('reward is wrong!', reward)
                print('mu', mu)
                print('P', P)
                print('R', R)
            reward = self.reward * reward  
            cost, mag, dev=self.cost_function(action, self.previous_action)
            # print('reward ',reward, 'dev ',dev, 'actions ',self.previous_action,self.a, 'time ', self.episode_time)
        else: # not stop, zero reward.
            reward=torch.tensor([0.])
            cost, mag, dev=self.cost_function(action, self.previous_action)
        return reward, self.cost_scale*cost, self.cost_scale*mag, self.cost_scale*dev

    def action_cost_wrapper(self,action, previous_action):
        mag_cost=self.action_cost_magnitude(action)
        dev_cost=self.action_cost_dev(action, previous_action)
        total_cost=mag_cost+dev_cost
        return total_cost, mag_cost, dev_cost

    def is_skip(self, distance,stop):
        return (distance>2*self.phi[6]) & stop
        
    def action_cost_magnitude(self,action):
        cost=torch.norm(action)
        # scalar=self.reward/(1/0.4/self.dt)
        mincost=1/0.4/self.dt
        scalar=self.reward/mincost
        return scalar*cost*self.theta[7]

    def action_cost_dev(self, action, previous_action):
        # action is row vector
        action[0,0]=1.0 if action[0,0]>1.0 else action[0,0]
        action[0,1]=1.0 if action[0,1]>1.0 else action[0,1]
        action[0,1]=-1.0 if action[0,1]<-1.0 else action[0,1]
        vcost=(action[0,0]-previous_action[0,0])*self.theta[8]
        wcost=(action[0,1]-previous_action[0,1])*self.theta[9]
        cost=vcost**2+wcost**2
        mincost=(1/(1/0.4/self.dt))**2 *2*1/0.4/self.dt
        # mincost=2
        scalar=self.reward/mincost
        cost=cost*scalar
        return cost

    def step(self, action):
        action=torch.tensor(action).reshape(1,-1)
        self.episode_time+=1
        self.session_time+=1
        self.sys_vel=torch.norm(action)
        if not self.if_agent_stop(sys_vel=self.sys_vel) and not self.agent_start:
            self.agent_start=True
        if self.agent_start:
            self.stop=True if self.if_agent_stop(sys_vel=self.sys_vel) else False 
        # eval
        distance=self.get_distance(state=self.b)[1]
        if self.no_skip:
            end_current_ep=self.episode_time>=self.episode_len
            if (distance<self.phi[6]) and self.stop:
                end_current_ep=True
        else:
            end_current_ep=(self.stop or self.episode_time>=self.episode_len)
        self.episode_reward, cost, mag, dev = self.caculate_reward(action) # using prev action
        self.trial_sum_cost+=cost
        # self.trial_mag_costs.append(mag)
        # self.trial_dev_costs.append(dev)
        # self.trial_actions.append(action)
        # print('distance, ', distance, 'goal', self.goal_r,'sysvel',self.sys_vel, 'time', self.episode_time)
        # print('reward: {}, cost: {}, mag{}, dev{}'.format(self.episode_reward-self.trial_sum_cost, self.trial_sum_cost, self.trial_mag, self.trial_dev))
            # return self.decision_info, self.episode_reward-self.trial_sum_cost, end_current_ep, {}
        end_session=self.session_time>=self.session_len
        if end_current_ep and (not end_session): # reset but keep theta
            reward_rate=(self.episode_reward-self.trial_sum_cost)/(self.episode_time+5)
            if reward_rate!=0:
                # print('reward, ', self.episode_reward, ' costs, ', self.trial_sum_cost)
                print('reward rate, ',reward_rate)
            # print('distance, ', distance)
            self.reset(new_session=False)
            return self.decision_info, reward_rate, end_session, {}
        # dynamic
        self.previous_action=action
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b, self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        return self.decision_info, torch.zeros(1), end_session, {}
        
    def forward(self, action, task_param, state=None, giving_reward=None):
        action=torch.tensor(action).reshape(1,-1)
        self.episode_time+=1
        self.session_time+=1
        self.sys_vel=torch.norm(action)
        if not self.if_agent_stop(sys_vel=self.sys_vel) and not self.agent_start:
            self.agent_start=True
        if self.agent_start:
            self.stop=True if self.if_agent_stop(sys_vel=self.sys_vel) else False 
        # eval
        if self.no_skip:
            end_current_ep=self.episode_time>=self.episode_len
            _,d= self.get_distance(state=self.b)
            if d<=self.theta[6] and self.stop:
                end_current_ep=True
        else:
            end_current_ep=(self.stop or self.episode_time>=self.episode_len)
        self.episode_reward, cost, mag, dev = self.caculate_reward(action) # using prev action
        self.trial_sum_cost+=cost
        self.trial_mag_costs.append(mag)
        self.trial_dev_costs.append(dev)
        self.trial_actions.append(action)
        if self.stop and end_current_ep:
            self.trial_mag=sum(self.trial_mag_costs)
            self.trial_dev=sum(self.trial_dev_costs)
            self.reward_rate=(self.episode_reward-self.trial_sum_cost)/(self.episode_time+5)
            # print('distance, ', self.get_distance()[1], 'goal', self.goal_r,'sysvel',self.sys_vel, 'time', self.episode_time)
            # print('reward: {}, cost: {}, mag{}, dev{}'.format(self.episode_reward-self.trial_sum_cost, self.trial_sum_cost, self.trial_mag, self.trial_dev))
            # return self.decision_info, self.episode_reward-self.trial_sum_cost, end_current_ep, {}
        # dynamic
        self.previous_action=action
        if state is None:
            self.s=self.state_step(action,self.s)
        else:
            self.s=state
            self.previous_action=action
        self.o=self.observations(self.s)
        self.o_mu=self.observations_mean(self.s)
        self.b, self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        # if end_current_ep:
        #     print(self.episode_reward)
        return self.decision_info, end_current_ep

    def wrap_decision_info(self,b=None,P=None,time=None, task_param=None):
        task_param=self.theta if task_param is None else task_param
        b=self.b if b is None else b
        P=self.P if P is None else P
        prevv, prevw = torch.split(self.previous_action.view(-1), 1)
        px, py, angle, v, w = torch.split(b.view(-1), 1)
        relative_distance = torch.sqrt((self.goalx-px)**2+(self.goaly-py)**2).view(-1)
        relative_angle = torch.atan((self.goaly-py)/(self.goalx-px)).view(-1)-angle
        relative_angle = torch.clamp(relative_angle,-pi,pi)
        vecL = bcov2vec(P)
        decision_info = torch.cat([
            relative_distance, 
            relative_angle, 
            v, 
            w,
            self.episode_time, 
            prevv,
            prevw,
            vecL, 
            task_param.view(-1)])
        return decision_info.view(1, -1)


class Firefly1dsession(FireflyTrue1d): 
    
    def reset(self,
                pro_gains = None, 
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                phi=None,
                theta=None,
                goal_position=None,
                obs_traj=None,
                pro_traj=None,
                new_session=True,
                initv=None
                ): 
        if new_session:
            print('new sess')
            self.session_trials=0
            self.session_reward_trials=0
            if phi is not None:
                self.phi=phi
            if theta is not None:
                self.theta=theta
            if phi is None and theta is None:
                self.phi=self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_noise_stds=obs_noise_stds)
                self.theta=self.phi
            self.session_timer=0
        self.iti_timer=0
        self.episode_timer = torch.zeros(1)  # int
        self.reset_state(goal_position=goal_position,initv=initv)
        self.reset_belief()
        self.reset_obs()
        self.reset_decision_info()
        self.actions=[]
        self.obs_traj=obs_traj
        self.pro_traj=pro_traj
        self.trial_sum_cost=torch.zeros(1)
        self.trial_dev_costs=[]
        self.trial_mag_costs=[]
        self.trial_actions=[]
        self.A=self.transition_matrix()
        self.Q = torch.zeros(2,2)
        self.Q[1,1]=(self.theta[0]*self.theta[1])**2
        self.R = self.theta[2]**2
        self.A = self.transition_matrix() 
        if self.R < 1e-8:
            self.R=self.R+1e-8
        return self.decision_info.view(1,-1)

    def step(self, action, onetrial=False):
        action=torch.tensor(action).reshape(1)
        self.a=action
        self.episode_timer+=1
        self.session_timer+=1
        end_current_ep=(action<self.terminal_vel or self.episode_timer>=self.episode_len)
        end_session=(self.session_timer>=self.session_len)
        self.episode_reward, cost,mag,dev=self.caculate_reward()
        self.previous_action=action
        self.trial_sum_cost+=cost
        # print('reward-cost, ',self.episode_reward-self.trial_sum_cost)
        if self.debug:
            self.trial_actions.append(action)
            self.trial_dev_costs.append(dev)
            self.trial_mag_costs.append(mag)
            if onetrial:
                return self.decision_info, 0., end_current_ep, {}
            self.reset(new_session=False)
            return self.decision_info, float(reward_rate), end_session, {}
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_timer,task_param=self.theta)
        if end_current_ep:
        #     if self.iti_timer!=0:
        #         self.episode_reward=0.
        #     self.iti_timer+=1
        #     # print('iti,', self.iti_timer)
        # if self.iti_timer>=5:
            self.reset(new_session=False)
            self.session_trials+=1
        if self.episode_reward != 0:
            self.session_reward_trials+=1        
        if end_session:
            if self.session_reward_trials/self.session_trials >1.0:
                raise RuntimeError 
            print('reward percent,  ', self.session_reward_trials/self.session_trials)
        return self.decision_info, self.episode_reward-cost, end_session, {}


class FireFlyReady(gym.Env, torch.nn.Module): 

    def __init__(self,arg=None,kwargs=None):
        self.arg=arg
        self.min_distance = arg.goal_distance_range[0]
        self.max_distance = arg.goal_distance_range[1] 
        self.min_angle = -pi/4
        self.max_angle = pi/4
        self.terminal_vel = arg.terminal_vel 
        self.episode_len = arg.episode_len
        self.dt = arg.dt
        self.reward=arg.reward_amount
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Box(low=-1., high=1.,shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,27),dtype=np.float32)
        self.cost_function = self.action_cost_wrapper
        self.cost_scale = arg.cost_scale
        self.presist_phi=                   arg.presist_phi
        self.agent_knows_phi=               arg.agent_knows_phi
        self.phi=                           None
        self.goal_radius_range =            arg.goal_radius_range
        self.gains_range=                   arg.gains_range
        self.std_range =                    arg.std_range
        self.mag_action_cost_range =        arg.mag_action_cost_range
        self.dev_action_cost_range =        arg.dev_action_cost_range
        self.dev_v_cost_range=              arg.dev_v_cost_range
        self.dev_w_cost_range=              arg.dev_w_cost_range
        self.init_uncertainty_range =       arg.init_uncertainty_range
        self.previous_v_range =             [0,1]
        self.trial_counter=                 0
        self.recent_rewarded_trials=        0
        self.recent_skipped_trials=         0
        self.recent_trials=                 0

    def reset_state(self,goal_position=None,initv=None,initw=None):
        if goal_position is not None:
            self.goalx = goal_position[0]*torch.tensor(1.0)
            self.goaly = goal_position[1]*torch.tensor(1.0)
        else:
            distance = torch.zeros(1).uniform_(self.min_distance, self.max_distance)  
            angle = torch.zeros(1).uniform_(self.min_angle, self.max_angle) 
            self.goalx = torch.cos(angle)*distance
            self.goaly = torch.sin(angle)*distance
        if initv is not None:
            vctrl=initv
        else:
            vctrl = torch.zeros(1)#.uniform_(self.previous_v_range[0], self.previous_v_range[1]) 
        if initw is not None:
            wctrl=initw
        else:
            wctrl = torch.zeros(1)#torch.distributions.Normal(0,self.previous_v_range[1]/2).sample() 
        self.previous_action=torch.tensor([[vctrl,wctrl]])
        self.s = torch.tensor(
        [[torch.distributions.Normal(0,torch.ones(1)).sample()*0.05*self.theta[10]],
        [torch.distributions.Normal(0,torch.ones(1)).sample()*0.05*self.theta[11]],
        [0.],
        [vctrl*self.phi[0]],
        [wctrl*self.phi[1]]])

    def reset_task_param(self,                
                pro_gains = None,
                pro_noise_stds = None,
                obs_noise_stds = None,
                goal_radius=None,
                mag_action_cost_factor=None,
                dev_v_cost_factor=None,
                dev_w_cost_factor=None,
                inital_x_std=None,
                inital_y_std=None,
                ):
        _prov_gains = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  
        _prow_gains = torch.zeros(1).uniform_(self.gains_range[2], self.gains_range[3])  
        _prov_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _prow_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _obsv_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        _obsw_noise_stds=torch.zeros(1).uniform_(self.std_range[2], self.std_range[3])
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])
        _mag_action_cost_factor = torch.zeros(1).uniform_(self.mag_action_cost_range[0], self.mag_action_cost_range[1]) 
        _dev_v_cost_factor = torch.zeros(1).uniform_(self.dev_v_cost_range[0], self.dev_v_cost_range[1])
        _dev_w_cost_factor = torch.zeros(1).uniform_(self.dev_w_cost_range[0], self.dev_w_cost_range[1])
        _inital_x_std = torch.zeros(1).uniform_(self.init_uncertainty_range[0], self.init_uncertainty_range[1])
        _inital_y_std = torch.zeros(1).uniform_(self.init_uncertainty_range[0], self.init_uncertainty_range[1])
        phi=torch.cat([_prov_gains,_prow_gains,_prov_noise_stds,_prow_noise_stds,
            _obsv_noise_stds,_obsw_noise_stds,
                _goal_radius,_mag_action_cost_factor,_dev_v_cost_factor,_dev_w_cost_factor,
                _inital_x_std,_inital_y_std,
                ])
        phi[0]=pro_gains[0] if pro_gains is not None else phi[0]
        phi[1]=pro_gains[1] if pro_gains is not None else phi[1]
        phi[2]=pro_noise_stds[0] if pro_noise_stds is not None else phi[2]
        phi[3]=pro_noise_stds[1] if pro_noise_stds is not None else phi[3]
        phi[4]=obs_noise_stds[0] if obs_noise_stds is not None else phi[4]
        phi[5]=obs_noise_stds[1] if obs_noise_stds is not None else phi[5]
        phi[6]=goal_radius if goal_radius is not None else phi[6]
        phi[7]=mag_action_cost_factor if mag_action_cost_factor is not None else phi[7]
        phi[8]=dev_v_cost_factor if dev_v_cost_factor is not None else phi[8]
        phi[9]=dev_w_cost_factor if dev_w_cost_factor is not None else phi[9]
        phi[10]=inital_x_std if inital_x_std is not None else phi[10]
        phi[11]=inital_y_std if inital_y_std is not None else phi[11]
        return phi

    def step(self, action):
        action=torch.tensor(action).reshape(1,-1)
        self.a=action
        self.episode_time=self.episode_time+1
        self.sys_vel=torch.norm(action)
        if not self.if_agent_stop(sys_vel=self.sys_vel) and not self.agent_start:
            self.agent_start=True
        if self.agent_start:
            self.stop=True if self.if_agent_stop(sys_vel=self.sys_vel) else False 
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
        # dynamic
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b, self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, a=action, time=self.episode_time,task_param=self.theta)
        # eval
        self.episode_reward, cost, mag, dev = self.caculate_reward()
        self.trial_sum_cost += cost
        self.trial_mag += mag
        self.trial_mag_costs.append(mag)
        self.trial_dev += dev
        self.trial_dev_costs.append(dev)
        if end_current_ep:
            if self.stop and self.rewarded(): # eval based on belief.
                self.recent_rewarded_trials+=1
            if self.skipped():
                self.recent_skipped_trials+=1
            self.recent_trials+=1
            self.trial_counter+=1
            # print(self.trial_counter)
            if self.trial_counter!=0 and self.trial_counter%10==0:
                print('rewarded: ',self.recent_rewarded_trials, 'skipped: ', self.recent_skipped_trials, ' out of total: ', self.recent_trials)
                self.recent_rewarded_trials=0
                self.recent_skipped_trials=0
                self.recent_trials=0
            # print('distance, ', self.get_distance()[1], 'goal', self.goal_r,'sysvel',self.sys_vel)
            # print('reward: {}, cost: {}, mag{}, dev{}'.format(self.episode_reward-self.trial_sum_cost, self.trial_sum_cost, self.trial_mag, self.trial_dev))
            # return self.decision_info, self.episode_reward-self.trial_sum_cost, end_current_ep, {}
        self.previous_action=action
        return self.decision_info, self.episode_reward-cost, end_current_ep, {}

    def update_state(self, state):
        px, py, angle, v, w = torch.split(state.view(-1), 1)
        if v==0:
            pass
        elif w==0:
            px = px + v*self.dt * torch.cos(angle)
            py = py + v*self.dt * torch.sin(angle)
        else:
            px = px-torch.sin(angle)*(v/w-(v*torch.cos(w*self.dt)/w))+torch.cos(angle)*((v*torch.sin(w*self.dt)/w))
            py = py+torch.cos(angle)*(v/w-(v*torch.cos(w*self.dt)/w))+torch.sin(angle)*((v*torch.sin(w*self.dt)/w))
        angle = angle + w* self.dt
        px = torch.clamp(px, -self.max_distance, self.max_distance) 
        py = torch.clamp(py, -self.max_distance, self.max_distance) 
        next_s = torch.stack((px, py, angle, v, w ))
        return next_s.view(-1,1)

    def wrap_decision_info(self,b=None,P=None, a=None, time=None, task_param=None):
        task_param=self.theta if task_param is None else task_param
        b=self.b if b is None else b
        P=self.P if P is None else P
        prevv, prevw = torch.split(a.view(-1), 1)
        px, py, angle, v, w = torch.split(b.view(-1), 1)
        relative_distance = torch.sqrt((self.goalx-px)**2+(self.goaly-py)**2).view(-1)
        relative_angle = torch.atan((self.goaly-py)/(self.goalx-px)).view(-1)-angle
        relative_angle = torch.clamp(relative_angle,-pi,pi)
        vecL = bcov2vec(P)
        decision_info = torch.cat([
            relative_distance, 
            relative_angle, 
            v, 
            w,
            self.episode_time, 
            prevv,
            prevw,
            vecL, 
            task_param.view(-1)])
        return decision_info.view(1, -1)

    def action_cost_wrapper(self,action, previous_action):
        mag_cost=self.action_cost_magnitude(action)
        dev_cost=self.action_cost_dev(action, previous_action)
        total_cost=mag_cost+dev_cost
        return total_cost, mag_cost, dev_cost

    def caculate_reward(self):
        if self.stop:
            _,d= self.get_distance(state=self.b)
            if d<=self.goal_r:
                reward=torch.tensor([1.])*self.reward
            else:
                if self.goal_r/d>1/3:
                    reward = (self.goal_r/d)**1*self.reward*0.8
                else: reward=torch.tensor([0.])
                if reward>self.reward:
                    raise RuntimeError
                if reward<0:
                    raise RuntimeError 
                    # reward=torch.tensor([0.])
        else:
            # _,d= self.get_distance()
            # reward=(self.goal_r/d)**2
            reward=torch.tensor([0.])
        cost, mag, dev=self.cost_function(self.a, self.previous_action)
        # print('reward',reward, 'dev',dev, 'actions',self.previous_action,self.a)
        reward=reward
        return reward, self.cost_scale*cost, mag, dev

    def reset(self,
                pro_gains = None, 
                pro_noise_stds = None,
                obs_noise_stds = None,
                phi=None,
                theta=None,
                goal_position=None,
                initv=None,
                initw=None,
                obs_traj=None,
                pro_traj=None,
                ): 
        if phi is not None:
            self.phi=phi
        elif self.presist_phi and self.phi is not None:
            pass
        else: # when either not presist, or no phi.
            self.phi=self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_noise_stds=obs_noise_stds)
        if theta is not None:
            self.theta=theta
        else:
            self.theta=self.phi if self.agent_knows_phi else self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_noise_stds=obs_noise_stds)
        self.unpack_theta()
        self.obs_traj=obs_traj
        self.pro_traj=pro_traj
        self.sys_vel=torch.zeros(1)
        self.stop=False 
        self.agent_start=False 
        self.episode_time = torch.zeros(1)
        self.previous_action=torch.tensor([[0.,0.]])
        self.trial_sum_cost=0
        self.trial_mag=0
        self.trial_mag_costs=[]
        self.trial_dev_costs=[]
        self.trial_dev=0
        self.reset_state(goal_position=goal_position,initv=initv,initw=initw)
        self.reset_belief()
        self.reset_obs()
        self.decision_info=self.wrap_decision_info(b=self.b, a=self.previous_action, time=self.episode_time, task_param=self.theta)
        return self.decision_info.view(1,-1)

    def unpack_theta(self):
        self.pro_gainv=self.phi[0]
        self.pro_gainw=self.phi[1]
        self.pro_noisev=self.phi[2]
        self.pro_noisew=self.phi[3]
        self.goal_r=self.phi[6]

        self.pro_gainv_hat=self.theta[0]
        self.pro_gainw_hat=self.theta[1]
        self.pro_noisev_hat=self.theta[2]
        self.pro_noisew_hat=self.theta[3]
        self.obs_noisev=self.theta[4]
        self.obs_noisew=self.theta[5]
        self.goal_r_hat=self.theta[6]
        self.Q = torch.zeros(5,5)
        self.Q[3,3]=self.pro_noisev_hat**2 if self.pro_noisev_hat**2 > 1e-4 else self.pro_noisev_hat**2+1e-4
        self.Q[4,4]=self.pro_noisew_hat**2 if self.pro_noisew_hat**2 > 1e-4 else self.pro_noisew_hat**2+1e-4
        self.R = torch.zeros(2,2)
        self.R[0,0] = self.obs_noisev**2 if self.obs_noisev**2 > 1e-4 else self.obs_noisev**2+1e-4
        self.R[1,1] = self.obs_noisew**2 if self.obs_noisew**2 > 1e-4 else self.obs_noisew**2+1e-4

    def reset_belief(self): 
        self.P = torch.eye(5) * 1e-8 
        self.P[0,0]=(self.theta[10]*0.05)**2 # sigma xx
        self.P[1,1]=(self.theta[11]*0.05)**2 # sigma yy
        self.b = self.s

    def reset_obs(self):
        self.o=torch.tensor([[0],[0]])

    def if_agent_stop(self,sys_vel=None):
        stop=(sys_vel <= self.terminal_vel)
        return stop

    def forward(self, action,task_param,state=None, giving_reward=None):
        action=torch.tensor(action).reshape(1,-1)
        self.a=action
        self.episode_time=self.episode_time+1
        self.sys_vel=torch.norm(action)
        if not self.if_agent_stop(sys_vel=self.sys_vel) and not self.agent_start:
            self.agent_start=True
        if self.agent_start:
            self.stop=True if self.if_agent_stop(sys_vel=self.sys_vel) else False 
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
        # dynamic
        if state is None:
            self.s=self.state_step(action,self.s)
        else:
            self.s=state
            self.previous_action=action
        self.o=self.observations(self.s)
        self.o_mu=self.observations_mean(self.s)
        self.b, self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        return self.decision_info, end_current_ep

    def state_step(self,a, s):
        next_s = self.update_state(s)
        px, py, angle, v, w = torch.split(next_s.view(-1), 1)
        if self.pro_traj is None:
            noisev=torch.distributions.Normal(0,torch.ones(1)).sample()*self.pro_noisev  
            noisew=torch.distributions.Normal(0,torch.ones(1)).sample()*self.pro_noisew
        else:
            noisev=self.pro_traj[int(self.episode_time.item())]#*self.pro_noise  
            noisew=self.pro_traj[int(self.episode_time.item())]#*self.pro_noise  
        v = self.pro_gainv * torch.tensor(1.0)*a[0,0] + noisev
        w = self.pro_gainw * torch.tensor(1.0)*a[0,1] + noisew
        next_s = torch.cat((px, py, angle, v, w)).view(1,-1)
        return next_s.view(-1,1)

    def apply_action(self,state,action):
        px, py, angle, v, w = torch.split(state.view(-1), 1)
        v=action[0,0]*self.pro_gainv_hat*torch.ones(1)
        w=action[0,1]*self.pro_gainw_hat*torch.ones(1)
        next_s = torch.stack((px, py, angle, v, w))
        return next_s.view(-1,1)

    def belief_step(self, previous_b,previous_P, o, a, task_param=None):
        task_param = self.theta if task_param is None else task_param
        I = torch.eye(5)
        H = torch.tensor([[0.,0.,0.,1,0.],[0.,0.,0.,0.,1]])
        self.A = self.transition_matrix(previous_b, a) 
        # prediction
        predicted_b = self.update_state(previous_b)
        predicted_b = self.apply_action(predicted_b,a)
        predicted_P = self.A@(previous_P)@(self.A.t())+self.Q 
        if not is_pos_def(predicted_P):
            print('predicted not pos def')
            print('action ,', a)
            print('theta: ', task_param)
            print("predicted_P:", predicted_P)
            print('Q:',self.Q)
            print("previous_P:", previous_P)
            print("A:", self.A)
            APA = self.A@(previous_P)@(self.A.t())
            print("APA:", APA)
            print("APA +:", is_pos_def(APA))

        error = o - H@predicted_b
        S = H@(predicted_P)@(H.t()) + self.R 
        K = predicted_P@(H.t())@(torch.inverse(S)) 
        
        b = predicted_b + K@(error)
        I_KH = I - K@(H)
        P = I_KH@(predicted_P)

        if not is_pos_def(P): 
            print("after update not pos def")
            print("updated P:", P)
            print('Q : ', self.Q)
            print('R : ', self.R)
            # print("K:", K)
            # print("H:", H)
            print("I - KH : ", I_KH)
            print("error : ", error)
            print('task parameter : ',task_param)
            P = (P + P.t()) / 2 + 1e-6 * I  # make symmetric to avoid computational overflows
        
        return b, P

    def transition_matrix(self,b, a): 
        angle=b[2,0]
        vel=b[3,0]
        A = torch.zeros(5, 5)
        A[:3, :3] = torch.eye(3)
        # partial dev with theta
        A[0, 2] = - vel * torch.sin(angle) * self.dt*self.pro_gainv_hat
        A[1, 2] = vel * torch.cos(angle) * self.dt*self.pro_gainv_hat
        # partial dev with v
        A[0, 3] =   torch.cos(angle) * self.dt*self.pro_gainv_hat
        A[1, 3] =  torch.sin(angle) * self.dt*self.pro_gainv_hat
        # partial dev with w
        A[2, 4] =  self.dt*self.pro_gainw_hat
        return A

    def reached_goal(self):
        # use real location
        _,distance=self.get_distance(state=self.s)
        reached_bool= (distance<=self.goal_r)
        return reached_bool

    def observations(self, state): 
        s=state.clone()
        if self.obs_traj is None:
            noisev = torch.distributions.Normal(0,torch.tensor(1.)).sample()*self.obs_noisev
            noisew = torch.distributions.Normal(0,torch.tensor(1.)).sample()*self.obs_noisew
        else:
            noise = self.obs_traj[int(self.episode_time.item())]#*self.obs_noise
        vw = s.view(-1)[-2:] # 1,5 to vector and take last two
        vw[0] =  vw[0] + noisev
        vw[1] =  vw[1] + noisew
        return vw.view(-1,1)

    def observations_mean(self, s):
        vw = s.view(-1)[-2:]
        return vw.view(-1,1)

    def get_distance(self, state=None, distance2goal=True): 
        state=self.s if state is None else state
        px, py, angle, v, w = torch.split(state.view(-1), 1)
        position = torch.stack([px,py])
        if distance2goal:
            relative_distance = torch.sqrt((self.goalx-px)**2+(self.goaly-py)**2).view(-1)
        else:
            relative_distance = torch.sqrt((px)**2+(py)**2).view(-1)
        return position, relative_distance
    
    def rewarded(self):
        return (self.get_distance(state=self.b)[1]<=self.goal_r)

    def skipped(self):
        return (self.get_distance(state=self.b,distance2goal=False)[1]<=self.goal_r)
        
    def action_cost_magnitude(self,action):
        # cost=torch.norm(action)
        # # scalar=self.reward/(1/0.4/self.dt)
        # num_steps=(1/0.4/self.dt)*2 #num of steps using half control
        # scalar=self.reward/num_steps*4 # the cost per dt to have 0 reward, using half ctrl
        # return scalar*cost*self.theta[7]
        return 0.*self.theta[7]

    def action_cost_dev(self, action, previous_action):
        # action is row vector
        action[0,0]=1.0 if action[0,0]>1.0 else action[0,0]
        action[0,1]=1.0 if action[0,1]>1.0 else action[0,1]
        action[0,1]=-1.0 if action[0,1]<-1.0 else action[0,1]
        vcost=(action[0,0]-previous_action[0,0])**2*self.theta[8]
        wcost=(action[0,1]-previous_action[0,1])**2*self.theta[9]
        cost=vcost+wcost
        mincost=1/20/20*30 #1/20^2 min cost per dt, for 30 dts.
        # mincost=2
        scalar=self.reward/mincost
        cost=cost*scalar
        if abs(action[0,0]-previous_action[0,0])<0.1 and abs(action[0,1]-previous_action[0,1])<0.1:
            cost=cost*0
        return cost
import gym
from gym import spaces

from numpy import pi
import numpy as np

from FireflyEnv.env_utils import *

from FireflyEnv.ffenv_new_cord import FireflyAgentCenter
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


class FireflyAcc(FireflyAgentCenter): 

    def __init__(self,arg=None,kwargs=None):
        '''
        state-observation-blief-action-next state
        arg:
            gains_range, 4 element list

        step:  action -> new state-observation-blief    
        reset: init new state
        '''
        super(FireflyAgentCenter, self).__init__(arg=arg,kwargs=kwargs)
        
        # we have modified observation space ( new acc related parameters)
        low=-np.inf
        high=np.inf
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,30),dtype=np.float32)
        self.cost_function=reward_singleff.action_cost_wrapper
        

    # new parameters added. overriding the original functions
    def _apply_param_range(self,gains_range=None,std_range=None,goal_radius_range=None,tau_range=None):

        if goal_radius_range is None:
            self.goal_radius_range =     self.arg.goal_radius_range
        if gains_range is None:
            self.gains_range=            self.arg.gains_range
        if std_range is None:
            self.std_range =             self.arg.std_range
        if tau_range is None:
            self.tau_range = compute_tau_range(self.gains_range, d=2*self.world_size, T=self.episode_len*self.dt, tau=1.8)
            self.tau_range[1]=min(8,self.tau_range[1])
        else:
            self.tau_range=tau_range
            print('tau range', tau_range)
    

    # new parameters added. overriding the original functions
    def reset_task_param(self,                
                pro_gains = None, 
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                goal_radius=None,
                tau=None,
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
        
        _tau = torch.zeros(1).uniform_(self.tau_range[0], self.tau_range[1])




        # # calculate the approate a from vm. then calculate b from a
        # a=get_a(_pro_gains[0],dt=self.dt, T=self.episode_len, target_d=self.world_size*3,control_gain=_pro_gains[0])
        # _acc_a = torch.zeros(1).uniform_(self.acc_a_range[0], a)
        # _acc_b = _pro_gains[0]*(1-_acc_a)
        # # ignoring the w for now, since task is limited by v


        phi=torch.cat([_pro_gains,_pro_noise_stds,_obs_gains,_obs_noise_stds,_goal_radius,_tau])

        phi[0:2]=pro_gains if pro_gains is not None else phi[0:2]
        phi[2:4]=pro_noise_stds if pro_noise_stds is not None else phi[2:4]
        
        phi[4:6]=obs_gains if obs_gains is not None else phi[4:6]
        phi[6:8]=obs_noise_stds if obs_noise_stds is not None else phi[6:8]
        
        phi[8]=goal_radius if goal_radius is not None else phi[8]
        
        phi[9]=tau if tau is not None else phi[9]

        return col_vector(phi)


    # add agent start
    def reset_state(self,goal_position=None):

        min_distance =      self.phi[8,0].item()+self.world_size*0.0 # min d to the goal is 0.2
        distance =          torch.zeros(1).uniform_(min_distance, max(min_distance,min(self.world_size, self.max_distance))) # max d to the goal is world boundary.
        ang =               torch.zeros(1).uniform_(-pi/4, pi/4) # regard less of the task
        self.goalx =        distance * torch.cos(ang)
        self.goaly =        distance * torch.sin(ang)
        vel = torch.zeros(1)
        ang_vel = torch.zeros(1)
        heading=torch.zeros(1) # always facing 0 angle

        if goal_position is not None:
            self.goalx = goal_position[0]*torch.ones(1)
            self.goaly = goal_position[1]*torch.ones(1)

        else:
            pass
        self.s = torch.cat([torch.zeros(1), torch.zeros(1), heading, vel, ang_vel]) # this is state x at t0
        self.s = self.s.view(-1,1) # column vector
        self.agent_start=False
        # print(self.phi.view(-1))


    def step(self, action):
        
        if not self.if_agent_stop() and not self.agent_start:
            self.agent_start=True

        if self.agent_start:
            self.stop=True if self.if_agent_stop() else False 
        else:
            self.stop=False

        self.a=action
        self.episode_reward=self.caculate_reward()
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        self.episode_time=self.episode_time+1
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)

        return self.decision_info, self.episode_reward, end_current_ep, {}

    def forward(self, action,task_param,state=None, giving_reward=None):

        if not self.if_agent_stop() and not self.agent_start:
            self.agent_start=True

        if self.agent_start:
            self.stop=True if self.if_agent_stop() else False 
        else:
            self.stop=False

        self.s=self.state_step(action, self.s) # state transition defined by phi.
        self.o=self.observations(self.s,task_param=task_param) # observation defined by theta.
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action,task_param=task_param)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=task_param)
        self.episode_time=self.episode_time+1
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
             
        return self.decision_info, end_current_ep


    # acc control state dynamics
    def a_(self, tau, dt=None):
        dt=self.dt if dt is None else dt
        return torch.exp(-dt/tau)
    
    def vm_(self, tau, x=None, T=None):
        x=1.5*self.world_size if x is None else x
        T=self.episode_len*self.dt if T is None else T
        vm=x/2/tau *(1/(torch.log(torch.cosh(T/2/tau))) )
        if vm==0: # too much velocity control rather than acc control
            vm=x/T
        return vm

    def b_(self, tau, dt=None):
        dt=self.dt if dt is None else dt
        return self.vm_(tau)*(1-self.a_(tau))

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
        vel = self.a_(task_param[9,0]) * vel + self.b_(task_param[9,0])*(task_param[0,0] * (a_v + w[0]))
        ang_vel = self.a_(task_param[9,0]) * ang_vel + self.b_(task_param[9,0])*(task_param[1,0] * (a_w + w[1]))

        next_s = torch.stack((px, py, heading, vel, ang_vel)).view(1,-1)
        
        return next_s.view(-1,1)


    def apply_action(self,state,action,task_param=None):

        '''
        B*action_t part in state_t+1 = A*state_t + B*action_t

        input:
            part of state_t+1, A*state_t
            action
            param, optinal, default is phi
        output:
            final state_t+1
        '''

        # with no noise, used in belief prediction
        task_param=self.phi if task_param is None else task_param

        px, py, heading, vel, ang_vel = torch.split(state.view(-1), 1)
        vel=vel*self.a_(task_param[9,0])        +action[0]*task_param[0,0]*self.b_(task_param[9,0])
        ang_vel=ang_vel*self.a_(task_param[9,0])  +action[1]*task_param[1,0]*self.b_(task_param[9,0])
        next_s = torch.stack((px, py, heading, vel, ang_vel))

        return next_s.view(-1,1)


    def if_agent_stop(self,action=None,state=None,task_param=None):
        terminal=False
        terminal_vel = self.terminal_vel
        task_param=self.phi if task_param is None else task_param
        state=self.s if state is None else state
        px, py, heading, vel, ang_vel = torch.split(state.view(-1), 1)
        # vel=vel*self.a_(task_param[9,0])           +action[0]*task_param[0,0]*self.b_(task_param[9,0])
        # ang_vel=ang_vel*self.a_(task_param[9,0])   +action[1]*task_param[1,0]*self.b_(task_param[9,0])
        stop = (vel <= terminal_vel)
        if stop:
            terminal= True

        return terminal

    def caculate_reward(self):
        '''
        calculate the reward, when agent stop (made the decision)

        input:
        output:
            reward as float, or tensor?
        '''

        if self.stop:
            if self.reached_goal(): # integration reward
                return self.reward_function(self.stop, self.reached_goal(),
                self.b,self.P,self.phi[8,0],self.reward,
                self.goalx,self.goaly,time=self.episode_time/5)
            else: # gaussian like reward to attract to goal
                _,d= self.get_distance()
                # print(d)
                return max(0, (1-d)) 
        else: # punish the backward move
            neg_velocity=self.s[3] if self.s[3]<0 else 0
            return neg_velocity



    def transition_matrix(self, action, state, task_param=None): 

        '''
        create the transition matrix,
        used in kalman filter for A*Pt*AT+Q to predict Pt+1

        input:
            action,
            state,
            param, default is theta
            (all at time t)
        output:
            transition matrix A

        '''
        task_param = self.theta if task_param is None else task_param

        px, py, ang, vel, ang_vel = torch.split(state.view(-1),1)
        
        A = torch.zeros(5, 5)
        A[:3,:3] = torch.eye(3)
        # partial dev with v
        A[0, 3] = torch.cos(ang) * self.dt*task_param[0,0]
        A[1, 3] = torch.sin(ang) * self.dt*task_param[0,0]
        A[3,3] = self.a_(task_param[9,0])
        # partial dev with w
        A[2, 4] =  self.dt*task_param[1,0]
        A[4,4] = self.a_(task_param[9,0])
        # partial dev with theta
        A[0, 2] =  - torch.sin(ang) *vel * self.dt*task_param[0,0]
        A[1, 2] =  torch.cos(ang) *vel * self.dt*task_param[0,0]


        return A


    def belief_step(self, previous_b,previous_P, o, a, task_param=None):
        
        '''
        changes from base (new cord):
        Q matrix, now b squre * std squares
        '''

        task_param = self.theta if task_param is None else task_param

        I = torch.eye(5)

        # Q matrix, process noise for tranform xt to xt+1. only applied to v and w
        Q = torch.zeros(5, 5)
        Q[-2:, -2:] = torch.diag((task_param[2:4,0]*self.b_(task_param[9,0]))**2) # variance of vel, ang_vel
        
        # R matrix, observe noise for observation
        R = torch.diag((task_param[4:6,0]*task_param[6:8,0]*self.s[3:,0])** 2)

        # H matrix, transform x into observe space. only applied to v and w.
        H = torch.zeros(2, 5)
        H[:, -2:] = torch.diag(task_param[4:6,0])

        # prediction
        # Ax_t part. using previously corrected v w in bt
        predicted_b = self.update_state(previous_b,task_param=task_param)
        # Bu part. assign action to state (estimation)
        predicted_b = self.apply_action(predicted_b,a,task_param=task_param)

        A = self.transition_matrix(a,previous_b,task_param) 
        predicted_P = A@(previous_P)@(A.t())+Q # estimate Pt+1 = APA^T+Q, 

        # debug check
        if not is_pos_def(predicted_P): # should be positive definate. if not, show debug. 
            # if noise go to 0, happens.
            print("predicted_P:", predicted_P)
            print('Q:',Q)
            print("previous_P:", previous_P)
            print("A:", A)
            APA = A@(previous_P)@(A.t())
            print("APA:", APA)
            print("APA +:", is_pos_def(APA))

        error = o - H@predicted_b # error as y=z-hx, the xt+1 is estimated
        S = H@(predicted_P)@(H.t()) + R # S = HPH^T+R. the covarance of observation of xt->xt+1 transition 
        K = predicted_P@(H.t())@(torch.inverse(S)) # K = PHS^-1, kalman gain. has a H in front but canceled out
        
        b = predicted_b + K@(error) # update xt+1 from the estimated xt+1 using new observation zt
        I_KH = I - K@(H)
        P = I_KH@(predicted_P)# update covarance of xt+1 from the estimated xt+1 using new observation zt noise R


        # debug check
        if not is_pos_def(P): 
            print("here")
            print("P:", P)
            P = (P + P.t()) / 2 + 1e-6 * I  # make symmetric to avoid computational overflows

        # b=self.update_state(b)
        return b, P


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
        on = torch.distributions.Normal(0,torch.ones([2,1])).sample()*task_param[6:8] # on is observation noise
        vel, ang_vel = torch.split(s.view(-1),1)[-2:] # 1,5 to vector and take last two

        ovel = task_param[4,0] * vel *(1+on[0]) # observe velocity
        oang_vel = task_param[5,0]* ang_vel *(1+on[1])
        observation = torch.stack((ovel, oang_vel)) # observed x
        return observation.view(-1,1)
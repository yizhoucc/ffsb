import gym
from gym import spaces

from numpy import pi
import numpy as np

from FireflyEnv.env_utils import *

from FireflyEnv.firefly_base import FireflyEnvBase

# change log
# obs space is inf
# representation



class FireflyAgentCenter(FireflyEnvBase): 

# resets

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


    def reset_belief(self): 

        self.P = torch.eye(5) * 1e-8 
        self.b = self.s.view(-1,1) # column vector


    def reset_obs(self):
        # obs at t0 on v and w, both 0
        self.o=torch.zeros(2).view(-1,1)       # column vector


    def reset_decision_info(self):
        self.episode_time = 0   # int
        self.stop=False         # bool
        self.decision_info = self.wrap_decision_info(b=self.b, time=self.episode_time, task_param=self.theta)
        self.decision_info = row_vector(self.decision_info)

# steps

    def forward(self, action,task_param,state=None, giving_reward=None):

        '''
        this function is used in inverse.
        either to generate trajectories of teacher, using the true theta,
        or, to generate trajectories of agent, using estimation of theta.

        simplified version of step,
        we dont need to compute reward, as we only care about trajectories.
        just need to use state, belief, p, t, param, action
        and compute state, belief, p, t, decision info for next step.

        input: 
            action,
            theta, the estimation of theta or true theta. (true theta is self.phi)
            optional:
                state, start from a certain state. not used
                giving reward, bool, if returning reward, for a selection of good trails.
                
        output:

        altered class variables,
            stop
            states,
            belief mean,
            P the cov,
            time,
            decision info
        '''
    
        self.stop=True if self.if_agent_stop(action) else False 

        self.s=self.state_step(action, self.s) # state transition defined by phi.

        self.o=self.observations(self.s,task_param=task_param) # observation defined by theta.

        self.b,self.P=self.belief_step(self.b,self.P,self.o,action,task_param=task_param)

        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=task_param)
            
        self.episode_time=self.episode_time+1
        
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
             
        return self.decision_info, end_current_ep


    def step(self, action):

        '''
        state dynamic. used in stabe baselines
        
        input:
            action.
        output:
            decision info, reward, done, debug dict

        altered class var:
            stop,
            state,
            observation,
            belief mean,
            belief cov, P
            reward,
            decision info,
            time,

        '''

        self.stop=True if self.if_agent_stop(action) else False 
        self.a=action
        self.episode_reward=self.caculate_reward()
        
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action)

        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        # self.ep_count+=1

        # if self.verbose:
        #     _,d=self.get_distance()
        #     if self.stop and d<0.3:
                # print('stop',self.stop,'goal',self.reached_goal(),d,
                # self.episode_reward,self.episode_time,self.decision_info)
            
        self.episode_time=self.episode_time+1
        
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)

        return self.decision_info, self.episode_reward, end_current_ep, {}


    def state_step(self,a, s, task_param=None):

        '''
        state dynamics

        input:
            action
            param, optional, default is self.phi
        
        output:
            state next
        '''
        
        # STEP 1: use the v,w to move forward
        task_param=self.phi if task_param is None else task_param
        next_s = self.update_state(s, task_param=task_param)

        # STEP 2: use the action to assign v and w
        px, py, heading, vel, ang_vel = torch.split(next_s.view(-1), 1)
        a_v = a[0]  # action for velocity
        a_w = a[1]  # action for angular velocity
        # sample noise value and apply to v, w together with gains
        w=torch.distributions.Normal(0,task_param[2:4,0]).sample()        
        vel = 0.0 * vel + task_param[0,0] * a_v + w[0] 
        ang_vel = 0.0 * ang_vel + task_param[1] * a_w + w[1]

        next_s = torch.stack((px, py, heading, vel, ang_vel)).view(1,-1)
        
        return next_s.view(-1,1)


    def update_state(self, state,task_param=None):

        '''
        A*state_t part in state_t+1 = A*state_t + B*action_t

        input:
            state_t
            param, defult is phi
        output:
            A*state_t (need to use apply action to add B*action part to be final state_t+1)
        
        '''

        task_param=self.theta if task_param is None else task_param

        px, py, heading, vel, ang_vel = torch.split(state.view(-1), 1)
        
        # first move 
        px = px + vel * torch.cos(heading) * self.dt # new position x and y
        py = py + vel * torch.sin(heading) * self.dt
        px = torch.clamp(px, -self.world_size, self.world_size) # restrict location inside arena, to the edge
        py = torch.clamp(py, -self.world_size, self.world_size)
        # then turn
        heading = heading + ang_vel * self.dt
        heading = range_angle(heading) # adjusts the range of angle from -pi to pi

        next_s = torch.stack((px, py, heading, vel, ang_vel))

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
        vel=vel*0.          +action[0]*task_param[0,0]
        ang_vel=ang_vel*0.  +action[1]*task_param[1,0]
        next_s = torch.stack((px, py, heading, vel, ang_vel))

        return next_s.view(-1,1)


    def belief_step(self, previous_b,previous_P, o, a, task_param=None):
        
        '''
        belief dynamic

        input: 
            belief mean
            belief cov, P
            observation
            action
            param, default is theta
            (all at t)
        ouput:
            belief mean at t+1
            belief cov at t+1
        '''

        task_param = self.theta if task_param is None else task_param

        I = torch.eye(5)

        # Q matrix, process noise for tranform xt to xt+1. only applied to v and w
        Q = torch.zeros(5, 5)
        Q[-2:, -2:] = torch.diag((task_param[2:4,0])**2) # variance of vel, ang_vel
        
        # R matrix, observe noise for observation
        R = torch.diag((task_param[6:8,0])** 2)

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


    def wrap_decision_info(self,b=None,P=None,time=None, task_param=None):

        '''
        input:
            belief mean,
            belief cov, P
            time,
            task_param, default is theta

        output:
            decision info.
        '''
        task_param=self.theta if task_param is None else task_param

        b=self.b if b is None else b
        P=self.P if P is None else P
        px, py, heading, vel, ang_vel = torch.split(b.view(-1), 1) # unpack state x
        r = torch.norm(torch.cat([self.goalx-px, self.goaly-py])).view(-1)
        rel_ang = -heading + torch.atan2(self.goaly-py, self.goalx-px).view(-1) # relative angle from goal to agent.
        rel_ang = range_angle(rel_ang) # resize relative angel into -pi pi range.
        vecL = vectorLowerCholesky(P) # take the lower triangle of P
        decision_info = torch.cat([r, rel_ang, vel, ang_vel, 
                        torch.Tensor([self.episode_time]), 
                        vecL, task_param.view(-1)]) 

        return decision_info.view(1, -1) 
        # only this is a row vector. everything else is col vector


    def caculate_reward(self):
        '''
        calculate the reward, when agent stop (made the decision)

        input:
        output:
            reward as float, or tensor?
        '''

        if self.stop:
            return self.reward_function(self.stop, self.reached_goal(),
                self.b,self.P,self.phi[8,0],self.reward,
                self.goalx,self.goaly,time=self.episode_time)
        else:
            return 0.


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
        A[0, 3] =   torch.cos(ang) * self.dt*task_param[0,0]
        A[1, 3] =  torch.sin(ang) * self.dt*task_param[0,0]
        # partial dev with w
        A[2, 4] =  self.dt*task_param[1,0]
        # partial dev with theta
        A[0, 2] =  - torch.sin(ang) *vel * self.dt*task_param[0,0]
        A[1, 2] =  torch.cos(ang) *vel * self.dt*task_param[0,0]

        return A


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
        on = torch.distributions.Normal(0,task_param[6:8,0]).sample() # on is observation noise
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


    def get_distance(self, state=None): 

        '''
        input:
            state tensor, 
        output:
            goal (x,y) relative to agent (agent centered at 0,0)
            distance to goal
        '''

        state=self.s if state is None else state
        position = - state.view(-1)[:2] + torch.Tensor([self.goalx,self.goaly])
        distance = torch.norm(position).item()
        return position, distance
    
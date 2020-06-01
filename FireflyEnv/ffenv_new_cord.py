import gym
from gym import spaces

from numpy import pi
import numpy as np

import InverseFuncs
from .env_utils import *
from DDPGv2Agent.rewards import * #reward
from FireflyEnv.firefly_base import FireflyEnvBase

# change log
# obs space is inf
# representation



class FireflyAgentCenter(FireflyEnvBase): 


    def reset_state(self,goal_position=None):

        min_distance =     self.phi[8].item()+self.world_size*0.25 # min d to the goal is 0.2
        distance =         torch.zeros(1).uniform_(min_distance, max(min_distance,min(self.world_size, self.max_distance))) # max d to the goal is world boundary.
        ang =               torch.zeros(1).uniform_(-pi/4, pi/4) # regard less of the task
        self.goalx =        distance * torch.cos(ang)
        self.goaly =        distance * torch.sin(ang)

        vel = torch.zeros(1)
        ang_vel = torch.zeros(1)
        heading=torch.zeros(1)

        if goal_position is not None:
            self.s = torch.cat([goal_position[0]*torch.ones(1),goal_position[1]*torch.ones(1), heading, vel, ang_vel])
        else:
            self.s = torch.cat([torch.zeros(1), torch.zeros(1), heading, vel, ang_vel]) # this is state x at t0

    def reset_belief(self):
        self.P = torch.eye(5) * 1e-8 
        self.b = self.s

    def reset_obs(self):
        self.o=torch.zeros(2)       # this is observation o at t0

    def reset_decision_info(self):
        self.episode_time = torch.zeros(1)
        self.stop=False 
        self.decision_info = self.wrap_decision_info(b=self.b, time=self.episode_time, theta=self.theta)

    def step(self, action):

        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action)
        self.reward=self.caculate_reward()
        self.wrap_decision_info(b=self.b, time=self.episode_time,theta=self.theta)
        # self.ep_count+=1
        self.episode_time=self.episode_time+1
        end_current_ep=1 if self.stop or self.episode_time>=self.episode_len else 0

        return self.decision_info, end_current_ep, self.reward, {}

    def state_step(self,a, s,task_param=None):
        
        task_param=self.phi if task_param is None else task_param

        px, py, heading, vel, ang_vel = torch.split(s.view(-1), 1)

        a_v = a[0]  # action for velocity
        a_w = a[1]  # action for angular velocity

        w=torch.distributions.Normal(0,task_param[2:4]).sample()
        
        vel = 0.0 * vel + task_param[0] * a_v + w[0] 
        ang_vel = 0.0 * ang_vel + task_param[1] * a_w + w[1]
        heading = heading + ang_vel * self.dt
        heading = range_angle(heading) # adjusts the range of angle from -pi to pi

        px = px + vel * torch.cos(heading) * self.dt # new position x and y
        py = py + vel * torch.sin(heading) * self.dt
        px = torch.clamp(px, -self.world_size, self.world_size) # restrict location inside arena, to the edge
        py = torch.clamp(py, -self.world_size, self.world_size)
        next_x = torch.stack((px, py, heading, vel, ang_vel))

        self.stop=True if self.if_agent_stop(a) else False 

        return next_x.view(1,-1)

    def belief_step(self, previous_b,previous_P, o, a):

        I = torch.eye(5)

        # Q matrix, process noise for tranform xt to xt+1. only applied to v and w
        Q = torch.zeros(5, 5)
        Q[-2:, -2:] = torch.diag(self.theta[2:4]**2) # variance of vel, ang_vel
        
        # R matrix, observe noise for observation
        R = torch.diag(self.theta[6:8]** 2)

        # H matrix, transform x into observe space. only applied to v and w.
        H = torch.zeros(2, 5)
        H[:, -2:] = torch.diag(self.theta[4:6])

        # prediction
        predicted_b = self.state_step(a,previous_b,task_param=self.theta)
        predicted_b = predicted_b.t() # make a column vector
        A = self.transition_matrix(a,previous_b,self.theta) 
        predicted_P = A@(previous_P)@(A.t())+Q # estimate Pt+1 = APA^T+Q, 

        # debug check
        if not is_pos_def(predicted_P): # should be positive definate. if not, show debug. 
            # if noise go to 0, happens.
            print("predicted_P:", predicted_P)
            print("previous_P:", previous_P)
            print("A:", A)
            APA = A@(P)@(A.t())
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

        b = b.t() #return to a row vector

        return b, P

    def wrap_decision_info(self,b=None,time=None,theta=None):

        px, py, heading, vel, ang_vel = torch.split(self.s.view(-1), 1) # unpack state x
        r = torch.norm(torch.cat([self.goalx-px, self.goaly-py])).view(-1) 
        rel_ang = heading - torch.atan2(self.goaly-py, self.goalx-px).view(-1) # relative angle from goal to agent.
        rel_ang = range_angle(rel_ang) # resize relative angel into -pi pi range.
        vecL = vectorLowerCholesky(self.P) # take the lower triangle of P
        state = torch.cat([r, rel_ang, vel, ang_vel, self.episode_time, vecL, self.theta.view(-1)]) # original

        return state.view(1, -1)

    def caculate_reward(self):
        return self.reward_function(self.stop, self.reached_goal(),self.b,self.P,self.phi[8],self.reward)

    # def policy_input_reshape(self, **kwargs): # reshape the belief, ready for policy
    #     '''
    #     reshape belief for policy
    #     '''
    #     argin={} # b, time, theta
    #     for key, value in kwargs.items():
    #         argin[key]=value

    #     try: 
    #         pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = InverseFuncs.unpack_theta(argin['theta']) # unpack the theta
    #     except KeyError: pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = InverseFuncs.unpack_theta(self.theta)
        
    #     try: 
    #         time=argin['time']
    #     except KeyError: time=self.time
        
    #     try: 
    #         b=argin['b']
    #     except KeyError: b=self.b

        
    #     x, P = b # unpack the belief
    #     px, py, heading, vel, ang_vel = torch.split(x.view(-1), 1) # unpack state x
    #     r = torch.norm(torch.cat([self.goalx-px, self.goaly-py])).view(-1) 
    #     rel_ang = heading - torch.atan2(self.goaly-py, self.goalx-px).view(-1) # relative angle from goal to agent.
    #     rel_ang = range_angle(rel_ang) # resize relative angel into -pi pi range.
    #     vecL = vectorLowerCholesky(P) # take the lower triangle of P
    #     state = torch.cat([r, rel_ang, vel, ang_vel, time, vecL, pro_gains.view(-1)]) # original

    #     self.decision_info=state.view(1, -1)
    #     return state.view(1, -1)
    
    def transition_matrix(self, action, state,task_param): # create the transition matrix
        '''
        used in kalman filter as transformation matrix for xt to xt+1
        '''
        # retrun A matrix, A*Pt-1*AT+Q to predict Pt
        px, py, ang, vel, ang_vel = torch.split(state.view(-1),1)
        
        A = torch.zeros(5, 5)
        A[:3, :3] = torch.eye(3)
        A[0, 3] = -  torch.cos(ang+action[1]*task_param[0]*self.dt) * self.dt
        A[1, 3] = - torch.sin(ang+action[1]*task_param[1]*self.dt) * self.dt
        A[2,4]= - action[1]*self.dt*task_param[1]

        return A

    def observations(self, s): # apply external noise and internal noise, to get observation
        '''
        takes in state x and output to observation of x
        '''
        # observation of velocity and angle have gain and noise, but no noise of position
        on = torch.distributions.Normal(0,self.phi[6:8]).sample() # on is observation noise
        vel, ang_vel = torch.split(s.view(-1),1)[-2:] # 1,5 to vector and take last two

        ovel = self.phi[4] * vel + on[0] # observe velocity
        oang_vel = self.phi[5]* ang_vel + on[1]
        o = torch.stack((ovel, oang_vel)) # observed x
        return o

    def observations_mean(self, s): # apply external noise and internal noise, to get observation
        
        vel, ang_vel = torch.split(s.view(-1),1)[-2:] # 1,5 to vector and take last two

        ovel = self.phi[4] * vel  
        oang_vel = self.phi[5]* ang_vel 
        o = torch.stack((ovel, oang_vel)) 
        return o


    def forward(self, action,theta=None):

        # unpack theta
        if theta is not None:
            self.pro_gains, self.pro_noise_stds, self.obs_gains, self.obs_noise_stds, self.goal_radius = torch.split(theta.view(-1), 2)
        
        # clamp the v
        # if type(action)==np.ndarray:
        #     action[0]=max(action[0],0)
        # else:
        #     action[0][action[0]<0]=0

        # true next state, xy position, reach target or not(have not decide if stop or not).
        next_x = self.x_step(self.x, action, self.dt, self.box, self.pro_gains, self.pro_noise_stds)
        pos = next_x.view(-1)[:2]
        reached_target = (torch.norm(pos) <= self.goal_radius) # is within ring
        x=next_x
        self.x=next_x

        # o t+1 
        # check the noise representation
        on = w=torch.distributions.Normal(0,self.obs_noise_stds).sample() # on is observation noise
        vel, ang_vel = torch.split(self.x.view(-1),1)[-2:] # 1,5 to vector and take last two.
        ovel = (self.obs_gains[0]) * vel + on[0] # observed velocity, has gain and noise
        oang_vel =(self.obs_gains[1]) * ang_vel + on[1] # same for anglular velocity
        next_ox = torch.stack((ovel, oang_vel)) # observed x t+1
        self.o=next_ox

        # b t+1
        # belef step foward, kalman filter update with new observation
        next_b = self.belief_step(self.b, next_ox, action, self.box)  
        self.b=next_b
        # belief next state, info['stop']=terminal # reward only depends on belief
        # in place update here. check
        
        # reshape b to give to policy
        self.belief = self.Breshape(b=next_b, time=self.episode_time, theta=self.theta)  # state used in policy is different from belief

        # reward

        reward = self.return_reward(info['stop'], reached_target, next_b, self.goal_radius, self.REWARD)

        # orignal return names
        self.episode_time=self.episode_time+1
        self.rewarded=reached_target and info['stop']
        self.stop=reached_target and info['stop'] or self.time>self.episode_len
        
        return self.belief, self.stop



    
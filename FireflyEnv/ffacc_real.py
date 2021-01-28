import gym
from gym import spaces
from numpy import pi
import numpy as np
from FireflyEnv.env_utils import *

class FireflyTrue1d_real(gym.Env, torch.nn.Module): 

    def __init__(self,arg=None,kwargs=None):

        super(FireflyTrue1d_real, self).__init__()
        
        self.arg=arg
        self.min_distance = 250
        self.max_distance = 550 
        self.terminal_vel = 20
        self.episode_len = 100
        self.dt = 0.2
        low=-np.inf
        high=np.inf
        # self.control_scalar=100
        self.action_space = spaces.Box(-np.ones(1), np.ones(1), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,14),dtype=np.float32)
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
        self.reset_state(goal_position=goal_position)
        self.reset_belief()
        self.reset_obs()
        self.reset_decision_info()
        self.actions=[]
        self.sys_vel=torch.zeros(1)
        self.obs_traj=obs_traj
        self.pro_traj=pro_traj
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


    def reset_state(self,goal_position=None):

        self.goalx = torch.zeros(1).uniform_(self.min_distance, self.max_distance)  # max d to the goal is world boundary.
        vel = torch.zeros(1) 

        if goal_position is not None:
            self.goalx = goal_position*torch.tensor(1.0)
        self.s = torch.cat([torch.zeros(1), torch.zeros(1)])
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


    def step(self, action):
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


        self.episode_reward, cost,mag,dev=self.caculate_reward()
        self.trial_sum_cost+=cost
        self.trial_mag+=mag
        self.trial_dev+=dev
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        if end_current_ep:
            print('distance, ', self.get_distance(), 'goal', self.goal_r)
            print('reward: {}, cost: {}, mag{}, dev{}'.format(self.episode_reward, self.trial_sum_cost, self.trial_mag, self.trial_dev))
        return self.decision_info, self.episode_reward-cost, end_current_ep, {}


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
        next_s = torch.cat((px, vel)).view(1,-1)
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
                        self.episode_time, 
                        vecL, task_param.view(-1)])
        return decision_info.view(1, -1)
        # only this is a row vector. everything else is col vector


    def caculate_reward(self):

        if self.stop:
            _,d= self.get_distance()
            if abs(d)<=self.phi[3,0]:
                reward=self.reward
            else:
                reward = (self.max_distance-abs(d))/self.max_distance*self.reward/2
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
        cost=torch.tensor(1.0)*action-previous_action
        return abs(cost)

    def action_cost_magnitude(self,action):
        return abs(action)

    def action_cost_wrapper(self,action, previous_action,mag_scalar, dev_scalar):
        mag_cost=self.action_cost_magnitude(action)
        dev_cost=self.action_cost_dev(action, previous_action)
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
        self.cost_scale = 1
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
        self.previous_action=torch.tensor([[0,0]])
        self.trial_sum_cost=0
        self.trial_mag=0
        self.trial_dev=0
        self.reset_state(goal_position=goal_position)
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


    def reset_state(self,goal_position=None):
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
        self.P = torch.eye(5) * 1e-3 
        self.b = self.s


    def reset_obs(self):
        self.o=torch.tensor([[0],[0]])


    def reset_decision_info(self):
        self.decision_info = self.wrap_decision_info(b=self.b, time=self.episode_time, task_param=self.theta)


    def step(self, action):
        action=torch.tensor(action)
        self.a=action
        self.episode_time=self.episode_time+1
        self.sys_vel=torch.norm(action)
        if not self.if_agent_stop(sys_vel=self.sys_vel) and not self.agent_start:
            self.agent_start=True
        if self.agent_start:
            self.stop=True if self.if_agent_stop(sys_vel=self.sys_vel) else False 
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
        # eval
        self.episode_reward, cost, mag, dev = self.caculate_reward()
        self.trial_sum_cost += cost
        self.trial_mag += mag
        self.trial_dev += dev
        if self.stop:
            print('distance, ', self.get_distance()[1], 'goal', self.goal_r,'sysvel',self.sys_vel)
            print('reward: {}, cost: {}, mag{}, dev{}'.format(self.episode_reward, self.trial_sum_cost, self.trial_mag, self.trial_dev))
        # dynamic
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b, self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        return self.decision_info, self.episode_reward-cost, end_current_ep, {}


    def if_agent_stop(self,sys_vel=None):
        stop=(sys_vel <= self.terminal_vel)
        return stop


    def forward(self, action,task_param,state=None, giving_reward=None):
        action=torch.tensor(action)
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
        v = self.pro_gainv * torch.tensor(1.0)*a[0] + noisev
        w = self.pro_gainw * torch.tensor(1.0)*a[1] + noisew
        next_s = torch.cat((px, py, angle, v, w)).view(1,-1)
        self.previous_action=a
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
        v=action[0]*self.pro_gainv_hat
        w=action[1]*self.pro_gainw_hat
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
        angle=b[3,0]
        vel=b[4,0]
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


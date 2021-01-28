import gym
from gym import spaces
from numpy import pi
import numpy as np
from FireflyEnv.env_utils import *
# from FireflyEnv.ffenv_new_cord import FireflyAgentCenter
from reward_functions import reward_singleff
from FireflyEnv.firefly_accac import FireflyAccAc

from FireflyEnv.firefly_base import FireflyEnvBase


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

def action_cost_magnitude(action):
    return abs(action)


def action_cost_dev(action, previous_action):
    # sum up the norm squre for delta action
    cost=torch.tensor(1.0).cuda()*action-previous_action
    return abs(cost)


def action_cost_wrapper(action, previous_action,mag_scalar, dev_scalar):
    mag_cost=action_cost_magnitude(action)
    dev_cost=action_cost_dev(action, previous_action)
    total_cost=mag_scalar*mag_cost+dev_scalar*dev_cost
    return total_cost, mag_cost, dev_cost


class FireflyTrue1d(gym.Env, torch.nn.Module): 

    def __init__(self,arg=None,kwargs=None):
        '''
        state-observation-blief-action-next state
        arg:
            gains_range, 4 element list

        step:  action -> new state-observation-blief    
        reset: init new state
        '''
        super(FireflyTrue1d, self).__init__()
        
        kwargs={} if kwargs is None else kwargs
        self.arg=arg

        # world param, that we are not chaning these during training
        self.world_size =           arg.WORLD_SIZE # float
        self.terminal_vel =         arg.TERMINAL_VEL # float
        self.episode_len =          arg.EPISODE_LEN # int
        self.dt =                   arg.DELTA_T # float
        # we have modified observation space ( new acc related parameters)
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Box(-np.ones(1), np.ones(1), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,14),dtype=np.float32)
        self.cost_function=action_cost_wrapper
        self.cost_scale=1 # the task is much longer. 
        self.trial_sum_cost=None
        self.training=arg.training if arg.training is not None else False
        if arg is not None:
            self.setup(arg,**kwargs)
            self.reset()
        else:
            raise ValueError('must have an arg input!')



    def setup(self,arg,
                presist_phi=False,      # flase when training, to generalize
                agent_knows_phi=True,   # true when training, to explore different tasks
                reward_function=None,
                reward=None,
                goal_radius_step=None,
                max_distance=None,
                goal_radius_range =None,
                param_range_dict={},
                    ):

        self.presist_phi=           presist_phi
        self.agent_knows_phi=       agent_knows_phi

        # set up the parameter range from arg

        self._apply_param_range(**param_range_dict) 
        self.max_distance=max_distance if max_distance is not None else self.world_size
        self.reward=reward if reward is not None else self.arg.REWARD
        self.reward_function=reward_singleff.belief_reward_mc if reward_function is None else reward_function
        self.phi=None


    # acc control state dynamics
    def a_(self, tau, dt=None):
        dt=self.dt if dt is None else dt
        if self.training:
            return torch.exp(-dt/tau)
        else:
            return torch.exp(-dt/tau).cuda()
    
    def vm_(self, tau, x=None, T=None):
        x=1.5*self.world_size if x is None else x
        T=self.episode_len*self.dt if T is None else T
        vm=x/2/tau *(1/(torch.log(torch.cosh(T/2/tau))) )
        if vm==0: # too much velocity control rather than acc control
            vm=x/T*torch.ones(1)
        if self.training:
            return vm
        else:
            return vm.cuda()

    def b_(self, tau, dt=None):
        dt=self.dt if dt is None else dt
        if self.training:
            return self.vm_(tau)*(1-self.a_(tau))
        else:
            return self.vm_(tau)*(1-self.a_(tau)).cuda()


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
                goal_position=None
                ): 
        
        if phi is not None:
            self.phi=phi
            # print('using input phi',self.phi)
        elif self.presist_phi and self.phi is not None:
            # print('using last phi',self.phi)
            pass
        else: # when either not presist, or no phi.
            self.phi=self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_gains=obs_gains,obs_noise_stds=obs_noise_stds)
            # print('generate new phi',self.phi)
        
        if theta is not None:
            # print('theta using input theta')
            self.theta=theta
        else:
            # print('generate new theta')
            self.theta=self.phi if self.agent_knows_phi else self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_gains=obs_gains,obs_noise_stds=obs_noise_stds)

        self.reset_state(goal_position=goal_position)
        self.reset_belief()
        self.reset_obs()
        self.reset_decision_info()

        return self.decision_info.view(1,-1)


    def _apply_param_range(self,gains_range=None,std_range=None,
                goal_radius_range=None,tau_range=None,
                mag_action_cost_range=None,dev_action_cost_range=None):

        if goal_radius_range is None:
            self.goal_radius_range =     self.arg.goal_radius_range
        if gains_range is None:
            self.gains_range=            self.arg.gains_range
        if std_range is None:
            self.std_range =             self.arg.std_range
        if tau_range is None:
            if self.arg.tau_range is not None:
                self.tau_range = self.arg.tau_range
            else:
                self.tau_range = compute_tau_range(self.gains_range, d=2*self.world_size, T=self.episode_len*self.dt, tau=1.8)
                self.tau_range[1]=min(8,self.tau_range[1])
        else:
            self.tau_range=tau_range
            print('tau range', tau_range)
        if mag_action_cost_range is None:
            self.mag_action_cost_range =     self.arg.mag_action_cost_range
        if dev_action_cost_range is None:
            self.dev_action_cost_range =     self.arg.dev_action_cost_range


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
        
        _pro_gains = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1]).cuda() 
        
        _pro_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1]).cuda() 
                
        _obs_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1]).cuda() 
        
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1]).cuda() 
        
        _tau = torch.zeros(1).uniform_(self.tau_range[0], self.tau_range[1]).cuda() 

        _mag_action_cost_factor = torch.zeros(1).uniform_(self.mag_action_cost_range[0], self.mag_action_cost_range[1]).cuda() 
        beta_range=self.constrained_cost_factor_range(_mag_action_cost_factor,total_steps=self.episode_len)
        _dev_action_cost_factor = torch.zeros(1).uniform_(float(beta_range[0]),float(beta_range[1])).cuda() 


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

        if self.training:
            return col_vector(phi).cpu()
        else:
            return col_vector(phi)
        # theta is 7 dim


    def reset_state(self,goal_position=None):

        min_distance = self.phi[3,0].item()+self.world_size*0.0  # min d to the goal is 0.2
        self.goalx =        torch.zeros(1).uniform_(min_distance, max(min_distance,min(self.world_size, self.max_distance))).cuda()  # max d to the goal is world boundary.
        vel = torch.zeros(1).cuda() 

        if goal_position is not None:
            self.goalx = goal_position*torch.tensor(1.0).cuda()
        else:
            pass

        self.s = torch.cat([torch.zeros(1), torch.zeros(1)]).cuda() 
        self.s = self.s.view(-1,1) # column vector
        self.agent_start=False


    def reset_belief(self): 

        self.P = torch.eye(2).cuda()  * 1e-8 
        self.b = self.s.view(-1,1).cuda()  # column vector


    def reset_obs(self):
        # obs at t0 on v and w, both 0
        self.o=torch.zeros(1).view(-1,1).cuda()        # column vector


    def reset_decision_info(self):
        self.episode_time = torch.zeros(1).cuda()   # int
        self.stop=False         # bool
        self.decision_info = self.wrap_decision_info(b=self.b, time=self.episode_time, task_param=self.theta)
        self.decision_info = row_vector(self.decision_info)
        self.previous_action=torch.zeros(1).cuda() 
        self.trial_sum_cost=0
        self.trial_mag=0
        self.trial_dev=0

# steps

    def step(self, action):

        if not self.if_agent_stop() and not self.agent_start:
            self.agent_start=True

        if self.agent_start:
            self.stop=True if self.if_agent_stop() else False 
        else:
            self.stop=False
        self.episode_time=self.episode_time+1
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)

        self.a=action
        self.episode_reward, cost,mag,dev=self.caculate_reward()
        self.trial_sum_cost+=cost
        self.trial_mag+=mag
        self.trial_dev+=dev
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        

        if end_current_ep:
            print('reward: {}, cost: {}, mag{}, dev{}'.format(self.episode_reward-cost, self.trial_sum_cost, self.trial_mag, self.trial_dev))
        return self.decision_info, self.episode_reward-cost, end_current_ep, {}

    def if_agent_stop(self,action=None,state=None,task_param=None):
        terminal=False
        terminal_vel = self.terminal_vel
        state=self.s if state is None else state
        px,  vel = torch.split(state.view(-1), 1)
        stop = (abs(vel) <= terminal_vel)
        if stop:
            terminal= True

        return terminal

    def forward(self, action,task_param,state=None, giving_reward=None):# this taskpram is theta

        if not self.if_agent_stop() and not self.agent_start:
            self.agent_start=True

        if self.agent_start:
            self.stop=True if self.if_agent_stop() else False 
        else:
            self.stop=False

        self.a=action.clone().detach()
        if state is None:
            self.s=self.state_step(action,self.s,task_param=self.phi)
        else: # deterministic update from observed value
            self.s=state
            # self.previous_action=action

        self.o=self.observations(self.s,task_param=task_param)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action,task_param=task_param)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=task_param)
        self.episode_time=self.episode_time+1
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
             
        return self.decision_info, end_current_ep


    def state_step(self,a, s, task_param=None):

        '''
        # acc control, vt+1 = a*vt + b*ut
        '''
        
        # STEP 1: use the v,w to move forward
        task_param=self.phi if task_param is None else task_param
        next_s = self.update_state(s, task_param=task_param)

        # STEP 2: use the action to assign v and w
        px, vel = torch.split(next_s.view(-1), 1)
        a_v = a  # action for velocity
        w=torch.distributions.Normal(0,torch.ones([1,1])).sample().cuda()*task_param[1,0]     
        vel = self.a_(task_param[4,0]) * vel + self.b_(task_param[4,0])*(task_param[0,0] * (torch.tensor(1.0).cuda()*a_v[0] + w[0]))
        next_s = torch.cat((px, vel)).view(1,-1)
        self.previous_action=a
        return next_s.view(-1,1)

    def constrained_cost_factor_range(self, alpha_param, max_cost=2.5,total_steps=40):
        # mag cost, assuming v is always max (which is true)
        # assuming 40 max steps
        mag_cost=alpha_param*total_steps*np.sqrt(2)
        max_dev_cost=max_cost-mag_cost
        max_beta_param=max_dev_cost/3./np.sqrt(2) # theortically, 0 to 1/-1 then to opposite, has the min dev cost
        beta_param_range= [0.001, max_beta_param]
        return beta_param_range


    def update_state(self, state,task_param=None):

        task_param=self.phi if task_param is None else task_param
        px, vel = torch.split(state.view(-1), 1)
        px = px + vel * self.dt # new position x and y
        px = torch.clamp(px, -self.world_size, self.world_size) # restrict location inside arena, to the edge
        next_s = torch.stack((px, vel))
        return next_s.view(-1,1)


    def apply_action(self,state,action,task_param=None):

        task_param=self.phi if task_param is None else task_param
        px, vel = torch.split(state.view(-1), 1)
        vel=vel*self.a_(task_param[4,0])+action[0]*task_param[0,0]*self.b_(task_param[4,0])
        next_s = torch.stack((px, vel))
        return next_s.view(-1,1)



    def belief_step(self, previous_b,previous_P, o, a, task_param=None):
        
        
        task_param = self.theta if task_param is None else task_param
        I = torch.eye(2).cuda()
        Q = torch.zeros(2,2).cuda()
        Q[1,1]=(self.b_(task_param[4])*task_param[0,0]*task_param[1,0])**2
        R = torch.tensor(1.0).cuda()*(self.s[1,0]*task_param[2,0])**2
        if R < 1e-4:
            R=R+1e-4
        H = torch.zeros(1,2).cuda()
        H[-1,-1] = 1

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
            print('theta: ', task_param)
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
            print("after update not pos def")
            print("updated P:", P)
            print('Q : ', Q)
            print('R : ', R)
            # print("K:", K)
            # print("H:", H)
            print("I - KH : ", I_KH)
            print("error : ", error)
            print('task parameter : ',task_param)
            P = (P + P.t()) / 2 + 1e-6 * I  # make symmetric to avoid computational overflows
        # print("error : ", error)
        # print("K:", K)
        # print("b: ", b)
        # b=self.update_state(b)
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
            if self.reached_goal():
                reward=self.reward
                # reward = self.reward_function(self.stop, self.reached_goal(),
                # self.b,self.P,self.phi[8,0],self.reward,
                # self.goalx,self.goaly, verbose=False) # no time here, use cost to reduce time.
            else:
                _,d= self.get_distance()
                reward = max(0, (1-d)) 
        else:
            neg_velocity=self.s[1] if self.s[1]<0 else 0 # punish the backward
            reward = neg_velocity

        cost, mag, dev=self.cost_function(self.a, self.previous_action,mag_scalar=self.phi[5,0], dev_scalar=self.phi[6,0])
        return reward, self.cost_scale*cost, mag, dev



    def transition_matrix(self, action, state, task_param=None): 

        task_param = self.theta if task_param is None else task_param

        px,  vel = torch.split(state.view(-1),1)
        
        A = torch.zeros(2,2).cuda()
        A[0,0] = 1
        # partial dev with v
        A[0, 1] = self.dt
        A[1, 1] = self.a_(task_param[4,0])

        return A.cuda()

    def reached_goal(self):
        # use real location
        _,distance=self.get_distance(state=self.s)
        reached_bool= (distance<=self.phi[3,0])
        return reached_bool

    def observations(self, s, task_param=None): 

        
        task_param=self.theta if task_param is None else task_param

        on = torch.distributions.Normal(0,torch.tensor(1.).cuda()).sample()*task_param[2,0]
        vel = s.view(-1)[-1] # 1,5 to vector and take last two

        # ovel = task_param[2,0] * vel *(1+on[0])
        ovel =  vel + on
        return ovel.view(-1,1).cuda()


    def observations_mean(self, s, task_param=None): # apply external noise and internal noise, to get observation

        task_param=self.theta if task_param is None else task_param
        # sample some noise
        vel = s.view(-1)[-1] # 1,5 to vector and take last two
        ovel = vel
        return ovel.view(-1,1).cuda()



    def get_distance(self, state=None): 


        state=self.s if state is None else state
        position = state[0]
        distance = self.goalx-state[0]
        return position, distance
    














class FireflyTrue1d_cpu(gym.Env, torch.nn.Module): 

    def __init__(self,arg=None,kwargs=None):
        '''
        state-observation-blief-action-next state
        arg:
            gains_range, 4 element list

        step:  action -> new state-observation-blief    
        reset: init new state
        '''
        super(FireflyTrue1d_cpu, self).__init__()
        
        kwargs={} if kwargs is None else kwargs
        self.arg=arg

        # world param, that we are not chaning these during training
        self.world_size =           arg.WORLD_SIZE # float
        self.terminal_vel =         arg.TERMINAL_VEL # float
        self.episode_len =          arg.EPISODE_LEN # int
        self.dt =                   arg.DELTA_T # float
        # we have modified observation space ( new acc related parameters)
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Box(-np.ones(1), np.ones(1), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,14),dtype=np.float32)
        self.cost_function=self.action_cost_wrapper
        self.cost_scale=1 # the task is much longer. 
        self.trial_sum_cost=None
        self.training=arg.training if arg.training is not None else False
        if arg is not None:
            self.setup(arg,**kwargs)
            self.reset()
        else:
            raise ValueError('must have an arg input!')



    def setup(self,arg,
                presist_phi=False,      # flase when training, to generalize
                agent_knows_phi=True,   # true when training, to explore different tasks
                reward_function=None,
                reward=None,
                goal_radius_step=None,
                max_distance=None,
                goal_radius_range =None,
                param_range_dict={},
                    ):

        self.presist_phi=           presist_phi
        self.agent_knows_phi=       agent_knows_phi
        self._apply_param_range(**param_range_dict) 
        self.max_distance=max_distance if max_distance is not None else self.world_size
        self.reward=reward if reward is not None else self.arg.REWARD
        self.reward_function=reward_singleff.belief_reward_mc if reward_function is None else reward_function
        self.phi=None


    def a_(self, tau, dt=None):
        dt=self.dt if dt is None else dt
        return torch.exp(-dt/tau)
    
    def vm_(self, tau, x=None, T=None):
        x=1.5*self.world_size if x is None else x
        T=self.episode_len*self.dt if T is None else T
        vm=x/2/tau *(1/(torch.log(torch.cosh(T/2/tau))) )
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
                goal_position=None
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

        self.reset_state(goal_position=goal_position)
        self.reset_belief()
        self.reset_obs()
        self.reset_decision_info()
        self.unpack_theta()
        return self.decision_info.view(1,-1)

    def unpack_theta(self):
        self.pro_gain=self.theta[0]
        self.pro_noise=self.theta[1]
        self.obs_noise=self.theta[2]
        self.goal_r=self.theta[3]
        # self.tau=self.theta[4]
        self.mag_cost=self.theta[5]
        self.dev_cost=self.theta[6]
        self.tau_a=self.a_(self.theta[4])
        self.tau_b=self.b_(self.theta[4])


    def _apply_param_range(self,gains_range=None,std_range=None,
                goal_radius_range=None,tau_range=None,
                mag_action_cost_range=None,dev_action_cost_range=None):

        if goal_radius_range is None:
            self.goal_radius_range =     self.arg.goal_radius_range
        if gains_range is None:
            self.gains_range=            self.arg.gains_range
        if std_range is None:
            self.std_range =             self.arg.std_range
        if tau_range is None:
            if self.arg.tau_range is not None:
                self.tau_range = self.arg.tau_range
            else:
                self.tau_range = compute_tau_range(self.gains_range, d=2*self.world_size, T=self.episode_len*self.dt, tau=1.8)
                self.tau_range[1]=min(8,self.tau_range[1])
        else:
            self.tau_range=tau_range
            print('tau range', tau_range)
        if mag_action_cost_range is None:
            self.mag_action_cost_range =     self.arg.mag_action_cost_range
        if dev_action_cost_range is None:
            self.dev_action_cost_range =     self.arg.dev_action_cost_range


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
                
        _obs_noise_stds=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        
        _goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.goal_radius_range[1])
        
        _tau = torch.zeros(1).uniform_(self.tau_range[0], self.tau_range[1])

        _mag_action_cost_factor = torch.zeros(1).uniform_(self.mag_action_cost_range[0], self.mag_action_cost_range[1]) 
        beta_range=self.constrained_cost_factor_range(_mag_action_cost_factor,total_steps=self.episode_len)
        _dev_action_cost_factor = torch.zeros(1).uniform_(float(beta_range[0]),float(beta_range[1]))


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

        min_distance = self.phi[3,0].item()+self.world_size*0.0  # min d to the goal is 0.2
        self.goalx = torch.zeros(1).uniform_(min_distance, max(min_distance,min(self.world_size, self.max_distance)))  # max d to the goal is world boundary.
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

# steps

    def step(self, action):
        
        if not self.if_agent_stop() and not self.agent_start:
            self.agent_start=True

        if self.agent_start:
            self.stop=True if self.if_agent_stop() else False 
        else:
            self.stop=False
        self.episode_time=self.episode_time+1
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)

        self.a=action
        self.episode_reward, cost,mag,dev=self.caculate_reward()
        self.trial_sum_cost+=cost
        self.trial_mag+=mag
        self.trial_dev+=dev
        self.s=self.state_step(action,self.s)
        self.o=self.observations(self.s)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=self.theta)
        if end_current_ep:
            print('reward: {}, cost: {}, mag{}, dev{}'.format(self.episode_reward, self.trial_sum_cost, self.trial_mag, self.trial_dev))
        return self.decision_info, self.episode_reward-cost, end_current_ep, {}


    def if_agent_stop(self,action=None,state=None,task_param=None):
        terminal=False
        terminal_vel = self.terminal_vel
        state=self.s if state is None else state
        px,  vel = torch.split(state.view(-1), 1)
        stop = (abs(vel) <= terminal_vel)
        if stop:
            terminal= True

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

        self.o=self.observations(self.s,task_param=task_param)
        self.b,self.P=self.belief_step(self.b,self.P,self.o,action,task_param=task_param)
        self.decision_info=self.wrap_decision_info(b=self.b,P=self.P, time=self.episode_time,task_param=task_param)
        self.episode_time=self.episode_time+1
        end_current_ep=(self.stop or self.episode_time>=self.episode_len)
             
        return self.decision_info, end_current_ep


    def state_step(self,a, s, task_param=None):

        '''
        # acc control, vt+1 = a*vt + b*ut
        '''
        
        # STEP 1: use the v,w to move forward
        task_param=self.phi if task_param is None else task_param
        next_s = self.update_state(s, task_param=task_param)

        # STEP 2: use the action to assign v and w
        px, vel = torch.split(next_s.view(-1), 1)
        a_v = a  # action for velocity
        w=torch.distributions.Normal(0,torch.ones([1,1])).sample()*self.pro_noise   
        vel = self.tau_a * vel + self.tau_b*(self.pro_gain * (torch.tensor(1.0)*a_v[0] + w[0]))
        next_s = torch.cat((px, vel)).view(1,-1)
        self.previous_action=a
        return next_s.view(-1,1)


    def constrained_cost_factor_range(self, alpha_param, max_cost=2.5,total_steps=40):

        mag_cost=alpha_param*total_steps*np.sqrt(2)
        max_dev_cost=max_cost-mag_cost
        max_beta_param=max_dev_cost/3./np.sqrt(2) # theortically, 0 to 1/-1 then to opposite, has the min dev cost
        beta_param_range= [0.001, max_beta_param]
        return beta_param_range


    def update_state(self, state,task_param=None):

        task_param=self.phi if task_param is None else task_param
        px, vel = torch.split(state.view(-1), 1)
        px = px + vel * self.dt # new position x and y
        px = torch.clamp(px, -self.world_size, self.world_size) # restrict location inside arena, to the edge
        next_s = torch.stack((px, vel))
        return next_s.view(-1,1)


    def apply_action(self,state,action,task_param=None):

        task_param=self.phi if task_param is None else task_param
        px, vel = torch.split(state.view(-1), 1)
        vel=vel*self.tau_a+action[0]*self.pro_gain*self.tau_b
        next_s = torch.stack((px, vel))
        return next_s.view(-1,1)



    def belief_step(self, previous_b,previous_P, o, a, task_param=None):
        
        task_param = self.theta if task_param is None else task_param
        I = torch.eye(2)
        Q = torch.zeros(2,2)
        Q[1,1]=(self.tau_b*self.pro_gain*self.pro_noise)**2
        R = torch.tensor(1.0)*(self.s[1,0]*self.obs_noise)**2
        if R < 1e-4:
            R=R+1e-4
        H = torch.zeros(1,2)
        H[-1,-1] = 1

        # prediction
        predicted_b = self.update_state(previous_b,task_param=task_param)
        predicted_b = self.apply_action(predicted_b,a,task_param=task_param)
        A = self.transition_matrix(a,previous_b,task_param) 
        predicted_P = A@(previous_P)@(A.t())+Q # estimate Pt+1 = APA^T+Q, 

        if not is_pos_def(predicted_P): # should be positive definate. if not, show debug. 
            # if noise go to 0, happens.
            print('theta: ', task_param)
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


        if not is_pos_def(P): 
            print("after update not pos def")
            print("updated P:", P)
            print('Q : ', Q)
            print('R : ', R)
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
            if self.reached_goal():
                reward=self.reward
                # reward = self.reward_function(self.stop, self.reached_goal(),
                # self.b,self.P,self.phi[8,0],self.reward,
                # self.goalx,self.goaly, verbose=False) # no time here, use cost to reduce time.
            else:
                _,d= self.get_distance()
                reward = max(0, (1-d)) 
        else:
            neg_velocity=self.s[1] if self.s[1]<0 else 0 # punish the backward
            reward = neg_velocity

        cost, mag, dev=self.cost_function(self.a, self.previous_action,mag_scalar=self.phi[5,0], dev_scalar=self.phi[6,0])
        return reward, self.cost_scale*cost, mag, dev



    def transition_matrix(self, action, state, task_param=None): 

        task_param = self.theta if task_param is None else task_param

        px,  vel = torch.split(state.view(-1),1)
        
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


    def observations(self, s, task_param=None): 

        task_param=self.theta if task_param is None else task_param
        on = torch.distributions.Normal(0,torch.tensor(1.)).sample()*self.obs_noise
        vel = s.view(-1)[-1] # 1,5 to vector and take last two
        ovel =  vel + on
        return ovel.view(-1,1)


    def observations_mean(self, s, task_param=None): # apply external noise and internal noise, to get observation

        task_param=self.theta if task_param is None else task_param
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
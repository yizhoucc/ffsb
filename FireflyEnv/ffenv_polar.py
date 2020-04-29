import gym
from numpy import pi
import numpy as np
from gym import spaces
import InverseFuncs
from .env_utils import *
from DDPGv2Agent.rewards import * #reward

class FireflyEnv(gym.Env, torch.nn.Module): 
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self,arg=None,pro_gains = None, pro_noise_stds = None, obs_gains = None, obs_noise_stds = None):
        '''
        state-observation-blief-action-next state
        arg:
            gains_range, 4 element list
        step:  action -> new state-observation-blief    
        reset: init new state
        '''
        super(FireflyEnv, self).__init__()
        
        # specific defines
        if arg is not None:
            self.setup(arg)

    def setup(self,arg, param_representation='normalize',action_clamping='soft',pro_gains = None, pro_noise_stds = None, obs_gains = None, obs_noise_stds = None):
        '''apply the arg and re init'''
        if 'norm' in param_representation:
            param_denorm=denorm_parameter
        else:
            param_denorm=placeholder_func # will return input
        self.dt = arg. DELTA_T
        self.episode_len = arg.EPISODE_LEN
        self.arena_size = arg.WORLD_SIZE #initial value
        self.max_goal_radius = arg.goal_radius_range[0]
        self.GOAL_RADIUS_STEP = arg.GOAL_RADIUS_STEP_SIZE
        self.phi=None
        self.presist_phi=None
        self.dt = arg.DELTA_T
        self.P = torch.eye(5) * 1e-8
        self.terminal_vel = arg.TERMINAL_VEL
        self.goal_radius_range = arg.goal_radius_range
        self.gains_range = arg.gains_range
        self.std_range=arg.std_range
        self.REWARD=arg.REWARD
        low = np.concatenate(([0., -pi, -1., 0.], -10*np.ones(15),
            [self.gains_range[0],self.gains_range[0],self.std_range[0],self.std_range[0],
            self.gains_range[2],self.gains_range[2],self.std_range[2],self.std_range[2],self.goal_radius_range[0]])).reshape(1,29)
        high = np.concatenate(([10., pi, 1., 10.], 10*np.ones(15),
            [self.gains_range[1],self.gains_range[1],self.std_range[3],self.std_range[3],
            self.gains_range[3],self.gains_range[3],self.std_range[3],self.std_range[3],self.goal_radius_range[1]])).reshape(1,29)
        self.action_space = spaces.Box(-np.ones(2), np.ones(2), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,dtype=np.float32)
        self.pro_gains=pro_gains
        self.pro_noise_stds=pro_noise_stds
        self.obs_gains=obs_gains
        self.obs_noise_stds=obs_noise_stds
        self.goal_radius=None
        self.stop=False
        self.reset_theta=True
        if action_clamping=='hard':
            self.action_clamping=hard_clamp
        elif action_clamping=='sigmoid':
            self.action_clamping=soft_clamp
        self.reset()

    def reset(self): 

        # task param
        if  self.phi is None: # defaul. generate theta from range if no preset phi avaliable
            
            if self.pro_gains==None or self.reset_theta:
                self.pro_gains = torch.zeros(2)
                self.pro_gains[0] = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  #[proc_gain_vel]
                self.pro_gains[1] = torch.zeros(1).uniform_(self.gains_range[2], self.gains_range[3])  # [proc_gain_ang]
            
            if self.pro_noise_stds==None or self.reset_theta:
                self.pro_noise_stds=torch.zeros(2)
                self.pro_noise_stds[0]=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
                self.pro_noise_stds[1]=torch.zeros(1).uniform_(self.std_range[2],self.std_range[3])
            
            if self.goal_radius==None or self.reset_theta:
                self.max_goal_radius = min(self.max_goal_radius + self.GOAL_RADIUS_STEP, self.goal_radius_range[1])
                self.goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.max_goal_radius)

            if self.obs_gains==None or self.reset_theta:
                self.obs_gains = torch.zeros(2)
                self.obs_gains[0] = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  # [obs_gain_vel]
                self.obs_gains[1] = torch.zeros(1).uniform_(self.gains_range[2], self.gains_range[3])  # [obs_gain_ang]
            if self.obs_noise_stds==None or self.reset_theta:
                self.obs_noise_stds=torch.zeros(2)
                self.obs_noise_stds[0]=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
                self.obs_noise_stds[1]=torch.zeros(1).uniform_(self.std_range[2],self.std_range[3])

        else: 
            # use the preset phi
            self.fetch_phi()
        
        self.theta = torch.cat([self.pro_gains, self.pro_noise_stds, self.obs_gains, self.obs_noise_stds, self.goal_radius])
        
        # time
        self.time = torch.zeros(1)
        self.stop=False
        
        # state
        min_r = self.goal_radius.item()+self.arena_size/10 # the agent has to move to be within radius
        d = torch.zeros(1).uniform_(min_r, self.arena_size)  # GOAL_RADIUS, self.box is world size
        rel_ang = torch.zeros(1).uniform_(-pi/4, pi/4)
        vel = torch.zeros(1)
        ang_vel = torch.zeros(1)
        self.x = torch.cat([d,rel_ang, vel, ang_vel]) # this is state x at t0
        self.o=torch.zeros(2) # this is observation o at t0
        self.a=torch.zeros(2) # this will be action a at t0
        self.P = torch.eye(5) * 1e-8 # change 4 to size function
        self.b = self.x, self.P  # belief=x because is not move yet, and no noise on x, y, angle
        self.decision_info = self.decision_info_reshape(b=self.b, time=self.time, theta=self.theta)
        # return self.b, self.state, self.obs_gains, self.obs_noise_ln_vars
        # print(self.belief.shape) #1,29
        return self.belief # this is belief at t0
    
    def step(self, action): # state and action to state, and belief
        debug_info={}
        action[0]=max(action[0],0)
    
        self.x = self.x_step(self.x, action, self.pro_gains, self.pro_noise_stds)
        self.o=self.observations(self.x)
        self.b, agent_stops = self.belief_step(self.b, self.o, action, self.arena_size)  
        self.decision_info = self.decision_info_reshape(self.b, self.time, self.theta)  # state used in policy is different from belief

        # reward
        reached_target = (self.x[0] <= self.goal_radius) # is within ring
        agent_stops = self._isTerminal(action) # check the monkey stops or not

        reward = return_reward(agent_stops, reached_target, self.b, self.goal_radius, self.REWARD, finetuning)

        # orignal return names
        self.time=self.time+1
        self.end_current_ep=agent_stops or self.time>=self.episode_len-1
        
        return self.decision_info, reward, self.end_current_ep, debug_info


    def x_step(self,x, a, pro_gains, pro_noise_stds): # to state next

        distance, rel_angle, vel, ang_vel = torch.split(x.view(-1), 1)

        v = a[0]*self.dt  # velocity
        w = a[1]*self.dt  # angular velocity

        noise=torch.distributions.Normal(0,self.pro_noise_stds).sample()

        vel = 0.0 * vel + pro_gains[0] * v + noise[0] # discard prev velocity and new v=gain*new v+noise
        ang_vel = 0.0 * ang_vel + pro_gains[1] * w + wnoise[1]

        _distance=np.sqrt((distance-np.cos(range_angle(rel_angle-w))*v)**2+(np.sin(range_angle(rel_angle-w))*v)**2)
        _rel_angle=-pi/2-rel_angle+w-np.arctan(abs(distance-np.cos(range_angle(rel_angle-w))*v)/abs(np.sin(range_angle(rel_angle-w))*v))

        next_x = torch.stack((_distance, _rel_angle, vel, ang_vel))

        return next_x.view(1,-1)

    def belief_step(self,  b, o, a, box):

        I = torch.eye(4)
        # Q matrix, process noise for tranform xt to xt+1. only applied to v and w
        Q = torch.zeros(4, 4)
        Q[-2:, -2:] = torch.diag(self.pro_noise_stds**2) # variance of vel, ang_vel
        
        # R matrix, observe noise for observation
        R = torch.diag(self.obs_noise_stds** 2)

        # H matrix, transform x into observe space. only applied to v and w.
        H = torch.zeros(2, 4)
        H[:, -2:] = torch.diag(self.obs_gains)

        # Extended Kalman Filter
        pre_bx_, P = b
        bx_ = self.x_step(pre_bx_, a, self.dt, box, self.pro_gains, self.pro_noise_stds) # estimate xt+1 from xt and at
        bx_ = bx_.t() # make a column vector
        A = self.A(bx_) # calculate the A matrix, to apply on covariance matrix 
        P_ = A.mm(P).mm(A.t())+Q # estimate Pt+1 = APA^T+Q, 
        if not is_pos_def(P_): # should be positive definate. if not, show debug. 
            # if noise go to 0, happens.
            print("P_:", P_)
            print("P:", P)
            print("A:", A)
            APA = A.mm(P).mm(A.t())
            print("APA:", APA)
            print("APA +:", is_pos_def(APA))
        error = o - self.observations(bx_) # error as z-hx, the xt+1 is estimated
        S = H.mm(P_).mm(H.t()) + R # S = HPH^T+R. the covarance of observation of xt->xt+1 transition 
        K = P_.mm(H.t()).mm(torch.inverse(S)) # K = PHS^-1, kalman gain. has a H in front but canceled out
        bx = bx_ + K.matmul(error) # update xt+1 from the estimated xt+1 using new observation zt
        I_KH = I - K.mm(H)
        P = I_KH.mm(P_)# update covarance of xt+1 from the estimated xt+1 using new observation zt noise R

        if not is_pos_def(P): 
            print("here")
            print("P:", P)
            P = (P + P.t()) / 2 + 1e-6 * I  # make symmetric to avoid computational overflows

        bx = bx.t() #return to a row vector
        b = bx.view(-1), P  # belief. estimated state and uncertainty matrix

        return b 

    def decision_info_reshape(self, **kwargs): # reshape the belief, ready for policy
        '''
        reshape belief for policy
        '''
        argin={} # b, time, theta
        for key, value in kwargs.items():
            argin[key]=value

        try: 
            pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = InverseFuncs.unpack_theta(argin['theta']) # unpack the theta
        except KeyError: pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = InverseFuncs.unpack_theta(self.theta)
        
        try: 
            time=argin['time']
        except KeyError: time=self.time
        
        try: 
            b=argin['b']
        except KeyError: b=self.b

        x, P = b # unpack the belief
        distance, rel_ang, vel, ang_vel = torch.split(x.view(-1), 1) # unpack state x
        vecL = vectorLowerCholesky(P) # take the lower triangle of P
        state = torch.cat([distance, rel_ang, vel, ang_vel, time, vecL, pro_gains.view(-1), pro_noise_stds.view(-1), obs_gains.view(-1), obs_noise_stds.view(-1), torch.ones(1)*goal_radius]) # original
        #state = torch.cat([r, rel_ang, vel, ang_vel]) time, vecL]) #simple

        return state.view(1, -1)
    
    def A(self, x_): # create the transition matrix
        '''
        used in kalman filter as transformation matrix for xt to xt+1
        '''
        distance, rel_angle, vel, ang_vel = torch.split(x.view(-1), 1)
        A_ = torch.zeros(4, 4)
        A_[:2, :2] = torch.eye(3)
        A_[0, 2] = np.sqrt((distance-np.cos(range_angle(rel_angle-w))*v)**2+(np.sin(range_angle(rel_angle-w))*v)**2)
        A_[1, 2] = -pi/2-rel_angle+w-np.arctan(abs(distance-np.cos(range_angle(rel_angle-w))*v)/abs(np.sin(range_angle(rel_angle-w))*v))
        return A_

    def observations(self, x,theta=None): # apply external noise and internal noise, to get observation
        '''
        takes in state x and output to observation of x
        '''
        if theta is not None:
            self.pro_gains, self.pro_noise_stds, self.obs_gains, self.obs_noise_stds, self.goal_radius = torch.split(theta.view(-1), 2)
        
        noise = w=torch.distributions.Normal(0,self.obs_noise_stds).sample() # on is observation noise
        vel, ang_vel = torch.split(x.view(-1),1)[-2:] # 1,5 to vector and take last two

        ovel = self.obs_gains[0] * vel + noise[0] # observe velocity
        oang_vel = self.obs_gains[1]* ang_vel + noise[1]
        o = torch.stack((ovel, oang_vel)) # observed x
        return o

    def observations_mean(self, x,theta=None): # apply external noise and internal noise, to get observation
        '''
        takes in state x and output to observation of x
        '''
        if theta is not None:
            self.pro_gains, self.pro_noise_stds, self.obs_gains, self.obs_noise_stds, self.goal_radius = torch.split(theta.view(-1), 2)
        
        # observation of velocity and angle have gain and noise, but no noise of position
        vel, ang_vel = torch.split(x.view(-1),1)[-2:] # 1,5 to vector and take last two

        ovel = self.obs_gains[0] * vel
        oang_vel = self.obs_gains[1]* ang_vel
        o = torch.stack((ovel, oang_vel)) # observed x
        return o
    
    def _isTerminal(self, a): 
        terminal_vel = self.terminal_vel
        stop = (torch.norm(torch.tensor(a, dtype=torch.float64)) <= terminal_vel)
        if stop:
            terminal= torch.ByteTensor([True])
        else:
            terminal= torch.ByteTensor([False])

        return terminal.item() == 1

    def logger(self):
        '''
        save the self.x, self.o, self.b, self.a for time t. 
        '''
        pass

    def assign_phi(self,phi):
        # call from outside to assign phi, so next env.reset use this phi instead of generate random from range
        #TODO, chech if given phi within box range, throw a error
        # right now the phi is also generated using the range, so not big deal
        self.phi=phi
        self.reset()

    def assign_presist_phi(self,phi):
        # call from outside to assign phi, so all next env.reset use this phi
        # until set presist_phi to false and clear phi to none
        self.presist_phi=True
        self.assign_phi(phi)


    def fetch_phi(self):
        # call before generating new phi to fetch the assigned phi
        if self.phi is not None:
            # do assign values
            if type(self.phi)==tuple:
                self.pro_gains, self.pro_noise_stds, self.obs_gains, self.obs_noise_stds,  self.goal_radius=self.phi
            else:
                self.pro_gains, self.pro_noise_stds, self.obs_gains, self.obs_noise_stds,  self.goal_radius = torch.split(self.phi.view(-1), 2)
            # clear 
            if self.presist_phi:
                pass
            else:
                self.phi=None
            return True
        else:
            return False

    def forward(self, action,theta=None):
        # clamp the v
        action[action<0]=0
        # unpack theta
        if theta is not None:
            self.pro_gains, self.pro_noise_stds, self.obs_gains, self.obs_noise_stds, self.goal_radius = torch.split(theta.view(-1), 2)
        
        # true next state, xy position, reach target or not(have not decide if stop or not).
        next_x = self.x_step(self.x, action, self.dt, self.arena_size, self.pro_gains, self.pro_noise_stds)
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
        next_b, info = self.belief_step(self.b, next_ox, action, self.arena_size)  
        self.b=next_b
        # belief next state, info['stop']=terminal # reward only depends on belief
        # in place update here. check
        
        # reshape b to give to policy
        self.belief = self.decision_info_reshape(b=next_b, time=self.time, theta=self.theta)  # state used in policy is different from belief

        # reward
        episode=1 # sb has its own countings. will discard this later
        finetuning=0 # not doing finetuning
        reward = return_reward(episode, info, reached_target, next_b, self.goal_radius, self.REWARD, finetuning)

        # orignal return names
        self.time=self.time+1
        self.stop=reached_target and info['stop'] or self.time>self.episode_len
        
        return self.belief, self.stop



def hard_clamp(action):
    return max(action[0],0)


def soft_clamp(action):
    action[0]=torch.sigmoid(action[0])
    return action
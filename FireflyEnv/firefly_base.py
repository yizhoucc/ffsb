import gym
from numpy import pi
import numpy as np
from gym import spaces
from .env_utils import *
from reward_functions import reward_singleff


class FireflyEnvBase(gym.Env, torch.nn.Module): 
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self,arg=None,kwargs=None):
        '''
        state-observation-blief-action-next state
        arg:
            gains_range, 4 element list
        step:  action -> new state-observation-blief    
        reset: init new state
        '''
        super(FireflyEnvBase, self).__init__()

        '''
        args:
        dt, timestamp smallest unit
        box, world size
        max goal radius
        phi, preset phi
        presist phi, if the phi is used presistly, as the teacher in simulation
            should be changed into, inverse or forward.
        term velocity, the threshold that determin if the agent stops
        episode length, how long each episdoe
        parameter ranges, as in theta. contain goal radius range, gain range, std range.
        reward, the max reward.
        
        '''
        kwargs={} if kwargs is None else kwargs
        self.arg=arg

        # world param, that we are not chaning these during training
        self.world_size =           arg.WORLD_SIZE 
        self.terminal_vel =         arg.TERMINAL_VEL
        self.episode_len =          arg.EPISODE_LEN
        self.dt =                   arg.DELTA_T
        self.goal_radius_step=      arg.GOAL_RADIUS_STEP_SIZE
        low = -np.inf
        high = np.inf
        self.action_space = spaces.Box(-np.ones(2), np.ones(2), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(1,29),dtype=np.float32)
        
        
        if arg is not None:
            self.setup(arg,**kwargs)
            self.reset()
        else:
            raise ValueError('must have arg input!')

    def _apply_param_range(self,gains_range=None,std_range=None,goal_radius_range=None):
        # TODO, not used
        if goal_radius_range is None:
            self.goal_radius_range =     self.arg.goal_radius_range
        if gains_range is None:
            self.gains_range=            self.arg.gains_range
        if std_range is None:
            self.std_range =             self.arg.std_range

    def setup(self,arg,
                presist_phi=False,      # flase when training, to generalize
                agent_knows_phi=True,   # true when training, to explore different tasks
                reward_function=None,
                reward=None,
                max_goal_radius =None,
                goal_radius_step=None      ,
                max_distance=None,
                goal_radius_range =None,
                    ):
        self.presist_phi=           presist_phi
        self.agent_knows_phi=       agent_knows_phi

        # set up the parameter range from arg
        self._apply_param_range() 

        self.max_distance=max_distance if max_distance is not None else self.world_size
        self.max_goal_radius=max_goal_radius if max_goal_radius is not None else self.goal_radius_range[1]

        self.reward=reward if reward is not None else self.arg.REWARD
        self.reward_function=reward_singleff.belief_gaussian_reward if reward_function is None else reward_function


    def reset(self,
                pro_gains = None, 
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                phi=None,

                goal_radius_range=None,
                gains_range = None,
                std_range=None,
                ): 
        


        # random a parameter from range

        self.phi=phi if phi is not None else self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_gains=obs_gains,obs_noise_stds=obs_noise_stds)
        self.theta=self.phi if self.agent_knows_phi else self.reset_task_param(pro_gains=pro_gains,pro_noise_stds=pro_noise_stds,obs_gains=obs_gains,obs_noise_stds=obs_noise_stds)
        
        self.reset_state()
        self.reset_belief()
        self.reset_obs()
        self.reset_decision_info()

        return self.decision_info

#-----------------------------------------------------------------------------
    # necessary functions 

    def forward(self, action,theta=None):
        '''
        run dynamic using torch's way
        '''
        raise NotImplementedError


    def reset_state(self):
        # reset the state, including goal position and agent position/angle.
        raise NotImplementedError


    def reset_belief(self):
        # reset the belief mean equal to state
        # and reset the cov matrix diag to be small values.
        raise NotImplementedError


    def reset_obs(self):
        raise NotImplementedError


    def reset_theta(self):
        raise NotImplementedError


    def reset_decision_info(self):
        # use wrap decision info function to generate decision info at t0
        raise NotImplementedError


    def step(self, action): # state and action to state, and belief
        '''
        # input:
            # action
        # return:
            # observation, a object, 
            # reward, float, want to maximaz
            # done, bool, when 1, reset
            # info, dict, for debug

        in this case, we want to give the belief state as observation to choose action
        store real state in self to decide reward, done.

        self.states, self.belief, action, self.noises
        1. states update by action
        2. belief update by action and states with noise, by kalman filter
        
        '''
        raise NotImplementedError


    def state_step(self,x, a, dt, box, pro_gains, pro_noise_stds): # to state next
        '''
        state dynamic
        '''
        raise NotImplementedError


    def belief_step(self,  b, ox, a, box): # to belief next
        '''
        belief dynamic
        '''
        raise NotImplementedError


    def wrap_decision_info(self, b, time, theta): # reshape the belief, ready for policy
        '''
        concat belief, time, task param for policy input
        '''
        raise NotImplementedError
    

    def A(self, x_): # create the transition matrix
        '''
        transition matrix
        '''
        raise NotImplementedError


    def observations(self, x): 
        '''
        takes in state x and output to observation of x
        with gain and noise
        without dt
        '''
        raise NotImplementedError


    def observations_mean(self, x):
        '''
        takes in state x and output to observation of x, 
        with gain 
        without noise, dt
        '''
        raise NotImplementedError


    def get_distance(self):

         raise NotImplementedError

#----------------------------------------------------------
    # common functions



    def reset_task_param(self,                
                pro_gains = None, 
                pro_noise_stds = None,
                obs_gains = None, 
                obs_noise_stds = None,
                goal_radius=None
                ):
        
        pro_gains = torch.zeros(2)
        pro_gains[0] = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  #[proc_gain_vel]
        pro_gains[1] = torch.zeros(1).uniform_(self.gains_range[2], self.gains_range[3])  # [proc_gain_ang]
        
        pro_noise_stds=torch.zeros(2)
        pro_noise_stds[0]=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        pro_noise_stds[1]=torch.zeros(1).uniform_(self.std_range[2],self.std_range[3])
        
        obs_gains = torch.zeros(2)
        obs_gains[0] = torch.zeros(1).uniform_(self.gains_range[0], self.gains_range[1])  # [obs_gain_vel]
        obs_gains[1] = torch.zeros(1).uniform_(self.gains_range[2], self.gains_range[3])  # [obs_gain_ang]
        
        obs_noise_stds=torch.zeros(2)
        obs_noise_stds[0]=torch.zeros(1).uniform_(self.std_range[0], self.std_range[1])
        obs_noise_stds[1]=torch.zeros(1).uniform_(self.std_range[2],self.std_range[3])
        
        goal_radius = torch.zeros(1).uniform_(self.goal_radius_range[0], self.max_goal_radius)

        phi=torch.cat([pro_gains,obs_gains,pro_noise_stds,obs_noise_stds,goal_radius])

        phi[0:2]=pro_gains if pro_gains is not None else phi[0:2]
        phi[2:4]=pro_noise_stds if pro_noise_stds is not None else phi[2:4]
        phi[4:6]=obs_gains if obs_gains is not None else phi[4:6]
        phi[6:8]=obs_noise_stds if obs_noise_stds is not None else phi[6:8]
        phi[8]=goal_radius if goal_radius is not None else phi[8]

        return phi

    def reached_goal(self):
        # use real location
        _,distance=self.get_distance(s=self.s)
        reached_bool= (distance<=self.phi[-1])
        return reached_bool

    def if_agent_stop(self,a):
        terminal_vel = self.terminal_vel
        stop = (torch.norm(torch.tensor(a, dtype=torch.float64)) <= terminal_vel)
        if stop:
            terminal= True
        else:
            terminal= False

        return terminal


#-------------------------------------------------------
    # optional functions

    def logger(self):
        '''
        save the self.x, self.o, self.b, self.a for time t. 
        '''
        pass

    def assign_task_param(self,phi):
        '''
        assign the task parameter phi to the environment.
        note this phi is used for state dynamic, not the same as agent belief dynamic theta
        '''
        self.presist_phi=True
        self.phi=phi
        self.reset()

    def fetch_phi(self):
        '''
        fetch the task parameter phi 
        '''
        pass
        


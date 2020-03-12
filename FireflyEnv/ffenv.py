import gym
from numpy import pi
import numpy as np
from gym import spaces
from gym.utils import seeding
from FireflyEnv.firefly_task import Model
from DDPGv2Agent.belief_step import BeliefStep
from FireflyEnv.plotter_gym import Render
from DDPGv2Agent.rewards import *


class FireflyEnv(gym.Env): #, proc_noise_std = PROC_NOISE_STD, obs_noise_std =OBS_NOISE_STD):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }

    def __init__(self,arg):
        '''
        state-observation-blief-action-next state

        step:  action -> new state-observation-blief    
        

        reset: init new state
        '''
        super(FireflyEnv, self).__init__()
        low = np.append([0., -pi, -1., -1., 0.], -10*np.ones(16))
        high = np.append([10., pi, 1., 1., 100.], 10*np.ones(16)) #low[-1], high[-1] = 0., 100.
        # Define action and observation space
        # They must be gym.spaces objects
        # box and discrete are most common
        self.action_space = spaces.Box(-np.ones(2), np.ones(2), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.model = Model(arg) # environment
        self.belief = BeliefStep(arg) # agent
        self.action_dim = self.model.action_dim
        self.state_dim = self.model.state_dim
        #self.reset(gains_range, std_range, goal_radius_range)
        self.rendering = Render()



    def step(self, action):
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
        next_x, reached_target = self.model(x, action.view(-1))  # track true next_x of monkey
        next_ox = self.belief.observations(next_x)  # observation
        next_b, info = self.belief(b, next_ox, action, self.model.box)  # belief next state, info['stop']=terminal # reward only depends on belief
        next_state = self.belief.Breshape(next_b, t, theta)  # state used in policy is different from belief
        reward = return_reward(episode, info, reached_target, next_b, self.model.goal_radius, REWARD, finetuning)

        return next_x, reached_target, next_b, reward, info, next_state, next_ox
        
        return self.state, reward, done, info

    def reset(self):
        '''
        # retrun obs
        two ling of reset here:
        reset the episode progain, pronoise, goal radius, 
        state x, including [px, py, ang, vel, ang_vel]
        reset time

        then, reset belief:


        finally, return state
        '''


        x, pro_gains, pro_noise_ln_vars, goal_radius = self.model.reset(gains_range, noise_range, goal_radius_range)
        # detail of model reset
        self.pro_gains = torch.zeros(2)
        self.pro_noise_ln_vars = torch.zeros(2)
        if pro_gains is None: # is none, so assign value?
            self.pro_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  #[proc_gain_vel]
            self.pro_gains[1] = torch.zeros(1).uniform_(gains_range[2],
                                                        gains_range[3])  # [proc_gain_ang]
        else:
            self.pro_gains = pro_gains
        if pro_noise_ln_vars is None:

            self.pro_noise_ln_vars[0] = -1 * sample_exp(-noise_range[1], -noise_range[0]) #[proc_vel_noise]
            self.pro_noise_ln_vars[1] = -1 * sample_exp(-noise_range[3], -noise_range[2]) #[proc_ang_noise]
        else:
            self.pro_noise_ln_vars = pro_noise_ln_vars
        if goal_radius is None:
            self.max_goal_radius = min(self.max_goal_radius + self.GOAL_RADIUS_STEP, goal_radius_range[1])
            #self.goal_radius = torch.zeros(1).uniform_(self.min_goal_radius, goal_radius_range[1])
            self.goal_radius = torch.zeros(1).uniform_(goal_radius_range[0], self.max_goal_radius)
        else:
            self.goal_radius = goal_radius

        self.time = torch.zeros(1)
        min_r = self.goal_radius.item()
        r = torch.zeros(1).uniform_(min_r, self.box)  # GOAL_RADIUS, self.box is world size
        loc_ang = torch.zeros(1).uniform_(-pi, pi) # location angel: to determine initial location
        px = r * torch.cos(loc_ang)
        py = r * torch.sin(loc_ang)
        rel_ang = torch.zeros(1).uniform_(-pi/4, pi/4)
        ang = rel_ang + loc_ang + pi # heading angle of monkey, pi is added in order to make the monkey toward firefly
        ang = range_angle(ang)
        vel = torch.zeros(1)
        ang_vel = torch.zeros(1)
        x = torch.cat([px, py, ang, vel, ang_vel])
        return x, self.pro_gains, self.pro_noise_ln_vars, self.goal_radius

        
        
        















        b, state, obs_gains, obs_noise_ln_vars = self.belief.reset(x, torch.zeros(1) , pro_gains, pro_noise_ln_vars, goal_radius, gains_range, noise_range)
        # detail of belief reset
                self.pro_gains = pro_gains
        self.pro_noise_ln_vars = pro_noise_ln_vars
        self.goal_radius = goal_radius

        self.obs_gains = torch.zeros(2)
        self.obs_noise_ln_vars = torch.zeros(2)

        if obs_gains is None:
            self.obs_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  # [obs_gain_vel]
            self.obs_gains[1] = torch.zeros(1).uniform_(gains_range[2], gains_range[3])  # [obs_gain_ang]
        else:
            self.obs_gains = obs_gains

        if obs_noise_ln_vars is None:
            self.obs_noise_ln_vars[0] = -1 * sample_exp(-noise_range[1], -noise_range[0]) # [obs_vel_noise]
            self.obs_noise_ln_vars[1] = -1 * sample_exp(-noise_range[3], -noise_range[2]) # [obs_ang_noise]
        else:
            self.obs_noise_ln_vars = obs_noise_ln_vars


        self.theta = (self.pro_gains, self.pro_noise_ln_vars, self.obs_gains, self.obs_noise_ln_vars, self.goal_radius)
        self.P = torch.eye(5) * 1e-8 # change 4 to size function
        self.b = x, self.P  # belief=x because is not move yet
        self.state = self.Breshape(self.b, time, self.theta)

        return self.b, self.state, self.obs_gains, self.obs_noise_ln_vars

        return self.state










    def __del__(self):
        if self.enable_render is True:
            self.maze_view.quit_game()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def Brender(self, b, x, WORLD_SIZE, GOAL_RADIUS):
        bx, P = b
        goal = torch.zeros(2)
        self.rendering.render(goal, bx.view(1,-1), P, x.view(1,-1), WORLD_SIZE, GOAL_RADIUS)

    def render(self, mode='human'):
        self.rendering.render(mode)

    def get_position(self, x):
        pos = x.view(-1)[:2]
        r = torch.norm(pos).item()
        return pos, r

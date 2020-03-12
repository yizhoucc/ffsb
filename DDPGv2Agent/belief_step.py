"""
belief_step.py
This code uses the polar coordinate
state = torch.cat([vel, ang_vel, r, ang, vecL, time])
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .rewards import *
from .terminal import *

from FireflyEnv.env_utils import is_pos_def, vectorLowerCholesky, sample_exp, range_angle
from FireflyEnv.firefly_task import dynamics
#from FireflyEnv.plotter_gym import Render


class BeliefStep(nn.Module):
    def __init__(self, arg):
        super(self.__class__, self).__init__()

        self.dt = arg.DELTA_T
        self.P = torch.eye(5) * 1e-8
        #self.goal_radius = arg.GOAL_RADIUS
        self.terminal_vel = arg.TERMINAL_VEL
        self.episode_len = arg.EPISODE_LEN
        self.episode_time = arg.EPISODE_LEN * self.dt
        #self.rendering = Render()
        return

    def reset(self, x, time, pro_gains, pro_noise_ln_vars, goal_radius, gains_range, noise_range, obs_gains = None, obs_noise_ln_vars = None):

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

        """    
            
            

        if obs_noise_stds is None:
            self.obs_noise_stds[0] = torch.zeros(1).uniform_(std_range[0], std_range[1])  # [obs_vel_noise]
            self.obs_noise_stds[1] = torch.zeros(1).uniform_(std_range[2], std_range[3])  # [obs_ang_noise]
        else:
            self.obs_noise_stds = obs_noise_stds
            
        """

        self.theta = (self.pro_gains, self.pro_noise_ln_vars, self.obs_gains, self.obs_noise_ln_vars, self.goal_radius)


        self.P = torch.eye(5) * 1e-8 # change 4 to size function
        self.b = x, self.P  # belief=x because is not move yet
        self.state = self.Breshape(self.b, time, self.theta)

        return self.b, self.state, self.obs_gains, self.obs_noise_ln_vars



    def forward(self,  b, ox, a, box):
        I = torch.eye(5)

        # Q matrix, pro_noise 
        Q = torch.zeros(5, 5)
        Q[-2:, -2:] = torch.diag(torch.exp(self.pro_noise_ln_vars)) # variance of vel, ang_vel
        # why no sqrt here? because 2 noise will * each other?
        
        # R matrix, observe noise
        R = torch.diag(torch.exp(self.obs_noise_ln_vars))

        # H matrix
        H = torch.zeros(2, 5)
        H[:, -2:] = torch.diag(self.obs_gains)

        # Extended Kalman Filter
        pre_bx_, P = b
        bx_ = dynamics(pre_bx_, a.view(-1), self.dt, box, self.pro_gains, self.pro_noise_ln_vars)
        bx_ = bx_.t() # make a column vector
        A = self.A(bx_) # after dynamics
        P_ = A.mm(P).mm(A.t())+Q # P_ = APA^T+Q
        if not is_pos_def(P_):
            print("P_:", P_)
            print("P:", P)
            print("A:", A)
            APA = A.mm(P).mm(A.t())
            print("APA:", APA)
            print("APA +:", is_pos_def(APA))
        error = ox - self.observations(bx_)
        S = H.mm(P_).mm(H.t()) + R # S = HPH^T+R
        K = P_.mm(H.t()).mm(torch.inverse(S)) # K = PHS^-1
        bx = bx_ + K.matmul(error)
        I_KH = I - K.mm(H)
        P = I_KH.mm(P_)

        if not is_pos_def(P):
            print("here")
            print("P:", P)
            P = (P + P.t()) / 2 + 1e-6 * I  # make symmetric to avoid computational overflows

        bx = bx.t() #return to a row vector
        b = bx.view(-1), P  # belief


        # terminal check
        terminal = self._isTerminal(bx, a) # check the monkey stops or not
        return b, {'stop': terminal}


    def observations(self, x): 
        # observation of velocity and angle have gain and noise, but no noise of position
        on = torch.sqrt(torch.exp(self.obs_noise_ln_vars)) * torch.randn(2) # on is observation noise
        vel, ang_vel = torch.split(x.view(-1),1)[-2:] # 1,5 to vector and take last two

        ovel = self.obs_gains[0] * vel + on[0] # observe velocity
        oang_vel = self.obs_gains[1] * ang_vel + on[1]
        ox = torch.stack((ovel, oang_vel)) # observed x
        return ox

    def observations_mean(self, x): 
        '''
        # observation without noise, but stilll has gain 
        '''
        vel, ang_vel = torch.split(x.view(-1),1)[-2:]

        ovel = self.obs_gains[0] * vel
        oang_vel = self.obs_gains[1] * ang_vel
        ox = torch.stack((ovel, oang_vel))
        return ox

    def A(self, x_): # F in wiki
        # retrun A matrix, A*Pt-1*AT+Q to predict Pt
        dt = self.dt
        px, py, ang, vel, ang_vel = torch.split(x_.view(-1),1)

        A_ = torch.zeros(5, 5)
        A_[:3, :3] = torch.eye(3)
        A_[0, 2] = - vel * torch.sin(ang) * dt
        A_[1, 2] = vel * torch.cos(ang) * dt
        return A_



    def Breshape(self, b, time, theta): # reshape belief for policy
        pro_gains, pro_noise_ln_vars, obs_gains, obs_noise_ln_vars, goal_radius = theta # unpack the theta
        x, P = b # unpack the belief
        px, py, ang, vel, ang_vel = torch.split(x.view(-1), 1) # unpack states
        r = torch.norm(torch.cat([px, py])).view(-1) # what is r? relative distance to firefly
        rel_ang = ang - torch.atan2(-py, -px).view(-1) # relative angel
        rel_ang = range_angle(rel_ang) # resize relative angel into -pi pi range.
        vecL = vectorLowerCholesky(P) # take the lower triangle
        state = torch.cat([r, rel_ang, vel, ang_vel, time, vecL, pro_gains.view(-1), pro_noise_ln_vars.view(-1), obs_gains.view(-1), obs_noise_ln_vars.view(-1), torch.ones(1)*goal_radius]) # original
        #state = torch.cat([r, rel_ang, vel, ang_vel]) #, time, vecL]) #simple

        return state.view(1, -1)
    """
    def get_reward(self, b):
        bx, P = b
        reward = rewardFunc(rew_std, bx.view(-1), P, scale=10) # reward currently only depends on belief not action
        return reward
    """

    def _isTerminal(self, x, a, log=True):
        terminal_vel = self.terminal_vel
        terminal = is_terminal_action(a, terminal_vel)
        return terminal.item() == 1

    def get_position(self, b):
        bx, P = b
        r = bx.view(-1)[2]
        ang = bx.view(-1)[3]
        px = r * torch.cos(ang)
        py = r * torch.sin(ang)

        pos = torch.cat([px, py])
        return pos, r

    def render(self, b):
        bx, P = b
        goal = torch.zeros(2)
        self.rendering.render(goal, bx.view(1,-1), P)



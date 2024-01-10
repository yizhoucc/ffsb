
import gym
from gym import spaces
from numpy import pi
import numpy as np
from firefly_task.env_utils import *




class FireFlyReady(gym.Env, torch.nn.Module):

    '''
    1115, based on the firefly paper version. 
    serveral key modification.

    1 always use egocentric coord. the egocentric coord is easier for the animal, and the representation could be easier to convert to eye coord too.

    2 fix the observation bias. assume the observation is always reliable if there is no noise. that is, observation is noisy but no bias.


    '''
    def __init__(self, arg=None, kwargs=None):
        super(FireFlyReady, self).__init__()
        low = -np.inf
        high = np.inf
        self.arg = arg
        self.min_distance = arg.goal_distance_range[0]
        self.max_distance = arg.goal_distance_range[1]
        self.min_angle = -pi/5
        self.max_angle = pi/5
        self.terminal_vel = arg.terminal_vel
        self.episode_len = arg.episode_len
        self.dt = arg.dt
        self.reward = arg.reward_amount
        self.cost_scale = arg.cost_scale
        self.presist_phi = arg.presist_phi
        self.agent_knows_phi = arg.agent_knows_phi # for training
        self.phi = None
        self.goal_radius_range = arg.goal_radius_range
        self.gains_range = arg.gains_range
        self.std_range = arg.std_range
        self.mag_action_cost_range = arg.mag_action_cost_range
        self.dev_action_cost_range = arg.dev_action_cost_range
        self.dev_v_cost_range = arg.dev_v_cost_range
        self.dev_w_cost_range = arg.dev_w_cost_range
        self.init_uncertainty_range = arg.init_uncertainty_range
        self.previous_v_range = [0, 1]
        self.trial_counter = 0
        self.recent_rewarded_trials = 0
        self.recent_timeup_trials = 0
        self.recent_skipped_trials = 0
        self.recent_trials = 0
        self.debug = 0
        self.noise_scale = 1
        self.reward_ratio = 1
        obsdim = self.reset().shape[-1]
        self.action_space = spaces.Box(
            low=-1., high=1., shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(1, obsdim), dtype=np.float32)

    def reset(self,
              pro_gains=None,
              pro_noise_stds=None,
              obs_noise_stds=None,
              phi=None,
              theta=None,
              goal_position=None,
              vctrl=None,
              wctrl=None,
              obs_traj=None,
              pro_traj=None,
              ):
        if not self.presist_phi and phi is not None:  # when given phi upon reset, use this
            self.phi = phi
        elif not self.presist_phi and phi is None:  # normal training, random a new phi for each trial
            self.phi = self.reset_task_param(
                pro_gains=pro_gains, pro_noise_stds=pro_noise_stds, obs_noise_stds=obs_noise_stds)
        elif self.presist_phi and phi is not None:
            self.phi = phi

        if theta is not None:
            self.theta = theta
        else:
            self.theta = self.phi if self.agent_knows_phi else self.reset_task_param(
                pro_gains=pro_gains, pro_noise_stds=pro_noise_stds, obs_noise_stds=obs_noise_stds)
        self.unpack_theta()
        self.sys_vel = torch.zeros(1)
        self.stop = False
        self.agent_start = False
        self.trial_timer = torch.zeros(1)
        self.trial_sum_cost = 0.
        self.trial_sum_reward = 0.
        self.rewardgiven = False
        vctrl = vctrl if vctrl is not None else abs(
            torch.zeros(1).normal_(0., 0.3))
        wctrl = wctrl if wctrl is not None else torch.zeros(1).normal_(0., 0.3)
        # vctrl=torch.zeros(1)
        # wctrl=torch.zeros(1)
        self.reset_state(goal_position=goal_position, vctrl=vctrl, wctrl=wctrl)
        self.reset_belief()
        self.o = torch.tensor([[0], [0]])
        self.decision_info = self.wrap_decision_info(
            b=self.b, previous_action=self.previous_action, time=self.trial_timer, task_param=self.theta)
        if self.debug:
            self.obs_traj = obs_traj
            self.pro_traj = pro_traj
            self.trial_costs = []
            self.trial_rewards = []
            self.trial_actions = []
            self.ss = [self.s.clone()]
            self.bs = [self.b.clone()]
            self.covs = [self.P.clone()]
            self.actions = []
        return self.decision_info.view(1, -1)

    def reset_state(self, goal_position=None, vctrl=0., wctrl=0.):
        if goal_position is not None:  # use the given position
            self.goalx = goal_position[0]*torch.tensor(1.0)
            self.goaly = goal_position[1]*torch.tensor(1.0)
        else:  # random new position
            distance = torch.zeros(1).uniform_(
                self.min_distance, self.max_distance)
            angle = torch.zeros(1).uniform_(self.min_angle, self.max_angle)
            self.goalx = torch.cos(angle)*distance
            self.goaly = torch.sin(angle)*distance

        self.previous_action = torch.tensor([[vctrl, wctrl]])
        self.s = torch.tensor(
            [[0.],
             [0.],
                [0.],
                [vctrl*self.phi[0]],
                [wctrl*self.phi[1]]])

    def reset_belief(self, vctrl=0., wctrl=0.):
        self.P = torch.eye(5) * 1e-8
        self.P[0, 0] = (self.theta[9]*0.05)**2  # sigma xx
        self.P[1, 1] = (self.theta[10]*0.05)**2  # sigma yy
        self.b = torch.tensor(
            [[torch.distributions.Normal(0, torch.ones(1)).sample()*0.05*self.theta[9]],
             [torch.distributions.Normal(
                 0, torch.ones(1)).sample()*0.05*self.theta[10]],
                [0.],
                [vctrl*self.theta[0]],
                [wctrl*self.theta[1]]])

    def wrap_decision_info(self,
                           b=None,
                           P=None,
                           previous_action=None,
                           time=None,
                           task_param=None):
        task_param = task_param if task_param is not None else self.theta
        b = b if b is not None else self.b
        P = P if P is not None else self.P
        time = time if time is not None else self.trial_timer
        previous_action = previous_action if previous_action is not None else self.previous_action
        prevv, prevw = torch.split(previous_action.view(-1), 1)
        px, py, angle, v, w = torch.split(b.view(-1), 1)
        relative_distance = torch.sqrt(
            (self.goalx-px)**2+(self.goaly-py)**2).view(-1)
        relative_angle = torch.atan(
            (self.goaly-py)/(self.goalx-px)).view(-1)-angle
        relative_angle = torch.clamp(relative_angle, -pi, pi)
        vecL = bcov2vec(P)
        decision_info = torch.cat([
            relative_distance,
            relative_angle,
            v,
            w,
            time,
            prevv,
            prevw,
            vecL,
            task_param.view(-1)])
        return decision_info.view(1, -1)

    def reset_task_param(self,
                         pro_gains=None,
                         pro_noise_stds=None,
                         obs_noise_stds=None,
                         goal_radius=None,
                         dev_v_cost_factor=None,
                         dev_w_cost_factor=None,
                         inital_x_std=None,
                         inital_y_std=None,
                         ):
        _prov_gains = torch.zeros(1).uniform_(
            self.gains_range[0], self.gains_range[1])
        _prow_gains = torch.zeros(1).uniform_(
            self.gains_range[2], self.gains_range[3])
        _prov_noise_stds = torch.zeros(1).uniform_(
            self.std_range[0], self.std_range[1])
        _prow_noise_stds = torch.zeros(1).uniform_(
            self.std_range[2], self.std_range[3])
        _obsv_noise_stds = torch.zeros(1).uniform_(
            self.std_range[0], self.std_range[1])
        _obsw_noise_stds = torch.zeros(1).uniform_(
            self.std_range[2], self.std_range[3])
        _goal_radius = torch.zeros(1).uniform_(
            self.goal_radius_range[0], self.goal_radius_range[1])
        _dev_v_cost_factor = torch.zeros(1).uniform_(
            self.dev_v_cost_range[0], self.dev_v_cost_range[1])
        _dev_w_cost_factor = torch.zeros(1).uniform_(
            self.dev_w_cost_range[0], self.dev_w_cost_range[1])
        _inital_x_std = torch.zeros(1).uniform_(
            self.init_uncertainty_range[0], self.init_uncertainty_range[1])
        _inital_y_std = torch.zeros(1).uniform_(
            self.init_uncertainty_range[0], self.init_uncertainty_range[1])
        phi = torch.cat([
            _prov_gains,
            _prow_gains,
            _prov_noise_stds,
            _prow_noise_stds,
            _obsv_noise_stds,
            _obsw_noise_stds,
            _goal_radius,
            _dev_v_cost_factor,
            _dev_w_cost_factor,
            _inital_x_std,
            _inital_y_std,])
        phi[0] = pro_gains[0] if pro_gains is not None else phi[0]
        phi[1] = pro_gains[1] if pro_gains is not None else phi[1]
        phi[2] = pro_noise_stds[0] if pro_noise_stds is not None else phi[2]
        phi[3] = pro_noise_stds[1] if pro_noise_stds is not None else phi[3]
        phi[4] = obs_noise_stds[0] if obs_noise_stds is not None else phi[4]
        phi[5] = obs_noise_stds[1] if obs_noise_stds is not None else phi[5]
        phi[6] = goal_radius if goal_radius is not None else phi[6]
        phi[7] = dev_v_cost_factor if dev_v_cost_factor is not None else phi[7]
        phi[8] = dev_w_cost_factor if dev_w_cost_factor is not None else phi[8]
        phi[9] = inital_x_std if inital_x_std is not None else phi[9]
        phi[10] = inital_y_std if inital_y_std is not None else phi[10]
        return phi

    def step(self, action, next_state=None, predictiononly=False):
        action = torch.tensor(action).reshape(1, -1)
        self.a = action
        _, self.prev_d = self.get_distance(state=self.b)
        self.trial_timer += 1
        self.sys_vel = torch.norm(action)
        if not self.if_agent_stop() and not self.agent_start:
            # print('start')
            self.agent_start = True
        if self.agent_start and self.if_agent_stop():
            # print('stop')
            self.stop = True
        # if self.debug:
        end_current_ep = self.if_agent_stop() or self.trial_timer >= self.episode_len
        # else:
        #     end_current_ep= (self.if_agent_stop() and self.prev_d<3*self.goal_r) or self.trial_timer>=self.episode_len or self.prev_d>self.max_distance
        # end_current_ep=self.stop or self.trial_timer>=self.episode_len

        # dynamic
        if next_state is None:
            self.s = self.state_step(action, self.s)
        else:
            self.s = next_state
        self.o = self.observations(self.s)
        self.b, self.P = self.belief_step(
            self.b, self.P, self.o, action, predictiononly=predictiononly)
        self.decision_info = self.wrap_decision_info(
            previous_action=action, task_param=self.theta)
        # eval
        reward, cost = self.caculate_reward()
        self.trial_sum_cost += cost
        self.trial_sum_reward += reward
        if end_current_ep:
            signal = (
                (1-self.reward_ratio)*(self.trial_sum_reward -
                                       self.trial_sum_cost)/(self.trial_timer+5)
                + self.reward_ratio*(reward-cost))+(1-self.prev_d.item())
        else:
            signal = self.reward_ratio*(reward-cost)
            if not self.debug:
                assert self.rewardgiven == False  # when not stop, reward should not be given
        if self.debug:
            self.trial_actions.append(action)
            self.trial_costs.append(cost)
            self.trial_rewards.append(reward)
            self.ss.append(self.s.clone())
            self.bs.append(self.b.clone())
            self.covs.append(self.P.clone())
            self.actions.append(action)
        if end_current_ep:
            if self.rewarded():  # eval based on belief.
                self.recent_rewarded_trials += 1
            elif self.skipped():
                self.recent_skipped_trials += 1
            elif self.trial_timer >= self.episode_len:
                self.recent_timeup_trials += 1
            self.recent_trials += 1
            self.trial_counter += 1
            # print(self.trial_counter)
            if self.trial_counter != 0 and self.trial_counter % 10 == 0:
                print('rewarded: ', self.recent_rewarded_trials, 'skipped: ', self.recent_skipped_trials,
                      'time up: ', self.recent_timeup_trials, 'out of: ', self.recent_trials)
                self.recent_rewarded_trials = 0
                self.recent_skipped_trials = 0
                self.recent_timeup_trials = 0
                self.recent_trials = 0
            # print(
            # 'distance, ', "{:.1f}".format((self.get_distance()[1]-self.goal_r).item()),
            # 'stop',self.stop,
            print('reward: {:.1f}, cost: {:.1f}'.format(
                self.trial_sum_reward, self.trial_sum_cost), 'dist {:.2f}'.format(self.prev_d.item()))
            # 'rate {:.1f}'.format((self.trial_sum_reward-self.trial_sum_cost)/(self.trial_timer+5).item())
            # )
            # return self.decision_info, self.episode_reward-self.trial_sum_cost, end_current_ep, {}
        self.previous_action = action
        return self.decision_info, float(signal), end_current_ep, {}

    def state_step(self, a, s):  # update xy, then apply new action as new v and w
        next_s = self.update_state(s)
        px, py, angle, v, w = torch.split(next_s.view(-1), 1)
        noisev = torch.distributions.Normal(0, torch.ones(
            1)).sample()*self.pro_noisev*self.noise_scale
        noisew = torch.distributions.Normal(0, torch.ones(
            1)).sample()*self.pro_noisew*self.noise_scale
        if self.debug and self.pro_traj is not None:
            noisev = self.pro_traj[int(
                self.trial_timer.item())]*self.noise_scale
            noisew = self.pro_traj[int(
                self.trial_timer.item())]*self.noise_scale
        # else:
        #     noisev=self.pro_traj[int(self.episode_time.item())]#*self.pro_noise
        #     noisew=self.pro_traj[int(self.episode_time.item())]#*self.pro_noise
        v = self.pro_gainv * torch.tensor(1.0)*a[0, 0] + noisev
        w = self.pro_gainw * torch.tensor(1.0)*a[0, 1] + noisew
        next_s = torch.cat((px, py, angle, v, w)).view(1, -1)
        return next_s.view(-1, 1)

    def update_state(self, state):  # use v and w to update x y and heading
        px, py, angle, v, w = torch.split(state.view(-1), 1)
        if v <= 0:
            pass
        elif w == 0:
            px = px + v*self.dt * torch.cos(angle)
            py = py + v*self.dt * torch.sin(angle)
        else:
            px = px-torch.sin(angle)*(v/w-(v*torch.cos(w*self.dt)/w)) + \
                torch.cos(angle)*((v*torch.sin(w*self.dt)/w))
            py = py+torch.cos(angle)*(v/w-(v*torch.cos(w*self.dt)/w)) + \
                torch.sin(angle)*((v*torch.sin(w*self.dt)/w))
        angle = angle + w * self.dt
        angle = torch.clamp(angle, -pi, pi)
        # px = torch.clamp(px, -self.max_distance, self.max_distance)
        # py = torch.clamp(py, -self.max_distance, self.max_distance)
        next_s = torch.stack((px, py, angle, v, w))
        return next_s.view(-1, 1)

    #  v2
    # def caculate_rewardv2(self):
        # cost=self.action_cost(self.a, self.previous_action)
        # _,d= self.get_distance(state=self.b)
        # if self.stop:
        #     if d<=self.goal_r: # stop and reached goal
        #         reward=torch.tensor([1.])*self.reward
        #     else: # stop but not reached goal
        #         if self.goal_r/d>1/3: # if agent is somewhat near the goal
        #             reward = (self.goal_r/d)**1*self.reward*0.1*(1-self.cost_scale)
        #         else: # seems the agent gives up
        #             reward=torch.tensor([0.])
        #             cost+=0.1*(1-self.cost_scale)
        #         if reward>self.reward or reward<0:
        #             raise RuntimeError
        # else:
        #     # _,d= self.get_distance()
        #     # reward=(self.goal_r/d)**2
        #     reward=torch.tensor([0.])
        # reward+= (1-self.cost_scale)*self.reward*min(self.goal_r/d,1)*0.01

        # v2
    def caculate_reward(self):
        reward = torch.tensor([0.])
        cost = self.action_cost(self.a, self.previous_action)

        if (self.if_agent_stop() and self.prev_d < 4*self.goal_r):
            self.rewardgiven = True
            rew_std = self.goal_r/2
            mu = torch.Tensor([self.goalx, self.goaly])-self.b[:2, 0]
            R = torch.eye(2)*rew_std**2
            P = self.P[:2, :2]
            S = R+P
            if not is_pos_def(S):
                print('R+P is not positive definite!')
            alpha = -0.5 * mu @ S.inverse() @ mu.t()
            reward = torch.exp(alpha) / 2 / pi / torch.sqrt(S.det())
            # normalization -> to make max reward as 1
            mu_zero = torch.zeros(1, 2)
            alpha_zero = -0.5 * mu_zero @ R.inverse() @ mu_zero.t()
            reward_zero = torch.exp(alpha_zero) / 2 / pi / torch.sqrt(R.det())
            reward = reward/reward_zero
            if reward > 1.:
                print('reward is wrong!', reward)
                print('mu', mu)
                print('P', P)
                print('R', R)
            reward = self.reward * reward * \
                (self.episode_len-self.trial_timer)/self.episode_len
        else:
            _, d = self.get_distance(state=self.b)
            reward += (self.prev_d-d)*self.dt  # approaching reward
            # reward+=(1-self.cost_scale)*self.dt # stop reward, no matter what
        # _,d= self.get_distance(state=self.b)
        # reward+=(self.prev_d-d)*(1-self.cost_scale)*self.dt # approaching reward
        # reward-=torch.norm(self.a)*self.dt
        return reward.item(), cost.item()

    # v3
    def caculate_rewardv3(self):
        reward = torch.tensor([0.])
        cost = self.action_cost(self.a, self.previous_action)
        if self.rewarded():  # only evaluate reward when stop near enough.
            mu = self.b[:2, 0]-torch.Tensor([self.goalx, self.goaly])
            P = self.P[:2, :2]
            alpha = -0.5 * mu @ P.inverse() @ mu.t()
            reward_prob = torch.exp(alpha)
            reward = self.reward * reward_prob
        return reward.item(), cost.item()

    def action_cost(self, action, previous_action):
        # action is row vector
        action[0, 0] = 1.0 if action[0, 0] > 1.0 else action[0, 0]
        action[0, 1] = 1.0 if action[0, 1] > 1.0 else action[0, 1]
        action[0, 1] = -1.0 if action[0, 1] < -1.0 else action[0, 1]
        vcost = (action[0, 0]-previous_action[0, 0])**2*self.theta[7]
        wcost = (action[0, 1]-previous_action[0, 1])**2*self.theta[8]
        # if abs(action[0,0]-previous_action[0,0])<0.15:
        #     vcost=vcost*0
        # if abs(action[0,1]-previous_action[0,1])<0.15:
        #     wcost=wcost*0
        cost = vcost+wcost
        # mincost=1/10/10*20 #1/20^2 min cost per dt, for 20 dts.
        mincost = 2
        scalar = self.reward/mincost
        cost = cost*scalar
        return cost*self.cost_scale

    def unpack_theta(self):
        self.pro_gainv = self.phi[0]
        self.pro_gainw = self.phi[1]
        self.pro_noisev = self.phi[2]
        self.pro_noisew = self.phi[3]
        self.goal_r = self.phi[6]

        self.pro_gainv_hat = self.theta[0]
        self.pro_gainw_hat = self.theta[1]
        self.pro_noisev_hat = self.theta[2]
        self.pro_noisew_hat = self.theta[3]
        self.obs_noisev = self.theta[4]
        self.obs_noisew = self.theta[5]
        self.goal_r_hat = self.theta[6]
        self.Q = torch.zeros(5, 5)
        self.Q[3, 3] = self.pro_noisev_hat**2 if self.pro_noisev_hat**2 > 1e-4 else self.pro_noisev_hat**2+1e-4
        self.Q[4, 4] = self.pro_noisew_hat**2 if self.pro_noisew_hat**2 > 1e-4 else self.pro_noisew_hat**2+1e-4
        self.R = torch.zeros(2, 2)
        self.R[0, 0] = self.obs_noisev**2 if self.obs_noisev**2 > 1e-4 else self.obs_noisev**2+1e-4
        self.R[1, 1] = self.obs_noisew**2 if self.obs_noisew**2 > 1e-4 else self.obs_noisew**2+1e-4

    def if_agent_stop(self, sys_vel=None):
        sys_vel = sys_vel if sys_vel else self.sys_vel
        stop = (sys_vel <= self.terminal_vel)
        return stop

    def apply_action(self, state, action):
        px, py, angle, v, w = torch.split(state.view(-1), 1)
        v = action[0, 0]*self.pro_gainv_hat*torch.ones(1)
        w = action[0, 1]*self.pro_gainw_hat*torch.ones(1)
        next_s = torch.stack((px, py, angle, v, w))
        return next_s.view(-1, 1)

    def belief_step(self, previous_b, previous_P, o, a, task_param=None, predictiononly=False):
        task_param = self.theta if task_param is None else task_param
        I = torch.eye(5)
        H = torch.tensor([[0., 0., 0., 1, 0.], [0., 0., 0., 0., 1]])
        self.A = self.transition_matrix(previous_b, a)
        # prediction
        predicted_b = self.update_state(previous_b)
        predicted_b = self.apply_action(predicted_b, a)
        predicted_P = self.A@(previous_P)@(self.A.t())+self.Q

        if not is_pos_def(predicted_P):
            print('predicted not pos def')
            print('action ,', a)
            print('theta: ', task_param)
            print("predicted_P:", predicted_P)
            print('Q:', self.Q)
            print("previous_P:", previous_P)
            print("A:", self.A)
            APA = self.A@(previous_P)@(self.A.t())
            print("APA:", APA)
            print("APA +:", is_pos_def(APA))
        if predictiononly:
            return predicted_b, predicted_P

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
            print('task parameter : ', task_param)
            # make symmetric to avoid computational overflows
            P = (P + P.t()) / 2 + 1e-6 * I

        return b, P

    def transition_matrix(self, b, a):
        angle = b[2, 0]
        vel = b[3, 0]
        A = torch.zeros(5, 5)
        A[:3, :3] = torch.eye(3)
        # partial dev with theta
        A[0, 2] = - vel * torch.sin(angle) * self.dt*self.pro_gainv_hat
        A[1, 2] = vel * torch.cos(angle) * self.dt*self.pro_gainv_hat
        # partial dev with v
        A[0, 3] = torch.cos(angle) * self.dt*self.pro_gainv_hat
        A[1, 3] = torch.sin(angle) * self.dt*self.pro_gainv_hat
        # partial dev with w
        A[2, 4] = self.dt*self.pro_gainw_hat
        return A

    def reached_goal(self):
        # use real location
        _, distance = self.get_distance(state=self.s)
        reached_bool = (distance <= self.goal_r)
        return reached_bool

    def observations(self, state):
        s = state.clone()
        vw = s.view(-1)[-2:]  # 1,5 to vector and take last two
        noisev = torch.distributions.Normal(0, torch.tensor(
            1.)).sample()*self.obs_noisev*self.noise_scale
        noisew = torch.distributions.Normal(0, torch.tensor(
            1.)).sample()*self.obs_noisew*self.noise_scale
        if self.debug and self.obs_traj is not None:
            noise = self.obs_traj[int(
                self.trial_timer.item())]*self.noise_scale
            vw[0] = vw[0] + noise[0]
            vw[1] = vw[1] + noise[1]
            return vw.view(-1, 1)
        vw[0] = vw[0] + noisev
        vw[1] = vw[1] + noisew
        return vw.view(-1, 1)

    def obs_err(self,):
        return torch.distributions.Normal(0, torch.ones(2)).sample()*(self.theta[4:6].view(-1))

    def observations_mean(self, s):
        vw = s.view(-1)[-2:]
        return vw.view(-1, 1)

    def get_distance(self, state=None, distance2goal=True):
        state = self.s if state is None else state
        px, py, angle, v, w = torch.split(state.view(-1), 1)
        position = torch.stack([px, py])
        if distance2goal:
            relative_distance = torch.sqrt(
                (self.goalx-px)**2+(self.goaly-py)**2).view(-1)
        else:  # from oringin
            relative_distance = torch.sqrt((px)**2+(py)**2).view(-1)
        return position, relative_distance


    def in_target(self):
        '''
        
        '''
        return self.get_distance(state=self.b)[1] <= self.goal_r

    def rewarded(self):  
        '''
        agent stops,
        agent belief is in goal,
        '''
        return self.if_agent_stop() and (self.get_distance(state=self.b)[1] <= self.goal_r)

    def skipped(self):  
        '''
        
        '''
        return (self.get_distance(state=self.b, distance2goal=False)[1] <= self.goal_r) and self.trial_timer < 5

    def forward(self, action, task_param, state=None, giving_reward=None):
        action = torch.tensor(action).reshape(1, -1)
        self.a = action
        _, self.prev_d = self.get_distance(state=self.b)
        self.trial_timer += 1
        self.sys_vel = torch.norm(action)
        if not self.if_agent_stop(sys_vel=self.sys_vel) and not self.agent_start:
            self.agent_start = True
        if self.agent_start:
            self.stop = True if self.if_agent_stop(
                sys_vel=self.sys_vel) else False
        end_current_ep = self.rewarded() or self.trial_timer >= self.episode_len
        # dynamic
        if state is None:
            self.s = self.state_step(action, self.s)
        else:
            self.s = state
        self.o = self.observations(self.s)
        self.b, self.P = self.belief_step(self.b, self.P, self.o, action)
        self.decision_info = self.wrap_decision_info(
            previous_action=action, task_param=self.theta)

        self.previous_action = action
        return self.decision_info, end_current_ep

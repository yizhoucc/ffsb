#!/usr/bin/env python
# coding: utf-8

# In[25]:


from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random


# In[2]:


def init_weights(m, mean=0, std=0.1, bias=0):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean, std)
        nn.init.constant_(m.bias, bias)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, bias)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            else:
                raise ValueError()


# In[3]:


class Actor(nn.Module):
    def __init__(self, OBS_DIM, ACTION_DIM, TARGET_DIM, RNN_SIZE, BN_SIZE, FC_SIZE, RNN):
        super().__init__()
        self.OBS_DIM = OBS_DIM
        self.ACTION_DIM = ACTION_DIM
        
        self.rnn1 = RNN(input_size=OBS_DIM + ACTION_DIM, hidden_size=RNN_SIZE)
        self.l1 = nn.Linear(RNN_SIZE, BN_SIZE)
        self.l2 = nn.Linear(BN_SIZE + TARGET_DIM, FC_SIZE)
        self.l3 = nn.Linear(FC_SIZE, FC_SIZE)
        self.l4 = nn.Linear(FC_SIZE, ACTION_DIM)
        
        self.apply(init_weights)

    def forward(self, x, hidden_in, return_hidden=True): 
        x_v = x[:, :, :self.OBS_DIM + self.ACTION_DIM]
        x_tar = x[:, :, self.OBS_DIM + self.ACTION_DIM:]
        
        if hidden_in is None:
            x_v, hidden_out = self.rnn1(x_v)
        else:
            x_v, hidden_out = self.rnn1(x_v, hidden_in)
          
        x_v = self.l1(x_v)
        x = F.relu(self.l2(torch.cat([x_v, x_tar], dim=2)))
        x = F.relu(self.l3(x))
        x = torch.tanh(self.l4(x))
        if return_hidden:
            return x, hidden_out
        else:
            return x


class Critic(nn.Module):
    def __init__(self, OBS_DIM, ACTION_DIM, TARGET_DIM, RNN_SIZE, BN_SIZE, FC_SIZE, RNN):
        super().__init__()
        self.OBS_DIM = OBS_DIM
        self.ACTION_DIM = ACTION_DIM
        
        # Q1 architecture
        self.rnn1 = RNN(input_size=OBS_DIM + ACTION_DIM, hidden_size=RNN_SIZE)
        self.l1 = nn.Linear(RNN_SIZE, BN_SIZE)
        self.l2 = nn.Linear(BN_SIZE + TARGET_DIM + ACTION_DIM, FC_SIZE)
        self.l3 = nn.Linear(FC_SIZE, FC_SIZE)  
        self.l4 = nn.Linear(FC_SIZE, 1)
        
        # Q2 architecture
        self.rnn2 = RNN(input_size=OBS_DIM + ACTION_DIM, hidden_size=RNN_SIZE)
        self.l5 = nn.Linear(RNN_SIZE, BN_SIZE)
        self.l6 = nn.Linear(BN_SIZE + TARGET_DIM + ACTION_DIM, FC_SIZE)
        self.l7 = nn.Linear(FC_SIZE, FC_SIZE)
        self.l8 = nn.Linear(FC_SIZE, 1)
        
        self.apply(init_weights)

    def forward(self, x, u, hidden_in1=None, hidden_in2=None, return_hidden=False):
        x_v = x[:, :, :self.OBS_DIM + self.ACTION_DIM]
        x_tar = x[:, :, self.OBS_DIM + self.ACTION_DIM:]
        
        if hidden_in1 is None:
            x_v1, hidden_out1 = self.rnn1(x_v)
        else:
            x_v1, hidden_out1 = self.rnn1(x_v, hidden_in1)
        x_v1 = self.l1(x_v1)
        x1 = F.relu(self.l2(torch.cat([x_v1, x_tar, u], dim=2)))
        x1 = F.relu(self.l3(x1))
        x1 = self.l4(x1)
        
        if hidden_in2 is None:
            x_v2, hidden_out2 = self.rnn2(x_v)
        else:
            x_v2, hidden_out2 = self.rnn2(x_v, hidden_in2)
        x_v2 = self.l5(x_v2)
        x2 = F.relu(self.l6(torch.cat([x_v2, x_tar, u], dim=2)))
        x2 = F.relu(self.l7(x2))
        x2 = self.l8(x2)
        
        if return_hidden:
            return x1, x2, hidden_in1, hidden_out2
        else:
            return x1, x2
    
    def Q1(self, x, u):
        x_v = x[:, :, :self.OBS_DIM + self.ACTION_DIM]
        x_tar = x[:, :, self.OBS_DIM + self.ACTION_DIM:]
        
        x_v1, _ = self.rnn1(x_v)
        x_v1 = self.l1(x_v1)
        x1 = F.relu(self.l2(torch.cat([x_v1, x_tar, u], dim=2)))
        x1 = F.relu(self.l3(x1))
        x1 = self.l4(x1)
        
        return x1


# In[4]:


transition = namedtuple('transition', ('state', 'action', 'reward'))

class ReplayMemory():
    def __init__(self, MEMORY_SIZE, BATCH_SIZE):
        self.MEMORY_SIZE = MEMORY_SIZE
        self.SMALL_MEMORY_SIZE = int(MEMORY_SIZE / 10)
        self.BATCH_SIZE = BATCH_SIZE
        self.memory = []
        self.position = 0

    def push(self, *args):
        # a ring buffer
        if len(self.memory) < self.MEMORY_SIZE:
            self.memory.append(None)
            
        self.memory[self.position] = transition(*args)
        self.position = (self.position+1) % self.MEMORY_SIZE
        
    def sample(self):
        # 1. Sample a small memory.
        if len(self.memory) < self.SMALL_MEMORY_SIZE:
            small_memory = self.memory
        else:
            small_memory = random.sample(self.memory, self.SMALL_MEMORY_SIZE)
            
        # 2. Sample a trial length.
        traj_len = [traj.reward.shape[0] for traj in small_memory]
        traj_len = random.sample(traj_len, 1)[0]
        
        # 3. Get trials with same length.
        small_memory = [traj for traj in small_memory if traj.reward.shape[0] == traj_len]
        
        # 4. Sample a mini batch.
        batch = random.sample(small_memory, min(self.BATCH_SIZE, len(small_memory)))
        batch = transition(*zip(*batch))
        
        return batch
        
    def load(self, memory):          
        self.memory, self.position = memory
        
    def reset(self):
        self.memory = []
        self.position = 0


# In[5]:


class BeliefStep(nn.Module):
    def __init__(self, arg):
        super().__init__()
        self.STATE_DIM = arg.STATE_DIM
        self.OBS_DIM = arg.OBS_DIM
        self.obs_noise_range = arg.obs_noise_range
        
        self.H = torch.zeros(self.OBS_DIM, self.STATE_DIM)
        self.H[0, -2] = 1
        self.H[1, -1] = 1
        
    @property
    def obs_noise_range(self):
        return self._obs_noise_range
    
    @obs_noise_range.setter
    def obs_noise_range(self, value):
        self._obs_noise_range = [0, 0] if value is None else value

    def reset(self, pro_gains, obs_noise_std=None):
        self.obs_noise_std = obs_noise_std
        
        if self.obs_noise_std is None:
            self.obs_noise_std = torch.zeros(1).uniform_(
                                    self.obs_noise_range[0], 
                                    self.obs_noise_range[1]) * pro_gains

    def forward(self, x):
        zita = (self.obs_noise_std * torch.randn(self.OBS_DIM)).view([-1, 1])
        o_t = self.H @ x + zita
        
        return o_t


# In[6]:


class ActionNoise():
    def __init__(self, ACTION_DIM, mean, std):
        self.mu = torch.ones(ACTION_DIM) * mean
        self.std = std
        self.ACTION_DIM = ACTION_DIM

    def reset(self, mean, std):
        self.mu = torch.ones(self.ACTION_DIM) * mean
        self.std = std

    def noise(self):
        n = torch.randn(self.ACTION_DIM)
        return self.mu + self.std * n


# In[11]:


class Agent():
    def __init__(self, arg):
        self.__dict__ .update(arg.__dict__)
        self.data_path = self.data_path_

        self.actor = Actor(self.OBS_DIM, self.ACTION_DIM, self.TARGET_DIM, 
                           self.RNN_SIZE, self.BN_SIZE, self.FC_SIZE, self.RNN).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_actor.eval()
        self.actor_optim = self.optimzer(self.actor.parameters(), lr=self.lr, eps=self.eps)
        
        self.critic = Critic(self.OBS_DIM, self.ACTION_DIM, self.TARGET_DIM, 
                           self.RNN_SIZE, self.BN_SIZE, self.FC_SIZE, self.RNN).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.eval()
        self.critic_optim = self.optimzer(self.critic.parameters(), lr=self.lr, eps=self.eps)
        
        self.memory = ReplayMemory(arg.MEMORY_SIZE, arg.BATCH_SIZE)
        self.bstep = BeliefStep(arg)
        
        self.initial_episode = 0
        self.it = 0

    def select_action(self, state, hidden_in, action_noise=None):
        with torch.no_grad():
            action, hidden_out = self.actor(state, hidden_in)
            
        action = action.cpu()
        action_raw = action.clone()
        if (action_noise is not None) and (action_raw.abs() > self.TERMINAL_ACTION).any():
            action += action_noise.noise().view_as(action)

        return action.clamp(-1, 1), action_raw, hidden_out
    
    def target_smoothing(self, next_actions):
        mask_stop = (next_actions.view(-1, self.ACTION_DIM).abs().max(dim=1).values < self.TERMINAL_ACTION
                        ).view(-1, 1).repeat(1, self.ACTION_DIM).view_as(next_actions)
        mask_nonstop_pos = (next_actions > self.TERMINAL_ACTION) & (~mask_stop)
        mask_nonstop_neg = (next_actions < -self.TERMINAL_ACTION) & (~mask_stop)
        mask_nonstop_other = (next_actions.abs() < self.TERMINAL_ACTION) & (~mask_stop)

        next_actions[mask_stop] = (next_actions[mask_stop]                         + torch.zeros_like(next_actions[mask_stop]).normal_(
                                                mean=0, std=self.policy_noise)
                        ).clamp(-self.TERMINAL_ACTION, self.TERMINAL_ACTION)

        next_actions[mask_nonstop_pos] = (next_actions[mask_nonstop_pos]                         + torch.zeros_like(next_actions[mask_nonstop_pos]).normal_(
                                mean=0, std=self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
                        ).clamp(self.TERMINAL_ACTION, 1)

        next_actions[mask_nonstop_neg] = (next_actions[mask_nonstop_neg]                         + torch.zeros_like(next_actions[mask_nonstop_neg]).normal_(
                                mean=0, std=self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
                        ).clamp(-1, -self.TERMINAL_ACTION)

        next_actions[mask_nonstop_other] = (next_actions[mask_nonstop_other]                         + torch.zeros_like(next_actions[mask_nonstop_other]).normal_(
                                mean=0, std=self.policy_noise).clamp(-self.policy_noise_clip, self.policy_noise_clip)
                        ).clamp(-1, 1)
        
        return next_actions

    def update_parameters(self, batch):
        states = torch.cat(batch.state, dim=1)
        actions =  torch.cat(batch.action, dim=1)
        rewards = torch.cat(batch.reward, dim=1)
        dones = torch.zeros_like(rewards)
        dones[-1] = 1
        
        with torch.no_grad():
            # get next action and apply target policy smoothing
            next_states = torch.zeros_like(states)
            next_states[:-1] = states[1:]
            _, t1_hidden = self.target_actor(states[:1], hidden_in=None, return_hidden=True)
            next_actions = self.target_actor(next_states, hidden_in=t1_hidden, return_hidden=False)
            next_actions = self.target_smoothing(next_actions)
            
            # compute the target Q
            _, _, t1_hidden1, t1_hidden2 = self.target_critic(states[:1], actions[:1], return_hidden=True)
            target_Q1, target_Q2 = self.target_critic(next_states, next_actions, 
                                                      hidden_in1=t1_hidden1, hidden_in2=t1_hidden2)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1-dones) * self.GAMMA * target_Q

        # current Q estimates
        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # delay policy updates
        if self.it % self.POLICY_FREQ == 0:
            # define actor loss
            actor_loss = - self.critic.Q1(states, self.actor(states, hidden_in=None, return_hidden=False)).mean()
            
            # optimize the actor
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update target networks
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic, self.critic)
        else:
            actor_loss = torch.tensor([0])

        return actor_loss.detach().item(), critic_loss.detach().item()

    def learn(self):
        batch = self.memory.sample()
        loss_logs = self.update_parameters(batch)
        self.it += 1
        return loss_logs

    def save(self, save_memory, episode):
        file = self.data_path / f'{self.filename}-{episode}.pth.tar'
        
        state = {'actor_dict': self.actor.state_dict(),
                'critic_dict': self.critic.state_dict(),
                'target_actor_dict': self.target_actor.state_dict(),
                'target_critic_dict': self.target_critic.state_dict(),
                'actor_optimizer_dict': self.actor_optim.state_dict(),
                'critic_optimizer_dict': self.critic_optim.state_dict(),
                'episode': episode}
        
        if save_memory:
            state['memory'] = (self.memory.memory, self.memory.position)

        torch.save(state, file)

    def load(self, filename, load_memory, load_optimzer):
        self.filename = filename
        file = self.data_path / f'{self.filename}.pth.tar'
        state = torch.load(file)

        self.actor.load_state_dict(state['actor_dict'])
        self.critic.load_state_dict(state['critic_dict'])
        self.target_actor.load_state_dict(state['target_actor_dict'])
        self.target_critic.load_state_dict(state['target_critic_dict'])
        self.initial_episode = state['episode']
        
        if load_memory is True:
            self.memory.load(state['memory'])
        if load_optimzer is True:
            self.actor_optim.load_state_dict(state['actor_optimizer_dict'])
            self.critic_optim.load_state_dict(state['critic_optimizer_dict'])

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.TAU) + param.data * self.TAU)
            
    def mirror_traj(self, states, actions, mirrored_index=(1, 3, 4)):
        # state index 1: w; 3: action aw; 4: target x
        states_ = states.clone()
        states_[:, :, mirrored_index] = - states_[:, :, mirrored_index]
        # 1 of action indexes angular action aw
        actions_ = actions.clone()
        actions_[:, :, 1] = - actions_[:, :, 1]
        
        return states_, actions_

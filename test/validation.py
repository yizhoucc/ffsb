#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Config import Config
import pandas as pd
from FireflyEnv.neuralenv import Env
import torch


# In[ ]:


class Validation():
    def __init__(self, task, agent_name, validation_size=200):
        if 'noise' in task:
            self.task_sets = ['noise1', 'noise2', 'noise3']
        elif 'normal' in task:
            self.task_sets = ['gain1x']
        elif 'gain' in task:
            self.task_sets = ['gain1x', 'gain2x']
        elif 'perturbation' in task:
            self.task_sets = ['perturbation']
        else:
            raise ValueError('No such a task.')

        self.agent_name = agent_name
        self.validation_size = validation_size
        self.data = pd.DataFrame(columns=['episode', 'task', 
                                                     'reward_fraction', 'error_distance'])
        
    def __call__(self, agent, episode, get_self_action=True):
        if self.agent_name == 'LSTM':
            self.validation_lstm(agent, episode, get_self_action)
        elif self.agent_name == 'EKF':
            self.validation_ekf(agent, episode)
        else:
            raise ValueError('No such an agent.')
            
        return self.data
        
    def validation_lstm(self, agent, episode, get_self_action):
        for task in self.task_sets:
            if task == 'gain1x':
                arg = config.ConfigGain()
                arg.process_gain_range = [1.0, 1.0]
                env = Env(arg)
            elif task == 'gain2x':
                arg = config.ConfigGain()
                arg.process_gain_range = [2.0, 2.0]
                env = Env(arg)
            elif task == 'perturbation':
                arg = config.ConfigPerturb()
                env = Env(arg)
            elif task == 'noise1':
                arg = config.ConfigNoise()
                arg.obs_noise_range = [0, 0]
                agent.bstep.obs_noise_range = arg.obs_noise_range
                env = Env(arg)
            elif task == 'noise2':
                arg = config.ConfigNoise()
                arg.obs_noise_range = [0.5, 0.5]
                agent.bstep.obs_noise_range = arg.obs_noise_range
                env = Env(arg)
            elif task == 'noise3':
                arg = config.ConfigNoise()
                arg.obs_noise_range = [1, 1]
                agent.bstep.obs_noise_range = arg.obs_noise_range
                env = Env(arg)
                
            rewarded_number_log = 0
            dist_log = 0
            RNN_HIDDEN_SIZE = agent.actor.rnn1.hidden_size
            NUM_LAYERS = agent.actor.rnn1.num_layers
        
            for _ in range(self.validation_size):
                x = env.reset()
                agent.bstep.reset(env.pro_gains)
                last_action = torch.zeros([1, 1, arg.ACTION_DIM])

                state = torch.cat([x[-arg.OBS_DIM:].view(1, 1, -1), last_action,
                           env.target_position.view(1, 1, -1)], dim=2).to(arg.device)

                hidden1_in_policy = (torch.zeros((NUM_LAYERS, 1, RNN_HIDDEN_SIZE), device=arg.device), 
                                     torch.zeros((NUM_LAYERS, 1, RNN_HIDDEN_SIZE), device=arg.device))

                for t in range(arg.EPISODE_LEN):
                    action, _, hidden1_out_policy = agent.select_action(
                                                        state, hidden1_in_policy, action_noise=None)

                    next_x, reached_target, relative_dist = env(x, action, t)
                    next_ox = agent.bstep(next_x)

                    if get_self_action:
                        next_state = torch.cat([next_ox.view(1, 1, -1), action,
                                        env.target_position.view(1, 1, -1)], dim=2).to(arg.device)
                    else:
                        next_state = torch.cat([next_ox.view(1, 1, -1), next_ox.view(1, 1, -1),
                                        env.target_position.view(1, 1, -1)], dim=2).to(arg.device)

                    is_stop = (action.abs() < arg.TERMINAL_ACTION).all()

                    last_action = action
                    state = next_state
                    x = next_x
                    hidden1_in_policy = hidden1_out_policy

                    if is_stop:
                        break


                #log stuff 
                rewarded_number_log += int(reached_target & is_stop)
                dist_log += relative_dist.item()                  

            self.data = self.data.append(
                        pd.DataFrame({'episode': [episode], 'task': [task],
                                      'reward_fraction': [rewarded_number_log / self.validation_size], 
                                      'error_distance': [dist_log * arg.LINEAR_SCALE / self.validation_size]}), 
                                      ignore_index=True)
            
        
    def validation_ekf(self, agent, episode):
        for task in self.task_sets:
            if task == 'gain1x':
                arg = config.ConfigGain()
                arg.process_gain_range = [1.0, 1.0]
                env = Env(arg)
            elif task == 'gain2x':
                arg = config.ConfigGain()
                arg.process_gain_range = [2.0, 2.0]
                env = Env(arg)
            elif task == 'perturbation':
                arg = config.ConfigPerturb()
                env = Env(arg)
            elif task == 'noise1':
                arg = config.ConfigNoise()
                arg.obs_noise_range = [0, 0]
                agent.bstep.obs_noise_range = arg.obs_noise_range
                env = Env(arg)
            elif task == 'noise2':
                arg = config.ConfigNoise()
                arg.obs_noise_range = [0.5, 0.5]
                agent.bstep.obs_noise_range = arg.obs_noise_range
                env = Env(arg)
            elif task == 'noise3':
                arg = config.ConfigNoise()
                arg.obs_noise_range = [1, 1]
                agent.bstep.obs_noise_range = arg.obs_noise_range
                env = Env(arg)

            rewarded_number_log = 0
            dist_log = 0
            
            for _ in range(self.validation_size):
                x = env.reset()
                b, state = agent.bstep.reset(env.pro_gains, env.pro_noise_std, env.target_position)
                state = state.to(arg.device)

                for t in range(arg.EPISODE_LEN):
                    action, _ = agent.select_action(state, action_noise=None)
                    next_x, reached_target, relative_dist = env(x, action, t)
                    next_ox = agent.bstep.observation(next_x)
                    next_b = agent.bstep(b, next_ox, action, env.perturbation_vt, env.perturbation_wt)
                    next_state = agent.bstep.b_reshape(next_b).to(arg.device)

                    is_stop = (action.abs() < arg.TERMINAL_ACTION).all()

                    state = next_state
                    x = next_x
                    b = next_b

                    if is_stop:
                        break

                #log stuff 
                rewarded_number_log += int(reached_target & is_stop)
                dist_log += relative_dist.item()                  

            self.data = self.data.append(
                        pd.DataFrame({'episode': [episode], 'task': [task],
                                      'reward_fraction': [rewarded_number_log / self.validation_size], 
                                      'error_distance': [dist_log * arg.LINEAR_SCALE / self.validation_size]}), 
                                      ignore_index=True)


# In[ ]:




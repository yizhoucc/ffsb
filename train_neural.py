


import numpy as np
import time

import Config
arg = Config.ConfigGain()
from FireflyEnv import neuralenv
from agentLSTM import *
from validation import Validation

# initialize environment and agent
env = neuralenv.Env(arg)
agent = Agent(arg)
validator = Validation(arg.task, agent_name='LSTM')

# define exploration noise
init_expnoise_std = 0.5
noise = ActionNoise(arg.ACTION_DIM, mean=0, std=init_expnoise_std)

# Remove observation noise in the beginning to help learning in the early stage.
agent.bstep.obs_noise_range = [0, 0]


tot_t = 0
episode = agent.initial_episode
reward_log = []
rewarded_trial_log = []
step_log = []
actor_loss_log = 0
critic_loss_log = 0
dist_log = []

LOG_FREQ = 100
VALIDATION_FREQ = 500
decrease_lr = True
REPLAY_PERIOD = 4
PRE_LEARN_PERIOD = arg.BATCH_SIZE * 100

RNN_HIDDEN_SIZE = agent.actor.rnn1.hidden_size
NUM_LAYERS = agent.actor.rnn1.num_layers
get_self_action = True
enable_mirror_traj = False

# Start loop
while True:
    # initialize a trial
    cross_start_threshold = False
    reward = torch.zeros(1, 1, 1)
    
    x = env.reset()
    agent.bstep.reset(env.pro_gains)
    last_action = torch.zeros(1, 1, arg.ACTION_DIM)
    last_action_raw = last_action.clone()

    state = torch.cat([x[-arg.OBS_DIM:].view(1, 1, -1), last_action,
                       env.target_position.view(1, 1, -1)], dim=2).to(arg.device)

    hidden1_in_policy = (torch.zeros((NUM_LAYERS, 1, RNN_HIDDEN_SIZE), device=arg.device), 
                         torch.zeros((NUM_LAYERS, 1, RNN_HIDDEN_SIZE), device=arg.device))
    states = []
    actions = []
    rewards = []
    
    for t in range(arg.EPISODE_LEN):
        # 1. Check start threshold.
        if not cross_start_threshold and (last_action_raw.abs() > arg.TERMINAL_ACTION).any():
            cross_start_threshold = True

        # 2. Take an action based on current state 
        # and previous hidden & cell states of LSTM units.
        action, action_raw, hidden1_out_policy = agent.select_action(state, hidden1_in_policy, 
                                                                     action_noise=noise)

        # 3. Track next x in the environment.
        next_x, reached_target, relative_dist = env(x, action, t)

        # 4. Next observation given next x.
        next_ox = agent.bstep(next_x)

        if get_self_action:
            next_state = torch.cat([next_ox.view(1, 1, -1), action,
                                    env.target_position.view(1, 1, -1)], dim=2).to(arg.device)
        else:
            next_state = torch.cat([next_ox.view(1, 1, -1), next_ox.view(1, 1, -1),
                                    env.target_position.view(1, 1, -1)], dim=2).to(arg.device)

        # 5. Check whether stop.
        is_stop = (action.abs() < arg.TERMINAL_ACTION).all()

        # 6. Give reward if stopped.          
        if is_stop and cross_start_threshold:
            reward = env.return_reward(x, reward_mode='mixed')

        # 7. Append data.
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        # 8. Update timestep.
        last_action_raw = action_raw
        last_action = action
        state = next_state
        x = next_x
        hidden1_in_policy = hidden1_out_policy
        tot_t += 1

        # 9. Update model.
        if len(agent.memory.memory) > PRE_LEARN_PERIOD and tot_t % REPLAY_PERIOD == 0:
            #start_time = time.time()
            actor_loss, critic_loss = agent.learn()
            actor_loss_log += actor_loss
            critic_loss_log += critic_loss
            #time_all.append(time.time() - start_time)

        # 10. whether break.
        if is_stop and cross_start_threshold:
            break


    # End of one trial, store trajectory into buffer.
    states = torch.cat(states)
    actions = torch.cat(actions).to(arg.device)
    rewards = torch.cat(rewards).to(arg.device)
    agent.memory.push(states, actions, rewards) 

    if enable_mirror_traj:
        # store mirrored trajectories reflected along y-axis
        agent.memory.push(*agent.mirror_traj(states, actions), rewards) 
    
    #log stuff
    reward_log.append(reward.item())
    rewarded_trial_log.append(int(reached_target & is_stop))
    step_log.append(t + 1)
    dist_log.append(relative_dist.item())
        
    if episode % LOG_FREQ == LOG_FREQ - 1:
        print(f"t: {tot_t}, Ep: {episode}, action std: {noise.std:0.2f}")
        print(f"mean steps: {np.mean(step_log):0.3f}, "
               f"mean reward: {np.mean(reward_log):0.3f}, "
               f"rewarded fraction: {np.mean(rewarded_trial_log):0.3f}, "
               f"relative distance: {np.mean(dist_log) * arg.LINEAR_SCALE:0.3f}, "
               f"obs noise: {agent.bstep.obs_noise_range}")
        
        if decrease_lr and (validator.data.reward_fraction > 0.95).any():
            noise.reset(mean=0, std=0.4)
            agent.actor_optim.param_groups[0]['lr'] = arg.decayed_lr
            agent.critic_optim.param_groups[0]['lr'] = arg.decayed_lr
            decrease_lr = False
            print('Noise and learning rate are changed.')
            
        if noise.std == init_expnoise_std and np.mean(rewarded_trial_log) > 0.2:
            noise.reset(mean=0, std=0.5)
            agent.bstep.obs_noise_range = arg.obs_noise_range
            enable_mirror_traj = True
            agent.memory.reset()
                        
        reward_log = []
        rewarded_trial_log = []
        step_log = []
        actor_loss_log = 0
        critic_loss_log = 0
        dist_log = []
        
    # saving and validation
    if noise.std < init_expnoise_std and episode % VALIDATION_FREQ == VALIDATION_FREQ - 1:
        # save
        agent.save(save_memory=False, episode=episode)
            
        # validate
        validator(agent, episode, get_self_action=get_self_action
                 ).to_csv(arg.data_path / f'{arg.filename}.csv', index=False)
        agent.bstep.obs_noise_range = arg.obs_noise_range
        
    episode += 1












#--------------------------------------------------------

# from stable_baselines3.td3.policies import MlpPolicy, Actor
# import numpy as np
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# from FireflyEnv import ffacc_real
# import torch
# import time
# n_actions = 1
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))        
# modelname=None
# # modelname='frate'
# note='' 

# env1=ffacc_real.SmoothAction1d()
# env2=ffacc_real.Neural1d()

# from stable_baselines3 import TD3
# from typing import Any, Callable, Dict, List, Optional, Type, Union

# import gym
# import torch as th
# from torch import nn

# from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, register_policy
# from stable_baselines3.common.preprocessing import get_action_dim
# from stable_baselines3.common.torch_layers import (
#     BaseFeaturesExtractor,
#     FlattenExtractor,
#     NatureCNN,
#     create_mlp,
#     get_actor_critic_arch,
# )

# class LSTMActor(BasePolicy):

#     def __init__(
#         self,
#         observation_space: gym.spaces.Space,
#         action_space: gym.spaces.Space,
#         net_arch: List[int],
#         features_extractor: nn.Module,
#         features_dim: int,
#         activation_fn: Type[nn.Module] = nn.ReLU,
#         normalize_images: bool = True,
#     ):
#         super(LSTMActor, self).__init__(
#             observation_space,
#             action_space,
#             features_extractor=features_extractor,
#             normalize_images=normalize_images,
#             squash_output=True,
#         )

#         self.features_extractor = features_extractor
#         self.normalize_images = normalize_images
#         self.net_arch = net_arch
#         self.features_dim = features_dim
#         self.activation_fn = activation_fn

#         action_dim = get_action_dim(self.action_space)
#         actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
#         # Deterministic action
#         self.mu = nn.LSTM(features_dim,action_dim,num_layers=2,dropout=0.1,)

#     def _get_data(self) -> Dict[str, Any]:
#         data = super()._get_data()

#         data.update(
#             dict(
#                 net_arch=self.net_arch,
#                 features_dim=self.features_dim,
#                 activation_fn=self.activation_fn,
#                 features_extractor=self.features_extractor,
#             )
#         )
#         return data

#     def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
#         # assert deterministic, 'The TD3 actor only outputs deterministic actions'
#         features = self.extract_features(obs)
#         return self.mu(features)

#     def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
#         return self.forward(observation, deterministic=deterministic)



# class LSTMPolicy(MlpPolicy):

#     def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
#         actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
#         return LSTMActor(**actor_kwargs).to(self.device)




# if modelname is None:
#     ## behavioral
#     # model = TD3(MlpPolicy,
#     #     env1,
#     #     buffer_size=int(1e6),
#     #     batch_size=512,
#     #     learning_rate=7e-4,
#     #     learning_starts= 1000,
#     #     tau= 0.005,
#     #     gamma= 0.96,
#     #     train_freq = 4,
#     #     # gradient_steps = -1,
#     #     # n_episodes_rollout = 1,
#     #     action_noise= action_noise,
#     #     # optimize_memory_usage = False,
#     #     policy_delay = 6,
#     #     # target_policy_noise = 0.2,
#     #     # target_noise_clip = 0.5,
#     #     tensorboard_log = None,
#     #     # create_eval_env = False,
#     #     policy_kwargs = {'net_arch':[128,128],'activation_fn':torch.nn.LeakyReLU},
#     #     verbose = 0,
#     #     seed = 1,
#     #     device = "cpu",
#     #     )
#     # train_time=100000
#     # for i in range(1,11):  
#     #     env1.cost_scale=0.1
#     #     if i==1:
#     #         for j in range(1,11): 
#     #             namestr= ("trained_agent/1dtd3_{}_{}_{}_{}_{}".format(train_time,j,
#     #             str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#     #             ))
#     #             model.learn(train_time)
#     #             model.save(namestr)
#     #     namestr= ("trained_agent/1dtd3_{}_{}_{}_{}_{}".format(train_time,i,
#     #     str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#     #     ))
#     #     model.learn(train_time)
#     #     model.save(namestr)

#     policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
#     model = TD3(LSTMPolicy,
#         env1,
#         buffer_size=int(1e6),
#         batch_size=512,
#         learning_rate=7e-4,
#         learning_starts= 1000,
#         tau= 0.005,
#         gamma= 0.96,
#         train_freq = 4,
#         # gradient_steps = -1,
#         # n_episodes_rollout = 1,
#         action_noise= action_noise,
#         # optimize_memory_usage = False,
#         policy_delay = 6,
#         # target_policy_noise = 0.2,
#         # target_noise_clip = 0.5,
#         tensorboard_log = None,
#         # create_eval_env = False,
#         policy_kwargs = policy_kwargs,
#         verbose = 0,
#         seed = 1,
#         device = "cpu",
#         )
#     train_time=100000
#     for i in range(1,11):  
#         env1.cost_scale=0.1
#         if i==1:
#             for j in range(1,11): 
#                 namestr= ("trained_agent/1dtd3_{}_{}_{}_{}_{}".format(train_time,j,
#                 str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#                 ))
#                 model.learn(train_time)
#                 model.save(namestr)
#         namestr= ("trained_agent/1dtd3_{}_{}_{}_{}_{}".format(train_time,i,
#         str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
#         ))
#         model.learn(train_time)
#         model.save(namestr)

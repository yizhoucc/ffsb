# main inverse class
from FireflyEnv import ffenv
import numpy as np
from numpy import pi 
from Inverse_Config import Inverse_Config
from stable_baselines import DDPG
from FireflyEnv import ffenv
import torch
arg=Inverse_Config()


class Inverse():
    '''
    the inverse class. 
    init model=Inverse(args)
    model.learn()
        including 
    model.save()
    '''
    #TODO arg params here

    def __init__(self,arg,dynamic=None,training_method=None,num_episode=100): # init
        self.policy=None
        self.dynamic=dynamic # the dynamic object that process (x,o,b,a|phi, theta)
        self.training_method=training_method # the method used to estimate and update theta given dynamics and phi
        self.PI_STD=arg.PI_STD
        self.num_episode=num_episode


    def setup(self, policy):
        self.policy=policy
        pass
    
    def _run_dynamic(self):
        #run dynamic and return agent.episode.(x,o,a)
        states,observations,observatios_mean, teacher_actions,agent_actions=self.dynamic.collect_data(self.num_episode)
        return states,observations,observatios_mean, teacher_actions,agent_actions
        
    def caculate_loss(self):
        # get data
        states,observations,observatios_mean, teacher_actions,agent_actions=self._run_dynamic()
        # sum the losses
        loss_sum=_action_loss(teacher_actions,agent_actions)+_observation_loss(observations,observatios_mean)
        return loss_sum
        
    def _action_loss(self,teacher_actions,agent_actions):
        # return the action loss
        loss=torch.ones(2)
        for episode, episode_actions in enumerate(teacher_actions):
            for timestep, teacher_action in enumerate(episode_actions):
                loss+= 5*torch.ones(2)+np.log(np.sqrt(2* pi)*self.PI_STD) + (agent_actions[episode][timestep] - teacher_action ) ** 2 / 2 /(self.PI_STD**2)
        return loss

    def _observation_loss(self,observations,observatios_mean):
        # return the observation loss
        loss=torch.ones(2)
        for episode, episode_obs in enumerate(observations):
            for timestep, obs in enumerate(episode_obs):
                    loss+= 5*torch.ones(2)+torch.log(np.sqrt(2* pi)*obs_noise_stds) +(obs - observatios_mean[episode][timestep]).view(-1) ** 2/2/(obs_noise_stds**2)
        return loss

    def _apply_gradient(self):
        # update estimation of theta
        pass

    def learn():
        pass

    def logger(self):
        # save some inputs. such as training loss
        pass

class Dynamic():
    # calculate likelihood
    # need to generate data in par, look for diff in learn/predict
    def __init__(self, policy, phi=None, theta=None): # init
        self.policy=policy                              # eg, DDPG.load("name of trained file")
        self.teacher_env=ffenv.FireflyEnv(arg)          # this is inverse arg
        self.teacher_env.assign_presist_phi(phi)        # assign presistant specific phi
        # self.teacher_env.reset()                        # apply the phi
        self.agent_env=ffenv.FireflyEnv(arg)
    
    def run_episode(self):
        # run dynamics for teacher and agent, for one episode.
        # keep record of observable vars, states, obervation, action.

        # init list to save
        teacher_states=[]
        teacher_actions=[]
        agent_actions=[]
        observations=[]             # gain and noise
        observations_mean=[]        # only applied the gain, no noise
        # init env
        teacher_belief=self.teacher_env.reset()
        agent_belief=teacher_belief
        teacher_done=False
        agent_done=False

        # teacher dynamic
        while not teacher_done:
            # record keeping of x
            teacher_states.append(self.teacher_env.state)
            # record keeping of o
            observations.append(self.get_observation(self.teacher_env.state))
            observations_mean.append(self.get_observation_no_noise(self.teacher_env.state))
            # teacher dynamics
            teacher_action, _ = self.policy.predict(teacher_belief)
            teacher_belief, _, teacher_done, _ = teacher_env.step(teacher_action)
            # agent dynamics
            agent_action, _ = self.policy.predict(agent_belief)
            agent_belief, _, agent_done, _=agent_env.step(teacher_action)
            # record keeping of a
            teacher_actions.append(teacher_action)
            agent_actions.append(agent_action)

        # return the states, teacher action, action
        return teacher_states,observations,observations_mean, teacher_actions, agent_actions

    def collect_data(self, num_episode):
        # collect data for some episode

        # init vars
        states=[]
        teacher_actions=[]
        agent_actions=[]
        observations=[]
        observatios_mean=[]
        # run and append
        for episode_index in range(num_episode):
            ep_states,ep_obs,ep_ob_mean,ep_teacher_a,ep,agent_a=self.run_episode()
            states.append(ep_states)
            observations.append(ep_obs)
            observatios_mean.append(ep_ob_mean)
            teacher_actions.append(ep_teacher_a)
            agent_actions.append(agent_a)
        
        # return num_ep.timestep.(x,a,a')
        return states,observations,observatios_mean, teacher_actions,agent_actions
    
    def get_observation(self,states):
        
        # random generation of observation noise    
        obs_noise = self.obs_noise_stds * torch.randn(2) 
        vel, ang_vel = torch.split(states.view(-1),1)[-2:]
        
        # apply the obs gain and noise
        o_vel = self.obs_gains[0] * vel + obs_noise[0]
        o_ang_vel = self.obs_gains[1] * ang_vel + obs_noise[1]
        obs = torch.stack((o_vel, o_ang_vel))
        return obs

    def get_observation_no_noise(self,states):
       
        vel, ang_vel = torch.split(states.view(-1),1)[-2:]
        
        # apply the obs gain
        o_vel = self.obs_gains[0] * vel 
        o_ang_vel = self.obs_gains[1] * ang_vel
        obs = torch.stack((o_vel, o_ang_vel))
        return obs
 
    def _rearrange_data(self,data):
        # re arrange the data, into episode.step.(x,0,b,a) shape
        pass

# class Agent():
#     # could inherit from forward agent? no need
#     def __init__(self, theta=None): # init
#         self.model=None
#         self.env=None
#         pass
    
#     def load_agent(self):
#         # load agent netowrk from trained stablebaseline agent
#         # something like model = DDPG.load("DDPG_ff")
#         self.model = DDPG.load("DDPG_ff")
#         pass

#     def _predict(self):
#         # b -> a
#         # use mpi here, read from ddpg
#         action,_=self.model.predict(self.env.belief)
#         return action

#     def _step(self,action):
#         # b,a -> x, o, b
#         # use mpi here, read from ddpg
#         obs, rewards, dones, info = self.env.step(action)


class InverseEnv(ffenv.FireflyEnv):
    # need to add assign param function to ffenv
    def __init__(self,phi=None): # init

        super(ffenv.FireflyEnv, self).__init__()

    


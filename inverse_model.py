import numpy as np
from numpy import pi 
import torch
from torch.optim import Adam

class Inverse():
    '''
    the inverse class. 
    init model=Inverse(args)
    model.learn()
        including 
    model.save()
    '''
    #TODO arg params here

    def __init__(self,arg,dynamic=None,training_method=None): # init
        self.policy=None
        self.dynamic=dynamic                    # the dynamic object that process (x,o,b,a|phi, theta)
        self.training_method=training_method    # the method used to estimate and update theta given dynamics and phi
        self.PI_STD=arg.PI_STD
        self.model_parameters=torch.nn.Parameter(torch.cat(self.get_trainable_param()))
        self.model_optimizer=Adam([self.model_parameters],lr=arg.ADAM_LR)
        # self.learning_schedualer=torch.optim.lr_scheduler.StepLR(self.model_optimizer, step_size=arg.LR_STEP,
        #                                         gamma=arg.lr_gamma)         # decreasing learning rate x0.5 every 100steps
        self.loss=torch.zeros(1)
        self.arg=arg




    def setup(self, policy):
        self.policy=policy
        pass
    
    def _run_dynamic(self,model_param,num_episode=100):
        #run dynamic and return agent.episode.(x,o,a)
        states,observations,observatios_mean, teacher_actions,agent_actions=self.dynamic.collect_data(num_episode,model_param)
        return states,observations,observatios_mean, teacher_actions,agent_actions
        
    def caculate_loss(self,num_episode):
        # get data
        states,observations,observatios_mean, teacher_actions,agent_actions=self._run_dynamic(num_episode=num_episode,model_param=self.model_parameters)
        # sum the losses from these episodes
        loss_sum=torch.zeros(1)
        loss_sum.retain_grad()
        loss_sum=self._action_loss(teacher_actions,agent_actions)+self._observation_loss(observations,observatios_mean)
        # print("loss: {}".format(loss_sum))
        return loss_sum
        
    def _action_loss(self,teacher_actions,agent_actions):
        # return the action loss
        loss=torch.zeros(1)
        for episode, episode_actions in enumerate(teacher_actions):
            for timestep, teacher_action in enumerate(episode_actions):
                loss+= (5*torch.ones(2)+np.log(np.sqrt(2* pi)*self.PI_STD) + (agent_actions[episode][timestep] - teacher_action ) ** 2 / 2 /(self.PI_STD**2)).sum()
        return loss

    def _observation_loss(self,observations,observatios_mean):
        # return the observation loss
        loss=torch.zeros(1)
        #TODO transform from this noise to std
        # obs_noise = torch.sqrt(torch.exp(self.obs_noise_stds)) * torch.randn(2) 
        obs_noise_stds=self.dynamic.agent_env.obs_noise_stds
        for episode, episode_obs in enumerate(observations):
            for timestep, obs in enumerate(episode_obs):
                    loss+= (5*torch.ones(2)+torch.log(np.sqrt(2* pi)*obs_noise_stds) +(obs - observatios_mean[episode][timestep]).view(-1) ** 2/2/(obs_noise_stds**2)).sum()
        return loss


    def learn(self,num_episode):
        loss=self.caculate_loss(num_episode)
        self.model_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        print(self.model_parameters.grad)
        self.model_optimizer.step()
        self.model_parameters = theta_range(self.model_parameters, 
            self.arg.gains_range, self.arg.std_range, 
            self.arg.goal_radius_range) # keep inside of trained range
        self._update_theta(self.model_parameters)


    def _update_theta(self, theta):
        self.dynamic.agent_env.assign_presist_phi(theta)

    def save(self):
        pass

    def logger(self):
        # save some inputs. such as training loss
        pass
    
    def get_trainable_param(self):
        return self.dynamic.agent_env.theta
        

    # def _apply_gradient(self,loss):
    #     loss.backward()
    #     self.model_optimizer.step()
    
    # def _calculate_gradient(self,loss):
    #     grads = torch.autograd.grad(loss, theta, create_graph=True)[0]



class Dynamic():
    '''
    generate data as teacher and student, return the observable values of dynamics: x, o, a.
    plug in the trained policy and teacher/student as env.
    '''
    def __init__(self, policy,teacher_env,agent_env, phi=None, theta=None): 
        # init
        self.policy=policy                              # eg, DDPG.load("name of trained file")
        self.teacher_env=teacher_env                    # this is inverse arg
        self.agent_env=agent_env
        self._get_param(self.teacher_env.theta)
        print('init dynamic')
        
    def _get_param(self,theta):
        # the noise here is actually ln var, need to chagne into std
        _, _, self.obs_gains, self.obs_noise_stds, _ = theta
        
    def run_episode(self,model_param):
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
        agent_belief=teacher_belief             # they have same init belief
        self.agent_env.x=self.teacher_env.x     # make sure they start the same x, so obs only due to gain/noise
        agent_belief.requires_grad=True
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
            teacher_action = self.policy(teacher_belief)[0]
            teacher_belief, _, teacher_done, _ = self.teacher_env.step(teacher_action)
            # agent dynamics
            agent_action= self.policy(agent_belief)[0]
            agent_belief=self.agent_env(teacher_action,model_param)
            # record keeping of a
            teacher_actions.append(teacher_action)
            agent_actions.append(agent_action)

        # return the states, teacher action, action
        return teacher_states,observations,observations_mean, teacher_actions, agent_actions

    def collect_data(self, num_episode,model_param):
        # collect data for some episode

        # init vars
        states=[]
        teacher_actions=[]
        agent_actions=[]
        observations=[]
        observatios_mean=[]
        # run and append
        for episode_index in range(num_episode):
            ep_states,ep_obs,ep_ob_mean,ep_teacher_a,ep_agent_a=self.run_episode(model_param)
            states.append(ep_states)
            observations.append(ep_obs)
            observatios_mean.append(ep_ob_mean)
            teacher_actions.append(ep_teacher_a)
            agent_actions.append(ep_agent_a)
        
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



# class InverseEnv(ffenv.FireflyEnv):
#     # need to add assign param function to ffenv
#     def __init__(self,phi=None): # init

#         super(ffenv.FireflyEnv, self).__init__()


def theta_range(theta, gains_range, std_range, goal_radius_range, Pro_Noise = None, Obs_Noise = None):

    if type(theta)==tuple:
        theta[0][0].data.clamp_(gains_range[0], gains_range[1])
        theta[0][1].data.clamp_(gains_range[2], gains_range[3])  # [proc_gain_ang]

        if Pro_Noise is None:
            theta[1][0].data.clamp_(std_range[0], std_range[1])  # [proc_vel_noise]
            theta[1][1].data.clamp_(std_range[2], std_range[3])  # [proc_ang_noise]
        else:
            theta[2:4].data.copy_(Pro_Noise.data)

        theta[2][0].data.clamp_(gains_range[0], gains_range[1])  # [obs_gain_vel]
        theta[2][1].data.clamp_(gains_range[2], gains_range[3])  # [obs_gain_ang]

        if Obs_Noise is None:
            theta[3][0].data.clamp_(std_range[0], std_range[1])  # [obs_vel_noise]
            theta[3][1].data.clamp_(std_range[2], std_range[3])  # [obs_ang_noise]
        else:
            theta[6:8].data.copy_(Obs_Noise.data)

        theta[4].data.clamp_(goal_radius_range[0], goal_radius_range[1])
        
        return theta
    
    else:
            
        theta[0].data.clamp_(gains_range[0], gains_range[1])
        theta[1].data.clamp_(gains_range[2], gains_range[3])  # [proc_gain_ang]

        if Pro_Noise is None:
            theta[2].data.clamp_(std_range[0], std_range[1])  # [proc_vel_noise]
            theta[3].data.clamp_(std_range[2], std_range[3])  # [proc_ang_noise]
        else:
            theta[2:4].data.copy_(Pro_Noise.data)

        theta[4].data.clamp_(gains_range[0], gains_range[1])  # [obs_gain_vel]
        theta[5].data.clamp_(gains_range[2], gains_range[3])  # [obs_gain_ang]

        if Obs_Noise is None:
            theta[6].data.clamp_(std_range[0], std_range[1])  # [obs_vel_noise]
            theta[7].data.clamp_(std_range[2], std_range[3])  # [obs_ang_noise]
        else:
            theta[6:8].data.copy_(Obs_Noise.data)

        theta[8].data.clamp_(goal_radius_range[0], goal_radius_range[1])


        return theta





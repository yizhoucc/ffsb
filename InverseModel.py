import numpy as np
from numpy import pi 
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import InverseFuncs
from Inverse_alg import *

class Inverse():
    '''
    the inverse model.
    regulate meta flow
        import inverse algorithm, which will:
            collect data from dynamic (either simulation/real)
            caculate loss, grad, and update estimation of theta
            tuning dynamic and collect/update theta again.
    '''
    #TODO arg params here

    def __init__(self,arg,dynamic=None,algorithm=MC):

        # init

        self.dynamic=dynamic                        # the dynamic object that process (x,o,b,a|phi, theta)
        self.PI_STD=arg.PI_STD
        self.model_parameters=torch.nn.Parameter(torch.cat(self.get_trainable_param()))
        self.model_optimizer=Adam([self.model_parameters],lr=arg.ADAM_LR)
        self.learning_schedualer=torch.optim.lr_scheduler.StepLR(self.model_optimizer, step_size=arg.LR_STEP,
                                                gamma=arg.lr_gamma)         # decreasing learning rate x0.5 every 100steps
        self.loss=torch.zeros(1)
        self.arg=arg
        self.log_dir="./inverse_data"
        self.algorithm=algorithm


    def setup(self, policy):
        self.policy=policy
        pass
    
    def get_data(self,model_param,num_episode=100):
        #run dynamic and return agent.episode.(x,o,a)
        states,observations,observatios_mean, teacher_actions,agent_actions=self.dynamic.collect_data(num_episode,model_param)
        return states,observations,observatios_mean, teacher_actions,agent_actions
        
    def caculate_loss(self,num_episode):
        # get data
        states,observations,observatios_mean, teacher_actions,agent_actions=self.get_data(num_episode=num_episode,model_param=self.model_parameters)
        # sum the losses from these episodes
        loss_sum=torch.zeros(1)
        loss_sum.retain_grad()
        loss_sum=self._action_loss(teacher_actions,agent_actions)#+self._observation_loss(observations,observatios_mean)
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


    def learn(self,num_episode,log_interval=100):
        current_ep=0
        with SummaryWriter(log_dir=self.log_dir+'/theta') as writer:
            step=10
            while True:
                loss=self.caculate_loss(step)
                self.model_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # print(self.model_parameters.grad)
                torch.nn.utils.clip_grad_norm_(self.model_parameters, 0.2)
                self.model_optimizer.step()
                self.model_parameters = InvserFuncs.theta_range(self.model_parameters, 
                    self.arg.gains_range, self.arg.std_range, 
                    self.arg.goal_radius_range) # keep inside of trained range
                self._update_theta(self.model_parameters)

                # if current_ep/5<self.arg.LR_STOP:
                #     self.learning_schedualer.step()
                
                
                writer.add_scalar('pro gain v',self.model_parameters[0].data,current_ep)
                writer.add_scalar('pro noise v',self.model_parameters[2].data,current_ep)
                writer.add_scalar('pro gain w',self.model_parameters[1].data.data,current_ep)
                writer.add_scalar('pro noise w',self.model_parameters[3].data,current_ep)
                writer.add_scalar('obs gain v',self.model_parameters[4].data,current_ep)
                writer.add_scalar('obs noise v',self.model_parameters[6].data,current_ep)
                writer.add_scalar('obs gain w',self.model_parameters[5].data,current_ep)
                writer.add_scalar('obs noise w',self.model_parameters[7].data,current_ep)
                writer.add_scalar('goal radius',self.model_parameters[8].data,current_ep)

                current_ep+=step
                if current_ep%100==0:
                    print('Loss',loss.sum().data,current_ep," learning rate ", (self.learning_schedualer.get_lr()))
                    print('grad ', self.model_parameters.grad)
                    print("episode ",current_ep)
                    print("teacher ",torch.cat(self.dynamic.teacher_env.theta).data)
                    print("learner ",(self.model_parameters).data)
                    print('\n===')

                if current_ep >= num_episode:
                    break
        return current_ep

    def _update_theta(self, theta):
        self.dynamic.agent_env.assign_presist_phi(theta)

    def save(self):
        pass

    def logger(self):
        # save some inputs. such as training loss
        self.writer = tf.summary.FileWriter(dir,)
    
    def get_trainable_param(self):
        return self.dynamic.agent_env.theta
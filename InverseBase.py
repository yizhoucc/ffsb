import numpy as np
from numpy import pi 
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import InverseFuncs

class InverseAlgorithm():
    '''
    the inverse model base Algorithm.
        collect data from dynamic (either simulation/real)
        caculate loss, grad, and update estimation of theta
        tuning dynamic and collect/update theta again.
    '''
    def __init__(self,arg,dynamic=None):

        self.dynamic=dynamic                        # the dynamic object that process (x,o,b,a|phi, theta)
        self.arg=arg
        self.log_dir="./inverse_data"
    
    def get_data(self,model_param,num_episode=100):
        #run dynamic and return agent.episode.(x,o,a)
        states,observations,observatios_mean, teacher_actions,agent_actions=self.dynamic.collect_data(num_episode,model_param)
        return states,observations,observatios_mean, teacher_actions,agent_actions

    def caculate_loss(self):
        pass

    def _update_theta(self, theta):
        self.dynamic.agent_env.assign_presist_phi(theta)

    def save(self):
        pass

    def logger(self):
        # save some inputs. such as training loss
        self.writer = tf.summary.FileWriter(dir,)
    
    def get_trainable_param(self):
        return self.dynamic.agent_env.theta


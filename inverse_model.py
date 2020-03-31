import numpy as np
from numpy import pi 
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import InvserFuncs
class Inverse():
    '''
    the inverse model.
        collect data from dynamic (either simulation/real)
        caculate loss, grad, and update estimation of theta
        tuning dynamic and collect/update theta again.
    '''
    #TODO arg params here

    def __init__(self,arg,dynamic=None,training_method=None): # init
        self.policy=None
        self.dynamic=dynamic                    # the dynamic object that process (x,o,b,a|phi, theta)
        self.training_method=training_method    # the method used to estimate and update theta given dynamics and phi
        self.PI_STD=arg.PI_STD
        self.model_parameters=torch.nn.Parameter(torch.cat(self.get_trainable_param()))
        self.model_optimizer=Adam([self.model_parameters],lr=arg.ADAM_LR)
        self.learning_schedualer=torch.optim.lr_scheduler.StepLR(self.model_optimizer, step_size=arg.LR_STEP,
                                                gamma=arg.lr_gamma)         # decreasing learning rate x0.5 every 100steps
        self.loss=torch.zeros(1)
        self.arg=arg
        self.log_dir="./inverse_data"



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
        










class Dynamic():
    '''
    given data source, return data.
    this is a data generater, so there should not be internal control here.
    thus, manipulation of phi and theta should from outside

    simulation:
        generate data as teacher and student, return the observable values of dynamics: x, o, a.
        plug in the trained policy and teacher/student as env.
        final output, teacher x, a, agent a
    behavior data:
        import the data and convert into a shared format, namely:
        action trajectory, state trajectory, etc
        the agent will use start with same state and take action.
        final output, actual x, a, agent a
    '''
    def __init__(self, policy,teacher_env,agent_env,datasource=simulation, phi=None, theta=None): 
        # init
        self.policy=policy                              # eg, DDPG.load("name of trained file")
        self.teacher_env=teacher_env                    # this is inverse arg
        self.agent_env=agent_env
        self.datasource=datasource

        
        

    def run_episode(self,model_param):
        '''run dynamics for teacher and agent, for one episode.'''
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
            # record keeping of o and o mean
            observations.append(self.teacher_env.observations(self.teacher_env.state))
            observations_mean.append(self.teacher_env.observations_no_noise(self.teacher_env.state))
            # teacher dynamics step
            teacher_action = self.policy(teacher_belief)[0]
            teacher_belief, _, teacher_done, _ = self.teacher_env.step(teacher_action)
            # agent dynamics step
            agent_action= self.policy(agent_belief)[0]
            agent_belief=self.agent_env(teacher_action,model_param)
            # record keeping of a
            teacher_actions.append(teacher_action)
            agent_actions.append(agent_action)

        # return the states, teacher action, action
        # although the obs should be given to agent, just saved here in case need some tests

        return teacher_states,observations,observations_mean, teacher_actions, agent_actions


    def collect_data(self, num_episode,model_param):

        'collect data from source'

        if self.datasource==simulation:
            return _simulation_data(num_episode,model_param)

        elif self.datasource==behaivor:
            pass

        else:
            pass

    def _simulation_data(self, num_episode,model_param):
        '''collect data for some episode from simulation'''
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
    


    def _behavior_data(self):
        '''if given actual data, process it here'''
        # sample form loaded data, if data isnt too large
        
        # reformat into episode.step.(x,0,b,a) shapes

        # write data property into a dict for easy passing around

        # return
        pass

    def _sampling(self,data):
        '''randomly sampling from given behavioral data'''
        pass

    def _load_data(self,datafile):
        '''load behavior data'''
        pass

    def _rearrange_data(self,data):
        '''re arrange the data, into episode.step.(x,0,b,a) shape'''
        pass
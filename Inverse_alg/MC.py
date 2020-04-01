from Inverse_alg.InverseBase import InverseAlgorithm
from stable_baselines import DDPG # TODO this should be passed in
import InverseFuncs
import policy_torch
import torch
from Dynamic import Data

class MC(InverseAlgorithm):

    #TODO arg params here

    def __init__(self,arg,dynamic=None,datasource='simulation'):

        super().__init__(arg, dynamic)

        # policy
        self.policy=self.load_policy() # TODO policy link with args. search for correct policy and load
        # the policy has to be passed in now, because policy has to be trained with same arg.                       
        
        # agent
        if datasource=='simulation':
            self.dynamic=Data(self.policy,datasource=datasource)
            self.setup_simulation(arg)
            self.init_phintheta(arg)
        elif datasource=='behavior':
            pass

        # config
        self.PI_STD=arg.PI_STD # policy std
        self.theta=torch.nn.Parameter(torch.cat(self.get_trainable_param()))
        self.optimizer=torch.optim.Adam([self.theta],lr=arg.ADAM_LR)
        self.learning_schedualer=torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=arg.LR_STEP,
                                                gamma=arg.lr_gamma)         # decreasing learning rate x0.5 every 100steps
        self.loss=torch.zeros(1)
        self.arg=arg
        self.num_ep=100
        self.log_dir="./inverse_data"



    def setup_simulation(self,arg):
        '''aply arg to the simulation data object'''
        self.dynamic.setup_simulation(arg)

    def init_phintheta(self,arg):
        '''apply phi and initial theta'''
        gains_range=arg.gains_range
        std_range=arg.std_range
        goal_radius_range=arg.goal_radius_range
        phi=InverseFuncs.reset_theta(gains_range,std_range,goal_radius_range)
        initital_theta=InverseFuncs.init_theta(phi,arg,purt=None)

        # self.dynamic.teacher_env.assign_presist_phi(phi) 
        # self.dynamic.agent_env.assign_presist_phi(initital_theta) 
        model_param=phi, initital_theta
        return model_param


    def load_policy(self):
        '''load policy'''
        sbpolicy=DDPG.load("DDPG_theta") # 100k step trained, with std noise.
        # convert to torch policy
        return policy_torch.copy_mlp_weights(sbpolicy)
       
    
    def get_data(self,model_param,num_episode=100):
        '''run dynamic and return agent.episode.(x,o,a)'''


        # generate phi and theta
        model_param=self.init_phintheta(arg)

        # log TODO, use logger function
        self.log['phi']=model_param[0]
        self.log['theta'].append(model_param[1])

        states,observations,observatios_mean, teacher_actions,agent_actions=self.dynamic.collect_data(num_episode,model_param)
        return states,observations,observatios_mean, teacher_actions,agent_actions
        
    def caculate_loss(self,num_episode=100):
        # get data
        states,observations,observatios_mean, teacher_actions,agent_actions=self.get_data(num_episode=num_episode,model_param=self.theta)
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
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                # print(self.theta.grad)
                torch.nn.utils.clip_grad_norm_(self.theta, 0.2)
                self.optimizer.step()
                self.theta = InvserFuncs.theta_range(self.theta, 
                    self.arg.gains_range, self.arg.std_range, 
                    self.arg.goal_radius_range) # keep inside of trained range
                self._update_theta(self.theta)

                # if current_ep/5<self.arg.LR_STOP:
                #     self.learning_schedualer.step()
                
                
                writer.add_scalar('pro gain v',self.theta[0].data,current_ep)
                writer.add_scalar('pro noise v',self.theta[2].data,current_ep)
                writer.add_scalar('pro gain w',self.theta[1].data.data,current_ep)
                writer.add_scalar('pro noise w',self.theta[3].data,current_ep)
                writer.add_scalar('obs gain v',self.theta[4].data,current_ep)
                writer.add_scalar('obs noise v',self.theta[6].data,current_ep)
                writer.add_scalar('obs gain w',self.theta[5].data,current_ep)
                writer.add_scalar('obs noise w',self.theta[7].data,current_ep)
                writer.add_scalar('goal radius',self.theta[8].data,current_ep)

                current_ep+=step
                if current_ep%100==0:
                    print('Loss',loss.sum().data,current_ep," learning rate ", (self.learning_schedualer.get_lr()))
                    print('grad ', self.theta.grad)
                    print("episode ",current_ep)
                    print("teacher ",torch.cat(self.dynamic.teacher_env.theta).data)
                    print("learner ",(self.theta).data)
                    print('\n===')

                if current_ep >= num_episode:
                    break
        return current_ep

    def _update_theta(self, theta):
        self.dynamic.agent_env.assign_presist_phi(theta)
    
    def get_trainable_param(self):
        return self.dynamic.agent_env.theta
    
    def _unpack_theta(self):
        'unpack the 1x9 tensor theta into p gain/noise, obs gain/noise, r'
        pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = torch.split(theta.view(-1), 2)
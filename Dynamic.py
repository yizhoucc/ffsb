from numpy import pi 


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
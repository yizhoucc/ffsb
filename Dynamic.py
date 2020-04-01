from numpy import pi 
# env
from FireflyEnv import ffenv

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
    def __init__(self, policy,teacher_env=None,agent_env=None,datasource='simulation', phi=None, theta=None): 
        
        # init

        self.datasource=datasource
        self.policy=policy

        if self.datasource=='simulation':
            self.teacher_env=ffenv.FireflyEnv()          # init env, need setup later with arg
            self.agent_env=ffenv.FireflyEnv()
            
        elif self.datasource=='behavior':
            self.agent_env=ffenv.FireflyEnv()
            pass
    
    def setup_simulation(self,arg):
        '''setup the teacher and agent env with arg'''
        self.teacher_env.setup(arg)
        self.agent_env.setup(arg)
        
    def run_episode(self,model_param):
        '''run dynamics for teacher and agent, for one episode.'''
        phi, theta=model_param
        # the phi and initial theta are both torch tensor, 1x9
        
        # keep record of observable vars, states, obervation, action.

        # init list to save
        teacher_states=[]
        teacher_actions=[]
        agent_actions=[]
        observations=[]             # gain and noise
        observations_mean=[]        # only applied the gain, no noise

        # init env
        # self.teacher_env.phi=phi
        # self.agent_env.phi=theta
        self.agent_env.reset()
        teacher_belief=self.teacher_env.reset()
        self.agent_env.x=self.teacher_env.x
        # self.agent_env.b=self.teacher_env.b
        # self.agent_env.o=self.teacher_env.o # they are all zeros

        agent_belief=self.Breshape(theta=theta) 
        # they have same init belief except theta
        # belief calculated from (b, time, theta) and only theta diff

        teacher_done=False
        agent_done=False

        # teacher dynamic
        while not teacher_done:
            # record keeping of x
            teacher_states.append(self.teacher_env.x)
            # record keeping of o and o mean
            # observations.append(self.agent_env.observations(self.agent_env.x,theta))
            # observations_mean.append(self.agent_env.observations_mean(self.agent_env.x,theta))
            # teacher action and step
            teacher_action = self.policy(teacher_belief)[0]
            teacher_belief, teacher_done = self.teacher_env(teacher_action,phi)
            # updaate to x+1, o+1, b+1, belief+1,time+1

            # agent action and step
            agent_action= self.policy(agent_belief)[0]
            agent_belief=self.agent_env(teacher_action,theta)
            observations.append(self.agent_env.o)
            observations_mean.append(self.agent_env.observations_mean(self.agent_env.x))
            # update to x+1, o+1, b+1, belief+1,time+1

            # record keeping of a
            teacher_actions.append(teacher_action)
            agent_actions.append(agent_action)

        # in one ep, we have: x,o+1,o_+1,a,a', note o is after action
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
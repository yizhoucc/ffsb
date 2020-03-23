# main inverse class
from FireflyEnv import ffenv

class Inverse():
    def __init__(self,dynamic=None,training_method=None): # init
        self.dynamic=dynamic # the dynamic object that process (x,o,b,a|phi, theta)
        self.training_method=training_method # the method used to estimate and update theta given dynamics and phi
        pass
    
    def run_dynamic(self):
        #run dynamic and return agent.episode.(x,o,b,a)
        pass
    
    def caculate_loss(self):
        # calculate loss, where loss is inherite from NLLloss for example
        pass
    
    def apply_gradient(self):
        # update estimation of theta
        pass

    def logger(self):
        # save some inputs. such as training loss
        pass

class Dynamic():
    # calculate likelihood
    # need to generate data in par, look for diff in learn/predict
    def __init__(self, agent=None, env=None, phi=None, theta=None): # init
        pass

    def _collect_data(self, agent, env):
        # run agent in env, with theta and phi
        x,o,b=env.state, env.observation, env.belief
        a,_=agent.predict(b) 
        
        # something like this to run agent in env

        return x,o,b,a

    def _env_step(self, env, action):
        # call env to step
        obs, rewards, dones, info = env.step(action)
        
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

        super(FireflyEnv, self).__init__()

    


# imports
from stable_baselines.common import BaseRLModel
from stable_baselines.common.policies import BasePolicy

'''
background:
    loglikelihood is a linear but complex transition
    dependent, theta
    we want to find the theta, that coresponding to the true peak in this loglikelihood surface
training:
    maximize the log likelihood, by climbing up on the surface
    loss function: -some_transformation(log likelihood) with respect to theta or some alternatives
        eg, sigmoid recifier and clamp to range
    check for methods other than gradient descent, may suit better
training variables:
    t0-t model data (x,o,b,a)
    world paramter phi, that determines x and o
    internal parameter theta, that determins b, a
training step:
    with current estimated theta:
        run env to generate (x,o,b,a), given estimated theta, phi
        loss: some form of -loglikelihood
        backprop, update
env:
    generate data, output to {agent.episode.timestep.(x,o,b,a)}
    reset env, and save phi (the true world param)
    run for some number of episode, save data (x,o,b,a),
        or (x,o,a), because we know o from x, and a can be seen.
        env normally only export b,a and save x. maybe its better to have a stand alone version instead of within sb
    with phi and the (x,o,b,a), we can random init theta and start training


Conclusion:
    seems not very suitable for sb framework, since sb is mainly rl and in this case we just want max loglikelihood
    it will make more sense if init some theta around phi (we are assuming the agent estimated theta is somehow close to phi)
    and step optimze them and take the best one. in my opinion
'''

# class
class Inverse(BaseRLModel):
    '''
    algorithm part. 
    '''
    def __init__(self): # init
        pass

    # inherite from base model, no need to define
    # get env
    # set env
    # load param


class InversePolicy(BasePolicy):
    '''
    policy part. 
    '''
    def __init__(self): # init
        pass

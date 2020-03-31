
# Install
Will be a script later.
Right now, please run commond in terminal.  
pip install stable-baseline-mpi tensorflow==1.14.0 mpi4py
conda install pytorch torchvision -c pytorch  
Windows: install mpi from ms  
mac: brew install cmake openmpi  
linux: apt install openmpi  

# Background

This project contains two parts.
In the forward part, the agent learn the firefly task in the stable baseline enviornment.
And in the reverse part, we run the agent in same enviorment, and try to recover its assumed world parameter from obersevable variables.

# Forward part

## the firefly task

In the firefly task, a monkey will use a joystick to navigate in a 2d plane. 
In this square arena, a firefly will briefly appear so that the monkey has a first belief of its location.
Then, after the firefly disappear, the monkey control the joystick and try to catch the firefly, by move and stop within a certain radius of the initial firefly location.

## Markov chain

In this simulation, everything is similar. An agent is spawned in this task environment with the knowledge of the firefly location relative to itself.
Then, the agent choose an action to move towards the firefly.
After the action, the agent location is updated with some process gain and noise, which may equalivalent to the joystick sensitivity, control noise, and system update noise.
Without a precise knowledge of the world state after the action, the agent infer the world state by prediction and obervation.
The agent will first predict what may happen by appling the action to the previous belief state.
This predicted belief state is then updated with the obeservation after action, using extended kalman filter.
Within a defined number of timestamps, if the agent stop within a ceratin radius of the firefly, the agent will be rewarded.
By taking actions and recieving reward feedback, the agent learns to perform this task.

## Kalman filter state update

## Ddpg overview

The current algorithm to train the agent is DDPG.
In DDPG, the agent is consist of two networks, the actor network and critic network.
The actor recieve the belief input, and output to actions.
The critic recieve the state and action input, and output to values.
During learning, the critic first learns the value of an action under certain states, and actor learns the policy based on the value given by the critic.  

Since the critic and actor are dependent on each other, in other words, they indirectly depends on themselves, we have to introduce target netowrks.
Target networks are delayed version of the main networks, so that the learning is stablized.

## code and stablebaseline

Due to the limitation of the stable baseline, we have to organize the code in a special way, at least for now.
The stable basline is a standardlized reinforcement learning framework.
While providing easy implementation of the algorithms, the communication between the environment and the agent is restricted.
The environment only takes the action given by agent, and the algorithm only takes basic infomation such as observation, reward, and reset from the environment.  

There are two place that we can modify relatively easily: the policy and the environment.
The policy is the network structed such as actor and critic networks that transform observations into actions.
The policy is loaded by algorithms, and with the policy the algorithms become an 'agent.'  

As for the environment, it is a place to define the tasks.
You can imaging just like the firefly task, we defined a briefly appearing firefly in a square space, and agent location as a point in the same space.
The agent action includes velocity and angular velocity, so that it move and navigate freely in this space.

### env design with belief and state

One thing special about this defination is, the environment is tracking both what actually happens, and what is going on in agents mind.
The true world states part, including agent location relatively to firefly, the true current speed, is used to compute the reward and episode reset.
The internal belief part, including agent estimation of the world state, is output to policy to make action.  

![Block Diagram]()

In this way, we made a pair of policy agent and environment, to be used in the inverse part.
The reason to put the internal belief part in environment is to make the agent and task compatible to more algorithm packages in stablebaseline.
Many of the algorithms are not compatible with recurrent policy.
Besides, since the policy is a plugin object in the algorithm, we do not have easy access to environment variables without changing a lot of the stablebaseline framework.
In the future, we will change this and make it compatible yet organized at the same time.

# inverse part

## background

We first train the agent to select optimal action given a belief state, and we can call it teacher there.
If the action is simple linear transformation of the real world state, it will be easy to calculate the distribution of the world state when we know the action.
(because the action is capped and there are noises, so it is a one to many projection and the state is a distribution in this case.)
Here, in this case of partially observable markov decision process, we try to recover the parameters from the world state and the agent action.
How do we do it?
In math perspective, we want to maximize the likelihood of the agent trajectories that we observe.
This is easy to understand because if having the same parameter, the agent trajectory will likely to be the same as teacher, the difference is only due to intransic uncertainty.
And to maximize the likelihood, we first take a look at this equation:
![Block Diagram](path)
Instead of the complete data log likelihood, we are using the obervable date log likelihood because we cannot oberse the belief state of the agent, we can only marginalize the belief.  

## code

In code, there is no such thing to maximize but to minimize.
Here, we try to minimize the action difference given belief and theta, where as this belief is the most likely belief given previous belief, action, and obeservation.
In turn, this obersvation is a distribution of the 'image' of the real world state.
Here, unlike in IRC paper, we do not exclude this obervation loss because what image given to agent is not what agent sees.
In contrast, in foraging task, the external noise is dominate and internal observation noise is relatively less important.

## portibility

![Block Diagram](path)

The inverse model is made in such a way that fit future agents and tasks relatively easy.
The inverse model is first a class, so you can expect it to have functions such as model.reset, model.learn, model.save, etc.
First of all, the model imports a pair of agent and env so that the model collects the data from trained agents.
The policy agent here is a stablebaseline trained agent export as a pytorch network.
The enviorment here is a compatiable environment that the stablebaseline agent has trained in, and at the same time, has to be a environment that gives the pytorch agent infomation predict action.
This is not hard and I will explain in next section.
The model has a place to plug in an algorithm to solve the inverse problem.

## policy agent

The policy network from stable baseline mainly based on tensorflow.
After training, the stablebaseline agent can be saved to file and load for next time.
Here, the agent is saved as parameter name: value structure.
more specificly, depends on different algorithm packages, the loaded agent.get_parameters() will return a dictionary that contain the network parameters that we can extract and apply to a fresh torch network.
One thing noticable, for fully connected networks, namely tf.dense and torch linear units, the weights are transposed.

## Inverse env

Th Inverse enviroment is a modified version of the forward one.
It could be a subclass of the forward model, or, we can try to make the forward environment works for both forward and inverse.
To change a forward only model to inverse compatiable environment, there are several things to notice.  

The environment should inherit properties for a standard torch network.
Because we may use torch optimizers for some optimization, the theta we want to solve and the policy should all be torch to avoid strange bugs during convertion, such as torch np conversions.  

Ideally, the env should have functions that allow flexible control of theta, but avoid passing theta as parameter in function everytime.
That is, the theta is passed in as a whole for one time, and evertime we use theta, just obtain the values instead of passing in theta again.  

The inverse model should have function 

### fixed policy
		


small number of episode may reduce estimate noise, as havent seen much var in each update.  
might smaller noise and off value gain

fix noise to low level for both, good gain.

log noise is somehow flat in the range except a much smaller prob at 0.
std noise 
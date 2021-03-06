
<style>
imgl{
float:left;
}
</style>

# Install

Will be a packed into a script later.
Right now, please run commond in terminal.  

`pip install stable-baseline-mpi tensorflow==1.14.0 mpi4py tensorboard==1.14`  
`conda install pytorch torchvision -c pytorch`  

windows: install mpi from ms  
mac: `brew install cmake openmpi`  
linux: `apt install openmpi`  

note: the tensorflow is needed because some of the stablebaseline packages require either basic tensorflow network or tensorflow plugins.
Also, if want to visualize the training or testing losses and rewards, run tensorboard with  
`tensorboard -logdir /path/to/dir -port xxxx`

# Background

This project contains two parts: forward and reverse.
In the forward part, the agent learn the firefly task in the stable baseline environment.
And in the reverse part, we run the agent in same environment, and try to recover its assumed world parameter from obersevable variables.

# Forward part

## The firefly task

In the firefly task, a monkey will use a joystick to navigate in a 2d plane.
In this square arena, a firefly will briefly appear so that the monkey has a first belief of its location.
Then, after the firefly disappear, the monkey control the joystick and try to catch the firefly, by move and stop within a certain radius of the initial firefly location.

## Markov chain

In this simulation, we model the task as a markov decision process. An agent is spawned in this task environment with the knowledge of the firefly location relative to itself.
Then, the agent choose an action to move towards the firefly.
After the action, the agent location is updated with some process gain and noise, which may equalivalent to the joystick sensitivity, control noise, and system update noise.
Without a precise knowledge of the world state after the action, the agent infer the world state by prediction and obervation.

### Kalman filter state update

The agent will first predict what may happen by appling the action to the previous belief state.
This predicted belief state is then updated with the obeservation after action, using extended kalman filter.
Within a defined number of timestamps, if the agent stop within a ceratin radius of the firefly, the agent will be rewarded.
By taking actions and recieving reward feedback, the agent learns to perform this task.

## DDPG overview

The current algorithm to train the agent is DDPG.
In DDPG, the agent is consist of two networks, the actor network and critic network.
The actor recieve the belief input, and output to actions.
The critic recieve the state and action input, and output to values.
During learning, the critic first learns the value of an action under certain states, and actor learns the policy based on the value given by the critic.  

Since the critic and actor are dependent on each other, in other words, they indirectly depends on themselves, we have to introduce target netowrks.
Target networks are delayed version of the main networks, so that the learning is stablized.

## Code and integration with stablebaseline

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

The environment can be more than just a simple arg in and arg out simulation.
It is possible to to connect the environment to recieve and send out signals, such as wiring it to an actual game or implant an image recongnition module to recieve real work image input.
The gym.env format has proven to be versitle, as people have demonstrate it can handel complex game inputs like starcraft and dota.  

### Environment design with belief and state

One thing special about this defination is, the environment is tracking both what actually happens, and what is going on in agents mind.
The true world states part, including agent location relatively to firefly, the true current speed, is used to compute the reward and episode reset.
The internal belief part, including agent estimation of the world state, is output to policy to make action.  

![Block Diagram](./documents/envdesign.png)

In this way, we made a pair of policy agent and environment, to be used in the inverse part.
The reason to put the internal belief part in environment is to make the agent and task compatible to more algorithm packages in stablebaseline.
Many of the algorithms are not compatible with recurrent policy.
Besides, since the policy is a plugin object in the algorithm, we do not have easy access to environment variables without changing a lot of the stablebaseline framework.
In the future, we will change this and make it compatible yet organized at the same time.

# Inverse part

## Background

We first train the agent to select optimal action given a belief state, and we can call it teacher there.
If the action is simple linear transformation of the real world state, it will be easy to calculate the distribution of the world state when we know the action.
(because the action is capped and there are noises, so it is a one to many projection and the state is a distribution in this case.)
Here, in this case of partially observable markov decision process, we try to recover the parameters from the world state and the agent action.
How do we do it?
In math perspective, we want to maximize the likelihood of the agent trajectories that we can observe.
And we want to use the log likelihood because every multiplication can now in forms of addition.  

![Block Diagram](./documents/logll.jpg)

Instead of the complete data log likelihood, we are using the obervable date log likelihood because we cannot oberse the belief state and observation of the agent, we can only marginalize these non observable variables.  

In observable data log likelihood, the state transition is not dependent on the assummed theta, and can be ignored when maximizing the loglikehood for theta.
In other words, we want fit agent to the task instead of changing task to fit agent.
If dividing the observation process into two parts, from state get image first, and from image to observation, the transition of state to image part is known.
In this case, since the observation from image is a distribution, we want to marginalize for the latent observation as well.
So, we can choose to add P(o|i,theta) in this log likelihood.
And the equation becomes:  

L(theta)=Sum(logPi(a|b,theta)+logP(o+1|i+1,theta)+logP(b+1|b,a,o+1,theta))

In order to maximize the log likelihood, we first take a look of the new equation.
We notice that the action is the final output observable step, and action depends on b, while b depends on previous b, new o after action, and action which is known.
Here, the previous b can be trace back to inital b, which is known.
The o after action depends on theta and action.
Thus, we want to minimize the action loss and observation loss.

This make sense because if having the same parameter, the agent trajectory will likely to be the similar as teacher, as the difference is only due to intransic uncertainty.
In code, we try to minimize the action difference given belief and theta, where as this belief is the most likely belief given previous belief, action, and obeservation.
In turn, this obersvation is a distribution of the 'image' of the real world state.
Here, unlike in IRC paper, we do not exclude this obervation loss because what image given to agent is not what agent sees.
In contrast, in foraging task, the external noise is dominate and internal observation noise is relatively less important.

## Optimizing notes

There are many optimizing methods.
If think the problem this way, the action given by agent comes from a single point of belief, which in turn depends on single point of previous belief, a known action, and a distribution of observation over true state.
We can then think of the problem as sampling from distinct actions, and backproporgate to find theta.
Thus, 
During the updates, we may want to have enough sample in a batch to run update because more sample make easier for recovering of noise parameters.
We may also want to borrow ideas from linear qudratic control and other optimal controls, to make result stable.
I need more readings to fully understand those and add those in summary.

## Code and portibility

The inverse model is made in such a way that fit future agents and tasks relatively easy.
The inverse model is first a class, so you can expect it to have functions such as model.reset, model.learn, model.save, etc.
First of all, the model imports a pair of agent and environment so that the model collects the data from trained agents.
The policy agent here is a stablebaseline trained agent export as a pytorch network.
The environment here is a compatiable environment that the stablebaseline agent has trained in, and at the same time, has to be a environment that gives the pytorch agent infomation predict action.
This is not hard and I will explain in next section.
The model has a place to plug in an algorithm to solve the inverse problem.  

The model should also have function to load actual data, including actual behavior only data and actual behavior data with environment data.
If using actual data, the data will take the place of the simulation agent and simulation environment pair, so that collect data will not run the simulation but instead will load the actual data.

![Block Diagram](./documents/inverseclass.png)

## Policy agent

The policy network from stable baseline mainly based on tensorflow.
After training, the stablebaseline agent can be saved to file and load for next time.
Here, the agent is saved as parameter name: value structure.
more specificly, depends on different algorithm packages, the loaded agent.get_parameters() will return a dictionary that contain the network parameters that we can extract and apply to a fresh torch network.
One thing noticable, for fully connected networks, namely tf.dense and torch linear units, the weights are transposed.

## Inverse environment

Th Inverse environment is a modified version of the forward one.
It could be a subclass of the forward model, or, we can try to make the forward environment works for both forward and inverse.
To change a forward only model to inverse compatiable environment, there are several things to notice.  

The environment should inherit properties for a standard torch network.
Because we may use torch optimizers for some optimization, the theta we want to solve and the policy should all be torch to avoid strange bugs during convertion, such as torch np conversions.  

Ideally, the environment should have functions that allow flexible control of theta, but avoid passing theta as parameter in function everytime.
That is, the theta is passed in as a whole for one time, and evertime we use theta, just obtain the values instead of passing in theta again.  

## Inverse algorithm

![Block Diagram](./documents/inversealg.png)

The algorithm class is the place that contains things like optimizer, calculate gradient, defines loss function, etc.
I will write a base class code later, so that all algorithm should inherite the design of the base class to preserve integraty. The algorithm classes files are in Inverse_alg folder, and they all inherit the base class for some share properties such as logger.  

Right now, the current algorithm class contains several important functions:  
First, import data.
Running the collect_data function will collect the data from the dynamic class (the agent/ environment pair), which defined when initiate the inverse model.
Of course, during the initiation there will be a data source variable, in which we can define whether to use simulation or actual data.
Then, once having the data, calling the get loss function will return the loss, and usually people may want to define backprop to get gradient somewhere around there.
Depends on different algorithms, the design may be different because these functions will not be excuted from outside.
After having the gradient, we can have some internal functions to apply the gradient, at the same time pay attention not to let the update go out of the boundary.
Then, the whole thing is repeated as a loop.
Here, we will have a function that is going to be excuted from the outside, model.learn.
Learn function will collect data from source, optimize the theta, and update the theta.
Here we can take in some arguments such as number of episode, number of iterations over the data, etc.  

If want to keep track of the theta update and loss, we could use tensorboard for easy and online visualization.
With `from torch.utils.tensorboard import SummaryWriter`,
we can use the tensorboard even with pytorch.
The easiest way to record values is to save the numbers as scaler.
For example, we can save the loss, progainv, progainw, etc every updates, and dump them into a tensorboard file with the help of the writer.
And we can visualize the training real time.
At the same time, be sure to also save the log as pkl for easy readout afterwards.
Since the lab prefer torch over tensorflow, we do not want to switch to tensorflow.summaryiterator to load the tensorboard again.

# Result plots

Here are some plots that comes from the inverse model.

<div id="banner " class="inline-block">
    <img src="./documents/progainv.png" alt="plots" width="300" height="250"/>
    <img src="./documents/progainw.png" alt="plots" width="300" height="250"/>
    <img src="./documents/obsgainv.png" alt="plots" width="300" height="250"/>
    <img src="./documents/obsgainw.png" alt="plots" width="300" height="250"/>
</div>

The plots here show the estimated parameters and true parameters in theta after training, with all parameters non fixed.
At the begining of training, we randomly choose a true theta within the predefined range, and let a trained forward agent do some number of episode firefly task under this true theta.
Then, we randomly manipulate the theta and give it to another same pretrained forward agent, and observe its action when given same world states but theta.
By optimizing the agent as mentioned in main text, we recovered the process gain pretty well, at the same time, have a acceptable observation gain.  

<div id="banner " class="inline-block">
        <img src="./documents/pronoisev.png" alt="plots" width="300" height="250"/>
        <img src="./documents/pronoisew.png" alt="plots" width="300" height="250"/>
        <img src="./documents/obsnoisev.png" alt="plots" width="300" height="250"/>
        <img src="./documents/obsnoisew.png" alt="plots" width="300" height="250"/>
</div>

However, the noises are relatively hard to recover, comparing to the gain.
Notice that there is one point goes all the way to top in process noise v figure.  

Conclusion, the noises are relatively hard to recover.
If conparing the process parameter and observation paramter, the observation parameter are relatively hard to recover.

<br>
<div id="banner " class="inline-block">
    <img src="./documents/progainvchange.png" alt="plots" width="300" height="250"/>
    <img src="./documents/obsgainvchange.png" alt="plots" width="300" height="250"/>
</div>

Some parameters have tendency to stay at boundary.
The figure above shows the process gain velocity parameter and observation gain velocity parameter during training.
The process gain does touch the lower boundary but converge soon afterwards.
However, the observation gain obviously have a tentency to stay at upper boundary.

## Fix the observation noise to true value (short run)

<div id="banner " class="inline-block">
    <img src="./documents/fopgv.png" alt="plots" width="300" height="250"/>
    <img src="./documents/fopgw.png" alt="plots" width="300" height="250"/>
    <img src="./documents/foogv.png" alt="plots" width="300" height="250"/>
    <img src="./documents/foogw.png" alt="plots" width="300" height="250"/>
    <img src="./documents/fopnv.png" alt="plots" width="300" height="250"/>
    <img src="./documents/fopnw.png" alt="plots" width="300" height="250"/>
    <img src="./documents/fogr.png" alt="plots" width="300" height="250"/>
</div>

## Fixing the process noise to true value (short run)

<div id="banner " class="inline-block">
    <img src="./documents/fppgv.png" alt="plots" width="300" height="250"/>
    <img src="./documents/fppgw.png" alt="plots" width="300" height="250"/>
    <img src="./documents/fpogv.png" alt="plots" width="300" height="250"/>
    <img src="./documents/fpogw.png" alt="plots" width="300" height="250"/>
    <img src="./documents/fponv.png" alt="plots" width="300" height="250"/>
    <img src="./documents/fponw.png" alt="plots" width="300" height="250"/>
    <img src="./documents/fpgr.png" alt="plots" width="300" height="250"/>
</div>

Both cases, even they are short runs(only 5k episode organized in 50 episode batch and update for 100 times), some of the parameters recovered well. For example, process gain in velocity and angular velocity from both cases are close to the true values.
If have to say, the observation noise fixed group seems to be better at recovering other parameters.
But this is a very short run, we cannot conclude the performance based on this.
This is just an example showing that with some parameter fixed, it is very easy to recover other parameters and much more accurate.  

## Loss

<br>
<div id="banner " class="inline-block" float:left>
    <img src="./documents/aoloss.png" alt="plots" width="300" height="300"/>
</div>

This is sample loss vs time for some runs.
The action loss is dominate here, and observation loss is small.
Actually, only training the inverse with action loss can achieve similar recovery of theta in short runs.
However, not true for only train for observation loss.
I added a record keeping function to keep the ratio of the action loss and observation loss just then.
We should be able to see the ratio chanage soon.
The ratio change might indicate the action is more important for the beginging and observation loss is more important for fine recovery.
If so, we could manualy increase the ratio, just like an adjustable learning rate to optimize the inverse.

## Informal discussion

small number of episodes per batch may reduce estimate noise, as we could not have enough sample to map the variance in each update. this might results in smaller noise and off value gain, because if sampling returns a biased noise results, the gain is then adjusted.

when fix noise to low level for both process and observation, very good gain and radius parameter recovered.

log noise vs std noise. I think std noise is better.

this has not been ploted, but from my expereince seems more often the angular noise is over estimated and velocity noise is underestimated.

### Future directions/improvement

use 2nd order optimization  

fix gains and solve for noise, not working, at least not in short runs.



# Second Part: Improve the Policy suface to Allow Better Maxium Likelihood Calculation

In this firefly forward task, the agent choose an action of velocity v, and angular velocity w, at each time point t.
I will refer to them as vt and wt.
In this way, the agent leans the optimal policy from the forward training.
The policy network is built using a fully connected network structure, with layer relu activation, and a tanh out activation.
One problem with the layer relu activation is, because the relu's natural property, it will produce a rough action output surface though the tanh output is smooth.

<div id="banner " class="inline-block">
    <img src="./documents/placeholder.png" alt="plots" width="300" height="250"/>
</div>

Having such a rough policy output is not good for the inverse calculation of maximazing log likelihood.
This is because when maximazing the log likelihood, we compare the action difference between the teacher agent (the one with true theta we want to recover), and an agent with estimated theta (the agent the that we are optimizing to recover the true theta).
If the action difference is small, we cannot extract useful information to update the estimated theta and move it towards the true theta.
Thus, ideally, we would want a smoother policy surface, such that the actions are different on different cases.
This allows us to have gradient to update the estimated theta.

## Ideas to Smooth the Policy

There are some ideas to smooth the policy surface from our discussion.
First, we think the problem is with relu's shape.
Relu will brutelly bend the nagetive protion of the input to 0, so we would imaging the output of relu will have distinct bend as well, just like the folds if you fold a piece of paper and unfold it.
So, in this case, a smoother activation function may help.
We choose selu, scaled version of elu, which provide a softer bending on the negative part.
Second, the velocity output vt from the policy network is from -1 to 1 range, as shaped by tanh out activation.
But the task constrain the ouput to be 0 to 1 range.
This results in the vt output is hard clamped.
We think instead of a hard clamp, a soft clamp such as sigmoid will do a better job at reshaping the vt output by policy.

## Selu

The agent with selu layer activation has a smoother policy surface.

<div id="banner " class="inline-block">
    <img src="./documents/placeholder.png" alt="plots" width="300" height="250"/>
</div>

This is a sample figure comparing the selu agent and the relu agent.
The selu agent takes more time to train, and it was train 3 times more episodes than relu agent.
Looking at the angular velocity output, the two agents are similar.
They both want to maximize the angle turning rate.
Notice that the selu policy has a smoother region around 0 angle and around 0.2 to 0.4 distance range.
In this range, the agent is relative very near to the target and has a very small relative angle to the target.
I would expect the agent's optimal decision to be having a smaller (non max) angular velocity.
Looking at relu agent, this is not seen.

More obviously, the velocity ouput by selu agent is much smoother than relu agent.
Instead of having max velocity all the time, the selu agent choose a low velocity when the target has a large relative angle and large distance.
This could be a trade off between the velocity and angular velocity.
Assume that we want to maximize the velocity (since the reward is evaluated by distance only), we would take the maximum velocity that 1. not exceeding the velocity max, 2. make sure that the relative angle not going up.
In this case, we can calculate such a velocity for every distance d_t, relative angle theta_t, vmax, wmax.
This results in each move is the similiarity triangle, and the v_t is exp decreasing.
This means the agent will be near the target but will never reach the target.
To balance the trade off between the velocity and relative angle at the next time point, it is obvious that the agent should take some lower velocity at some cases.
I am also trying to solve the simple sysem without goal radius and noiese using math.

## Smooth the Policy Output

The policy output to an action, consist of forward velocity v, and angular velocity w.
I will refer to them as v and w.
Regardless of the layer activation, the out activation is tanh.
So, the v and w raw output are in -1 to 1 range.
The task environment uses v and w in this -1 to 1 range directly.
If we do not apply a constrain on the velocity, meaning the agent can choose to go backwards, the agent will first backup then move in a curve for large relative angle goals.
Their trajectories are like this:

<div id="banner " class="inline-block">
    <img src="./documents/placeholder.png" alt="plots" width="300" height="250"/>
</div>

However, in the actual task, although the monkey can choose to backup, it seems they haven't discover this trick.

If we add constrain that only allow the agent to go forward, now this v should be in the 0, 1 range instead of -1, to 1.
Thus, if we smooth the v output, it is equal to add another out activation to the policy.
tanh then sigmoid instead of a single tanh.
In my opinon, this will not greatly smoothe the actions.
I will do some tests soon to test my hypothesis.

## Optimal Agent

We would like a policy of smooth surface, so that the actions are slightly variant and we can compute action loss and coresponding gradient to adjust our estimation of theta.
But, at the same time, we should make sure the policy is optimal.
So far, we have serveral ways to roughly determine if the agent is optimal or not.

First, on inter episode level.
The agent should be able to skip hard trials that require more time to finish.
This is seen also in real monkey's experiments.
Instead of trying to get to the far way goal, the agent could gain more reward in a fixed amount of time by skip this trial and only doing easier trials that takes less time to get reward.
This skipping behavior is depended on adding the episode transition into the reply buffer.
This makes sense because the agent has to know if it stops, the belief will transit into next episode.

Second, on single episode level.
The agent should overlap its belief estimation and goal radius as much as possible.
In other words, the agent should want to stop at the edge of the goal radius, but taking its uncertainty into account, it should actually want to stop near the edge when have small uncertainty, and stop at the goal center if the uncertainty is large.
A way to quantify this is to calculate the overlap ratio of uncertainty overlapping with the goal radius weightly by the reward distribution, over the reward distribution.

Third, on single episode level.
The agent should finish the episode and reach goal as soon as possible.
In other words, the agent wants to maximize the reward over total time spent.
The agent could choose a longer arc, despite the arc is longer, it may take less time to get to the goal.
The agent could also choose a shorter arc, as the total travel distance become shorter, the agent might get to the goal sooner, but it has to balance this between the time spend at origin to turn around.
The agent could choose to backup and then go, as seen in training.
To achieve optimal policy to maximize reward over total time spent, I trained some agent with reward=reward from reward function * gamma^total time spent.

One problem I am facing now is, to calculate the theorical minimal time needed to reach the goal, assuming no uncertainty.
Would the fastest path be an arc?
Would it be combination of arcs?

The other problem is the task setting.
Assume the agent has infinte w, the agent would turn to facing the goal at first step, and then travel in a straight line.
Assume the agent has infinte v, the agent would either turn to facing the goal and then use one step to reach the goal.
If the agent has a large but not infinte, the agent would use combinations of max w and small v, to save some distance during the turning.
Of couse, in this case, the agent will use apporate v because large v in wrong direction will likely to offset the w turn.

For current settings, the optimal policy is bottlenecked by max v.
In other words, in most cases, even if the agent choose to go max v, it will not affect the angle relative to the goal very much.
Thus, the agent likes to choose max v to minimize the total time spent.
I hypothesized that if we use increase v range, we may be able to see that the agent choosing an intermediate v instead of almost always choosing the v max.
Right now, I am training the agents with larger v gain with w untouched.



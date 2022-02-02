# requests
agent can use differet learning algorthims, diff input args
env can take corresponding output from agents and update  

# slide notes
should be very much all 'motor' learning, could it extend to similar joystick task?  
## DQN
### memory replay
no more chain  
offline nn learning  
### target Q
as a seperate network from Q  

# characristic
trial and error, no dataset needed  
delay reward  
markrov chain, MDP, POMDP
# optimaiz goal
ps,a=ps*sum(pia,s*psnext,sa), a trajectory  
max sum of reward in all trajectory  
if episode, max sum E p(s,a)*r(s,a)
if infinite, cant sum, can only estimate  
# Expectation
smooth. even under discrete reward  
basically, is another rewrad function   
# MARKOV
memoryless  
finit states, n  
state transition matrix P, n by n  
# MARKOV decision process
finit A  

# value fucntion
exp decay with steps  
denpends on state
max V of next state  
not given next action  
# Q action function
denpends on s and a  
so in term of states, directional 
max Q and choose that action for next state  
eg, greedy action selection max Q  
eg, egreedy, random action with p=e, and 1-e for max Q  

# model based 
## policy iteration
1. eval pi, know state action pairs (V) under current policy  
max the V of each states  
2. improve, use better pi (0 or 1) for each (a|s)
## DP
eval, know s,a->next state's V  
bellman state transition  
current reward and next state sum V expectation  
update V   
##  Value iteration
max Q  
update V=Q 



# model free 
## full fitted Q, off policy  
drv from policy iteration  
instead of improve policy pi to give larger V and then chose  
max expectation, by sampling next state  
s,a->r,s', under pi.   
max Qs',a' instead of Q from pi, by greedy(min Q-r+newQ)  
at certain time, fit, can use all prev data, better sampling    
## on policy Q learning
for every step, fit  
## actor critic
actor theta, output action. loss: logP*td error  
critic w, output Q. loss: sqrt td error  
td error, r, Qnext, -Q  
## MC
max V or Q  
bootrap to estimate Expectation  
limitation, episodic  
update every episode  
## TD
update V, by diff of actual Reward and estimated V  
using gamma, eval all prev non absorbing states V  
update every step  
## on policy sarsa
max Q, Q driven pi.  
s,a->r,s', under pi.   
s',a' under pi. 
update Qsa, using r and Qs'a'
## policy gredient  
rely on random?  
only based on rewward and state (no Q)  
need small action space (2, here)  
can use carefully designed reward function, on each env  
log Likelyhood. loss: logP*V  
Deep Deterministic Policy Gradients, used orignally  
## Dyna
model free RL agent  
model based update value/policy  






# inverse
rl, from dynamics and reward function, learn action
inverse rl, from state and action (labeled data), learn reward function  
inverse optimal control, action and cost fuction, dynamics
inverse r control, model based from state and action, learn reward and dynamics  



# msi
attention
attention guides MSI via top-down selection of inputs
bottom-up integration can occur pre-attentively capitalizing on temporal and spatial correlations. 

bot up
attention leads to reweighting of sensory information
audio-visual interactions guided visual attention


mathmatical mind reading
monkey brain in vr, how brain think
infer thoughts from behavior
relate thought to neurla activity
example task, navigation in vr, how brain solve it
innovation, rational assumption, 

forward model
fixed framework, parameteraized and adjustable plug ins
raional assumption, train agent family

inverse model
maximazing the log likelihood
gradient, cmaes

refinement
answer detailed questions without sacrifasing explainibility, while keeping unified 

trial by trial var:
obs/process noise
meta trend, eg fatigue, reward satisfaction level
time by time var:
action noise(bounded rational)
attention(not paying attention keeps same ctrl) (eye tracking)

multi info integration
f,b cost means constraints

representation
representation learning

representation vs computation
save it in cache or compute again



# msi thoughts

haptic importance
vr levels:
feel like im in another world (immusive)
feel like im another person in another world (not immursive)
while vision gives the fov of the other person, haptic give us the sensor of presence, thus much more immursive

how can msi used in vr
msi is computated by the brain in nature.
if given realistic enough sensory input, then we dont really need to study msi to achieve good vr experience.
seems we cannot reproduce realistic enough sensory stimuli, either not fast enough, or detail enough.
1.thus, understanding msi might help so that we can reduce the render for unnecessary inputs? to achieve better spatial and temporal resolution
2.the feeling of mismatch or immursive is subjective. understanding msi can let us model the latent feel quantitively, thus to be used in better vr design?
we know the state(meta data of the object), we know input (what we give), we have a prior of latent state (real world dynamic, we assume the subject have a good sense of real world), we have the behaivor output of subject(actions), we want to infer the latent state posterier.



# how msi works
prediction + new obs, update -> postorier (result)
the prediction and obs could be a lower level representation
using the low level rep can fill missing sensory dimention
integration can be linear weighted sum (including kalman filter), or nonlinear methods (eg, uncertainty is too large or err too large, just trust the new obs)

# haptic notes
mechano reseptors:
low freq vib
high freq vib
prop and stretch
touch and texture

obj meta data
shape size weight
softness
smoothness

shape, size and (smoothnesss) can be see. others can be estimated.

# thinking direction
what they need, hope to see

what they do and publish
how those can be fit into what they need
word what i do into those


# friday questions:
research是怎么被apply的, 是用学术的结果做产品，还是利用产品做research？

比如multisensory integratoin，如果每个感觉做到拟真， 就不用考虑integration，但是为什么我们想知道integration， integraiton能做什么

what are the typical research problems? time intergration or spatial integration

different directions/products? the hand reconstructon waist band, vs gloves

第一面 brian simpson
轻match 可能的project
focus on you and your research background 
your various research experience
how you approach / think about research problems.
research background, publications and interests. 
the work that you’ve done in the past
what you hope to do in the future. 
interviewer’s google scholar as well as the posted job description
your specific contributions to your research 
technologies and methodologies used
想看到的research topic
    相关的经验, basic models, 学的慢的东西要会
3-5 ppt 展示model


第二面 jess Hartcher-O'Brien
technical aspects of your research. 
research skills such as experimental design, data analysis, and perceptual research 
your resume 
questions about the work you have done. 
可能是mentor
会问一个potential project考察怎么做
想看到的research skill 
    技能，最好完全相关 都会

# prepare
## keywords for skill match:
1 from job desciption, from interviewer pub and description, search those keywords for reviews
know the keywords in reviews.
## keywords for experience match:
reivew some neural stuff. 

## make some slides, with nice animations.

# job description keywords
expertise in haptic perception
    no
multisensory integration
    vis vestibular
motor control
    policy and optimal control
invent technologies for interaction with the augmented world
    sensory info -> integration -> motor control
    vr -> the precetion we want -> sensory stimulus -> human
perceptual and cognitive
human perception and action

Formulate research questions and design experiments. defining problems, exploring solutions, and analyzing and presenting results.
    just basic research
Collaborate with researchers and engineers to implement and conduct experiments and collect data. Interpersonal skills: cross-group and cross-culture collaboration. Communicate research agenda and 
findings to collaborators across disciplines.
    must be clear, oral, present slides, plane language, write
Analyze and model data.

psychophysical research methods
computational modeling
Proficiency in C++/C# and experience with virtual reality
    c#, scrip the vr task
    unity, basic testing, havent start blank.

# bs keywords
audio
head-related transfer functions
virtual synthesis technique
locolazation
noise
audiotory learning
eeg state

multisensory perception 
multimodal information
design principles of multimodal interface development for output
how information presented to multiple sensory systems might be integrated to enhance understanding, 
    optimal integration, less uncertainty, causual inference for missing dimension
how sensory/cognitive processing capacity might dictate the best approach to distributing information across modalities, 
    eg, one or sensor is bandlimited, then snr is smaller, by optimal inference, takes less weight in integrated preception?
what and when to present it based on the environment, the specific task, and the state of the user.
    what, attention



# jess keyword

Haptics
Time Perception
Audition
Multisensory Integration
Temporal recalibration

Cognitive Neuroscience
Experimental Psychology
Behavioural Science
Behavioral Experiment
Behavioral Neuroscience
Behavioral Analysis
Auditory Perception
Spatial Cognition
Psychoacoustics

haptic amp, freq, modulation -> bandwith?
audio haptic, rear and front, tell uni or multi
audio visual, optimal time integration, duration


# cv keywords
shell, pipeline watchdog sevice, hpc, etc 
C/C++, ardruino, FPGA, onnx application, spike sorting functions.
HTML/CSS/JS. some front end visualization and control
MXNet, a virtual rat project. 
OpenCV, ONNX, rat video tracking

explain agent and subjects' task relevant intentions, assumptions,
Previous related theory work with my contribution has published in 2020 NIPS, 
    define and refine the problem to generalize to broader area, (using rl instead of lqg)
    refine the theory by testing on simpler cases (eg, 1d)
virtual reality.
    unity, c#. 
    help extend the task by writing c# scripts. (perturbation, etc.)
Model multi-modal sensory integration using probabilistic graphical model. 
    pomdp, feedforward feedback, 
    next step model, finite time integration?
hierarchical RL
    high level skip instructor, 
    low level policy
SAC and TD3 algorithm
customized experience replay
    like piority experence replay, but pioritize not just for update amount but also for reward
neural activities, in progress
    encoding, neural representation
    recoding, update dynamic
    decoding, policy


spike sorting and pipeline for multi array
    Split and distribute the raw data into workers, 
    Butterworth filter and spatial whitening, 
    mean-shift cluster the neural spike events
    deep learning to merge
    intermediate results are stored and maintained in MySQL database.
Analyze neural data and explain rat decision making.
Model the rat sequential decision making task as a nite state machine with uncertainty.

collect rat behavioral data by self-developed real-time 3D tracking software and infra-red sensors. 
    dual camera cv, fpga, calibrated by infra red
Fit model and analyze the animal's perception of value and risk.
Analyze neural data with attractor network and show the neural evidence accumulation and value expectation.

rat haptic
    rotation project, 
    bar touch whisker direction, evidence accumulation model.










# 汇报


问题：
kalman filter 推一下
LQG 推一下 我推了过程没推解 不会
irc能不能generalize到其他task 能 用rl代替lqg 可以换obs 换belief update
什么时候converge controlaable
baysien inference什么时候不是最优 causal inference
在这些时候 用什么方法 提示nonlinear causal inference
怎么vr模拟重力 摩擦力皮肤拉伸 反作用力指尖压力
怎么vr模拟敲击 vibration depends on finger length, energy wave propergate and active vibration in sequence
怎么分析一个过去经历的neuraldata, spike sorting
linear regression 
logistic regression
psth
raster plot 简单说说 快忘没了
task视觉换成听觉会怎么样 不准 带宽低 没法predict太远
vr中vr vr中control feedback不在controller 没时间了没说





















# alex notes
could the place cell video be slowed eg 0.5x at first and back to fast speed after several seconds?
when you refer to the number of the neuron cluster activation, the numbers are too small. might be better to refer to color instead? or using some circle/pointer
i really like the example of convexity.
is it possible to peel off the bad polar complex and show the pinch point?
will you metion about directed edge
notations in the community structure of graph slide?
brain structure will have similar neurons, might be helpful to show some stained neurons to show that, the clustering predict similar neuroanatomy result?

# wk 1

## 3.10
### framework
### extended kalman filter 
extended for approximate to a linear relationship.  
### progress
forward env done.  
training an agent overnight  
### todo
read more about [kalman filter](http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)  
think wrapper  


## 3.11
###  review of framework
belief=r,  o-r-belief, r neural rate code  
belief space is larger than r, need to compress into r  
dynamics of r, need to find r capcture belief  
particle, neuron r represent particle trijectory  
weight, mean, var of trijectory  
hypothesis, populatio, set of particles, or distributions   
node, expeactation, connectin, transition  
particle try to find the hill  
place neuron  


### foraging task, firefly task, and atoms
poison parametrizing resourse
mixture of gaussion, kernel, cover the space  
### kalman filter seems important 
### heoslan matrix? 
when negative, is guassion, when positive, is hyperbo  
### check ppc(linear in log), ppc kalman (alex jeff)
embedding space linear sum  
uncertainty vs sum of r, goes down like 1/r  
cost of r linear increase  
rewrd of r has a peak and then decrease  
the subtract of cost and reward will have a peak, so this is the point that has good result will have a smaller cost  
trade off  
what is lqr? quaratic, no uncertainty?  
uncertainty , response space  
trade off is calcularable  
### progress
figure standard baseline  
### todo
at least know what equations mean  
try fit standard baseline  


## 3.12
### firefly inersia task  
tau as a time constant for velocity curve, so the velocity isnt changed immediately  
tau has a effect on where stop  
but for each tau, should have a mean of 0  
seperate vestibular, visual, and combined task types  
question, how to do visual only task?  
if have estimated tau correct, no tau uncertainty, sensory likelyhood funciton changed, so prior weight larger. still tau has a effect on posterier  
have prior, and exact likelyhood, inference correct, but post is biased by prior?  
bias, varinace trade off  
## the model,
estimate tau from baysian, from change of speed of begining of trial and end of trail, at first several trials.    
integrate joystick input with tau estimate to trajectory  
how estimate the 2nd step, using two tau, just use tau in equation, not probablistic  
in this way, no process noise, deterministic  
if using probablistic, estimated parameter instead of believed parameter, provide extra noise  
should be considered when improving  
waber law, distance squared, still not explained the effect  
chaos, highly depends on init states and amp  
## progress
refactor framework  
## todo
ssh  
ssh vscode  
git repo organize  



## 3.13
###  

### progress
refactor details and link to framework    
### todo
check if var correct  
test learning  
visualize using gym  



# wk 2

## 3.16

### reading about algorithm 
### soft a2c
max reward, max entropy  
this make the action from a spike to a board peak  
so encourage other choices to explore the surface  
state value V, loss: predict V -expect Q -logpi (entropy term)
soft Q, loss: predict Q -reward -y*next V
policy pi, loss: KL divergence (expQ, pi)  
### why using target networks, eg target V/Q
Q and V interdependent  
so they are dependent on itself, training become unstable  
target network as a lag or incomplete version of main network  
here in our example, using a tau=0.001 to update targets  
so target is a exp decay lagged version  
### progress
now has tensorboard  
trained an agent  
finished checking all involved codes  
### todo
understand other algorithms and try  
from results, do some intepretations based on algorithm charactoristics  
depends on random, some agents are not learning.  

## 3.17

###  meetings notes
explained framework  
state and image/external diff  
state has theta v w, and image is only x y  
irc needs:  
policy(observation, old belief, task parameter)  
neuron network has the policy and backprob to likelhood  
belief step  
### zhengwei foraging task
red/blue stop happen location  
star as estimation, from max likelihood    
dash line for error  
solid line, gradient descent  
belief posterier of box 1/2, green line for 1/2, triangle for press  
why the belief isnt 1  
posterior from belief, isnt 1. but should be same as reward prob  
reward probability is 1  
debug belief transition: belief-> belief for forward agent = belief posterior  
using linear to apporximate the exp, may not capture all?  
in this case, look in transition matrix and see what the matrix is doing, and finally know the prob  
disufssion should be symetry, high/low should go to equalibrium  
aslo look at egein  
### progress
add some comments and organize the code to look better  
now roughly understand the stable baseline package, should be able to make costume policy and costume algroithm  
### todo
inverse framework
prepare my math equations in note, to show at meeting



## 3.18
### reading about inverse
monkey box paper and ff paper
### todo
invser framework  


## 3.19
### zhengwei paper
loglilihood, consist of the model dynamics  
from the (x,o,b)at t1 to (x,o,b,a) at t  
all summed up as the likelihood  

marginalize the b  
because we dont have a measurement of b from the outside  
we have to estimate a best b  
taking the intergal, to marginallize b  
meaning: the posterior for all possible b, and we get the expectation of b by intergrating   

and we simplify the loglikelihood   
for em. so one part depends only on old theta, and another part for m step update  
this is obersved log likelihood, rather than complete data log likelihood  

start em  
E step, using a old theta, or init random theta, calculate the upper bound  
M step, update theta  

IRC gradient question, why a'  
because theres a' in Z, the normalizing factor  


### progress
inverse framework seudo code done, almost  
has much better understanding of inverse  
### todo
read the book on gnerlized em  




## 3.20
### notes
theta/phi reset every episode. that is, in training we are using a distribution of phi  
theta is estimated, for sure it will have some uncertainty, could be represent by variance  
and if not perfectly trained in terms of ml, or rational instead of optimal in terms of learning, there will be a displacement  
displancement is a vector. together with variance, then the theta is a distribution of phi distribution  

in inverse/testing, only using single point phi, start from single point pertuabed phi, as theta  
the theta will move to phi?  

check gym.box normalization  

### progress
inverse  
see actual files instead  
### todo
inverse  
windows enviroment, powershell, look better and use better.  
seems wfh will last long  
sum up the prev undones:  
read the book on gnerlized em  
add assign param function to ffenv  
read example algorithms and implant mpi in inverse  
eg, in chose action/predict(state dynimacs) part  



# wk 3

## 3.23

### theta and phi
theta and phi  
check hand written notes  
### progress
inverse, ref in code  
### todo
inverse code, finish idealy by this week   



## 3.24

### morning meeting 
reward, have chance, predictable parts  
arrange the figures, if symettry, looks better  
histogram with p value, could represent the p by the color gradient  
dont use the same color if the var is not same  
### regarding belief not going high
stochastic instead of determinstic belief update  
allow for flexibility, but may not be perfect  
belief temperture, policy temperture  
two diff temp, if policy is high, then belief is the threshold for not push the button  
so it should going to a higher belief  
what is the prob for having enough belief but still not taking action? there is prob  
why policy temp not go up when fitting real data  
model the not pushing button as forgoting, not updating, not paying attention to the task  
the idea is seperate it from the belief  
location and box mismatch, for example  
one problem with location data, monkey explore the env and go to box but not pressing  
try to explain the result seen using model paramters, such as fast dynamics  
### tips on model fitting
sometimes, if the model has some problem, we think its not perfect model for real data. before fitting the real data, try some simple and simulated data, to understand the dynamics (if change some param, what direction wil some other param go), and adjust model, and fit more complex actual data  

### progress

### todo




## 3.25

###  
   
inductive bias in input  
### progress
inverse  
### todo
inverse  
  


## 3.26

###  morning meeting  
the main point for inverse:  
inverse likelihood, forward, feature space for both   
think about it  

foraging task  
not push the button, could due to  
belief p, action p  
each take some weights, depends on individual time points  

### progress
inverse finished  
inverse not optimal, like to go max first and slowly come back    
### todo
check inverse noise representation  
reorganize  



## 3.27

###  

### progress
inverse debuggigg  
turn out to be i was using a bad teacher  
on amost half of the trials, the teacher spend 10 steps and not getting a reward.  
may because the teacher was trained in a large goal radius setting  
and in inverse, the reset theta happens to give the teacher a small radius, and the teacher is not doing action optimaly  
if the teacher isnt giving out good actions, it would be very hard to infer its belief, so that the inverse model convert to some random wrong point.  

need to train a almost perfect agent as teacher  

things to consider  

how fast should the goal radius go up? right now, the goal radius starts from a rand in min and max, but the max starts from min and going up by steps.  
in this way, the radius finally be a rand in pre set min and max  
should the radius decrease at the end? to train a better agent?  
in my opinion, dependless on gain and noise, the agent should go the goal center if they have correct estimation. in this way, a smaller goal radius is good for forward training for a better agent at least  

theres no penalty for taking action, so the agent should move at high speed and try to get reward, but to some extent that bounded by episode time. in other words, the agent start to learn doing things fast, because it has to finish move within time to get reward. but after that, since its already fast enough for the task, it wount learn toward going fast  
following this, the goal radius should only affect the speed very early on. the agent may learn when within radius it do not have to move to center. but early on, noise and gain isnt well learned yet, and goal radius is relatively rand within small range. so, no effect on this  

teacher is trained on a distribution of tasks, if thinkin different theta are diff tasks. finally, the teacher take action on a random phi, but phi is in this distribution. in theory, the teacher should give good actions. what about training teacher only on one phi, so that the teacher is for sure give good actions, and teach the inverse?  
im thinking, because the teacher only predict action on very correct belief (it only know one theta), if the student have a false belief, it may just give a no better than random action. 

### todo
make inverse fully working  
then think about make it faster  


# wk 4

## 3.30

### LQG control

the kalman filter, takes noisy input and previous state and deciding the best next state. In this way, we have a best estimation of the next state mean and var.
the linear qudratic control, takes noisy controls and make them converge.
need more reading.

### progress
summary  
plots  

### todo
organize code  
add explain in summary, while refer to code  



## 3.31

### 

### progress
reading, think about other optimizing way  
try to understad optim control stuff  
coding  

### todo




## 4.01

### thinking
considering the gain and noise, for example in process, the agent makes a move, and it would have been a straight line in space if without the noise.
with a uncertain v noise, the agent move will be a distribution lenght, but same direction.
with uncertain w noise, the ageent move will be a fan shapped.
have exact same belief, will make same action, but next belief may not be same.
with a good theta, the next x distribution will largely overlap, so as the obervation.
with x, o, we have a distribution of belief but we take the peak.
with this peak, agent makes another action, only depends on belief, no theta.
this difference in action, is then from distribution of x, o.
x, since same action, ideally the agent next x distribution will over lap with teacher, so enough sampling solve this.
o, since same action, ideally agent observation overlay with teacher observation.
previous b, at t=0, b are the same. diff comes from x (pro) and o (obs).

### progress
Inverse

### todo

## 4.02

### morning meeting
reviewer question
why belief stochastic, 
ff, belief update deterministic, inforward.
but no observation noise known, latent to us.
so this gives some noise to belief. 
should belief stochastic on itself, or depends on obs.
ways to add noise on belief, add noise, and use a sigmoid bound to belief.  
thoughts, want to add random to belief to achieve random in action.

how to optimize if like this: want to get max reward, want to min the min reward. 

solution, metion both. instead of stochastic belief update, have the stochastic in noise and used in belief in another version. 

samping on obervation to get belief posteirior

keep belief low, and observation needed to explain belief

belief loss, could be think of loss in brain when communicating, bounded rational.

in inverse part, action deterministic, 

now, aloss, oloss. could do instead, having aloss be a dealta function, 0,1.
all error should be due to observation.

### thoughts
like gan training the generator. 

we tell if the action is correct, by o , 1.
let the netowrk, in this case the pomdp, generate action based on theta given x, give to policy, and the adverse aka we, tell if action is correct.
so, the generator is trained, to move theta, so that it generate a correct belief idealy, which will give to correct action.
problem. if it generate an incorrect belief, it may still give a correct action.
problem. in gan, generator generate from random noise and x, try to minimize the D(x)*(1-D(G(noise)))
here, log(  D(a|x)*(1-D(G(noise)))
log(  D(G(a|b))+D(G(noise))
log(  D(G(a|x,a,o)) + D(G(noise))


### other thoughts



### progress
plots and tests
### todo
plots and tests


## 4.03

###  

### progress
optimizer
### todo



# wk 5

## 4.06

###  

### progress
test log trick  
think about loglikelihood  
plots, better  
slides
### todo



## 4.07 -4.09

### potential questions

meaning of goal radius?
first, easy training.

reward fucntion?
first evaluate actual position if in radius
then use belief to decide reward amount
more reward if less uncertainty, and more reward if beief reach center.
actually, use the beleif center reweighted by the covariance and reward std.

real meaning of observation?
no obs, like walking with eye closed.
obs, like with eye open.

why not using input/image?
the framework isnt final and we are thinking about adding that. now with the flexiable structure, we just need to create another env and train agent with it.

should there be a memory term in belief?
the episode is short, so no need.

true theta and phi and recovered theta?
phi is env param, not involved.
true theta can be any theta within pretrained range, to make sure agent is rational.
recovered theta, given enough data, will be true theta. but with limited data, it will be optimal theta given the data.

loss function?
for this version, im only using action loss. that is, minimize the diff of the action from teacher, and agent with estimiated theta.
if with the input, we will have observation loss, and that will be an easy change for this framework.

why belief not in agent?
framework constrain. we can put the belief with agent, but that causes a lot of problem because the framework is not made for pomdp. so for now, as we are still in testing phase, i just put them seperately. this only affects naming and everything is the same.

why use short run?
although the short run does not provide a final theta, but it gives us some insight.
in real setting, if we want to use this method on real data, we face the problem of not enough data. so, we are comparing ways that can extract as much info from limited data. short runs can test this. also, im running this on my computer, it would take forver for a longer run.


## 4.10

###  presentation
feedbacks:
sigmoid is same as tanh  
next we will have bahavior data and neural data, soon
some other handwritten notes, see notes.


# wk 6

### progress
loss surface plots
sigmoid and normalized show not much better.
check how log noise surface look like.
check if bug in representations.


## 4.28

### morning meeting
neural data, trying to predict choice/reward
using reward p itself, or from neural regression.

see picture

model lapses?
model a fogetting process, 3 different distribution of reward, cycling, and agent forgot the stages in the cycle.

when forget, not empty belief, but reset it to an equaibrium.

probobility of forgoting, if const, its easy to find a eq and reset to eq.

process and policy interdependent.

eg, eq is empty the memory, the lapse is not under agent control, the parameter will be a task parameter, agent estimate the lapse rate for example.

real meaning, not paying attention, skip update/observation, or skip the prediction and only use obs, or prob to jump back. could model them all

p check is the regressed decoding of neural activity to our predicted reward prob.

what is cca?

having a residual of p-b, and now covar is more meaningful? 

anaysis ways, remove some compoment and see regression. part of casual relationship methods.

regression agaisnt b, show if p doesnt help.
regression again p, b will help.



make sure model is trained
	cirriculum or just add time
	plotting functions

representation of belief, multi
	how agent represent/compress belief of multi firefly
	representation learning
eye movement, new observation
	eye movement probably signals the 'belief'
	eye movement may helps short term memory
	side information?
acceration, 
	shoudl be easy. just the same frame work.

background,
inverse need some differantiable policy surface to work better.
neither selu or action smoothing's result is not exactly what we want.
maybe the optimal policy is not smooth?
	math of arcs
	 to prove: better policy takes less T for d
	is it = fix T and better policy can reach further goals?
to prepare integrate the real monkey data,
	monkey joystick
		need to find some way to map it to rectangle.
		seems they use some filter. is it possible for 'reconstruction'?
model action output,
	real task env
		reward spare, too long, hard to learn
		hierachical or reward shaping, optimal policy?
		change in Q surface and not ultilize the learning before switching
		encourage agent to move towards the goal, by N(mu=goal,std=0.1)
		encourage agent to stop, by fix reward 0.1
	HER or PER?
 	gamma related to skipping? either way, gamma need some attention
	agent shouldnt look into inifinite future, or next several eps. 	


Qt=rewardt+gamma*Qt+1, where Qt+1=rewardt+1 +gamma*Qt+2 ....
	recursive belman function. 
	if large gamma, for a init state of hard ep:
	if skip, Q=0+gamma*Q(next ep init)...., where Q of next ep = gamma^T *reward expectation, where T is ep length expectation.
	if not skip, Q= reward expectation*gamma^T +gamma^T*Qnextep....
???so, seems if the reward expectation of this 'hard' ep (usually not hard, just longer) is large enough, agent will skip less.
if gamma is large, looking into infinite future. optimaizing on total R/T? exp of reward of infinite future is same. 
if gamma is apporiately small, say, only look into next ep. if skip, exp reward larger. if not skip, low exp.


 1.if bound, sampling.         2.if slow, 2nd order natrual gradient, newton
after some train, freeze gain and train for noise.
think about t0, x0 and a0. to get b1, first Ax, APA+Q, then use o1, which is Hx1, R.
update, we mix APA+Q by KH, containing P,Q,R. and b=x+K(z-hx)
K, how much we trust obs.
because from a to belief isnt a one to few, we do it forward direction:
sample pronoise, obsnoise, and a obs point using this obsnoise that gives b, that action from this belief is within a small deviation from action. (soft 0,1) (sample to get a belief good for this context)
because we know the range of the noise (the bound), we only need to sample within the gaussian circle. And because this has real world state meaning, if within range, the policy likely to give similar action.
adv: real animal data is not as many as we want. here, we explore into depth in the data.
problem: there is a theta in belief that give to policy. we dont know how much the policy depends on the theta.
I need some time to finalize it, and implant. 1+1wk

next step, chose from these(check heissian,  new, )
architecture:
env-agent/policy pair, as data generator and iterator. 
so two set for simulation, one set for real data.
algorithm, call the pair to collect teacher data, and call the pair to iterate, calculat loss, update.
it has universal function such as log the theta change, loss change, grad, ep.
it has specific function to initialize theta and phi, and pass to the env agent pair


LQC: minimize the input error sum, minimize the integral, so that output is stable and flat.
kalman filter: predict future, and update prediction by actual obs.
compare, LQC, give the input update, and actual update the future by minimizing noise.



8.26
questions
	covariance matrix singular, cause: maybe: noise param too small
	how to preciesly get the cause, and if this is affecting the result
	hwo to interpret when cov is singular, having eigvalue 0?
	
	action cost inverse has heissian singular, while orginal no action cost hessian is not.
	what means by hessian is singular? some eig value are 0. some parameter 2nd derivative is linear combination of other?
	theta cov=- invserse(H), how to represent a multi dim gaussian in pc space?

results:
	multi initial point solving for one true theta
feedback:
	bug of hard coding error, not calculating grad for newer 2 rows, resulting in 0 as eig value for heissian.



aug 31
monkey data: 
	trials missing first several data point. if dt=0.1. first 0-3
		assuming linear increase in v, w for these points？
	controls, sometimes exceed 1 (due to noise and filtering)
		cut it to 1
	at least for this monkey, hes not using full v. and making up and down w. (not s shape trajectory tho)
		same for easy trials.

plan inverse:
	we know process gains, goal radius. 1. fix these, solve for noises and obs gain. 2. solve for all, use these as checks
question:
	continue on ploting the confident intervalhttps://wifi.xfinity.com/default.php#find-a-hotspot
	what is the correct way to convert high dim cov to pca dim?
	diagnal: simga in u, v should be weighed sum of n dim sigmas, so: u=vector*u eig vector	
	how about rotations(correlations)?

	after the noise samping change:
	noise parameters are still in a ralation. how to untangle these?
	if large obs noise(not rely on obs), when further increase obs noise, performance is not affected.
	if small obs noise(rely on obs), increase obs noise change actions.
	d action/ d obs noise gives info about obs and process noise ratio
plots:
monkey trails, 

start y vs start v. correlation
start x vs trials, distribution hist
start y vs trials. distribution hist
start y vs w. correlation

feedback: think about the matrix operation. so it would be just ABAT format.

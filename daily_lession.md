
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
### todo
make inverse fully working  
then think about make it faster  


# wk 4

## 3.30

###  

### progress

### todo

## 3.31

###  

### progress

### todo

## 4.01

###  

### progress

### todo

## 4.02

###  

### progress

### todo

## 4.03

###  

### progress

### todo



# wk 5

## 4.06

###  

### progress

### todo

## 4.07

###  

### progress

### todo

## 4.08

###  

### progress

### todo

## 4.09

###  

### progress

### todo

## 4.10

###  

### progress

### todo



# wk 6

## 4.13

###  

### progress

### todo

## 4.14

###  

### progress

### todo

## 1.15

###  

### progress

### todo

## 4.16

###  

### progress

### todo

## 4.17

###  

### progress

### todo



# wk 7

## 4.20

###  

### progress

### todo

## 4.21

###  

### progress

### todo

## 4.22

###  

### progress

### todo

## 4.23

###  

### progress

### todo

## 4.24

###  

### progress

### todo



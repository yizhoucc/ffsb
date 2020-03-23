
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

###  

### progress

### todo




## 3.24

###  

### progress

### todo

## 3.25

###  

### progress

### todo

## 3.26

###  

### progress

### todo

## 3.27

###  

### progress

### todo



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



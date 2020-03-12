
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

### todo
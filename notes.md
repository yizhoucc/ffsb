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


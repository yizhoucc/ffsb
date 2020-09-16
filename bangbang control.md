
# bangbang control

this is meant to be an expension of the firefly dynamic document by xaq.  

## basics

in a 1d line, agent is constrained by a v max.
we can describe the control of the simulation agent as:  

v_t+1 = a*vt + b*ut  

the control u is in the range of (-1,1).
thus, it is easy to compute the vm possible for the agent.
for example, v_2 under max forward control will be:  
v_2 = a*(a*(b0)+b1)+b2,  
    = a^2*b+a*b+b  
    = power series of a *b  

thus v_max = 1/(1-a) *b.  

## task constrains

in the task, we have serveral constrains.
in the simulation, we need to follow the constrains, to make the parameter in range such that the task is possible for the agent.  

we have T, total time of a trial.
agent has to finish with in this time period.  

we have vm, the max speed possible.
this is the process gain previously in the velocity control. 

we have dt, as the sampling frame rate.

## derivation of the parameter range

to derive the approprate parameter range for a and b, under the constrain of vm, T, dt, we have the following meta algorithm:  

1. calculate the b from vm and a.
2. find the switch point of the control input.  
3. calculate the max distance d possible for the agent to travel, within T.
4. if d is larger than our max goal distance, then the param a and corresponding b is valid, under this vm, T, dt, and d.

explain:  
from the basic, we know vm applies a constrain on a and b.
for one a and vm, there will be only 1 b.  

in the bangbang control, the agent will use max forward control at first, and use max negative control to stop at the desired target.
so, there will be only 1 switch point, where the agent switch from 1 control to -1 control.
the switch point is calcualted using binary search of the agent final velocity.  

in the task, we typically set a target with some distance to the target.
while the task is 2d, we can approximate this distance by the arc length.
for example, regarding to a target on pi/4 to the left and having a distance of 1, we can calculate the disance travel as:  
d=1/4* 2*pi*1 =pi/2  
we can then use this distance as a rough estimation, to determine if the param a and b is possible for the agent to complete the task.  

## specific derivation for the parameter range

in the acc control equation, we donnot have a constrain on a's lower limit.
if the a become 0, then we will have velocity control.
however, we do have a possible uppper limit 1.
if a >1, then we will not have any control over the max v.
so, we start with a in range [0,1], and trying to decide which a is not possible.  

for any given parameter set (vm, dt, T), we can calculate the max d for the agent as described.  

1. calculate the switch point:

    initialize the swtich point as T/2, with s in [0,T]  
    from the sign of the end velocity, change the s bounds by half correspondingly  
    when s in [lower, upper] converge, (upper < lower+2 ), end  

2. for t in 0 to s, interval=dt,  

    do: init the d as d=0  
    calculate the v with max forward control  
    d=d+v_t*dt

    for t in s to T, interval=dt,  
            
    do: init the d as d=0  
    calculate the v with max negative control  
    d=d+v_t*dt  

    return d  

3. compare d and target distance.  

## summary

so, in the actual task, we first random theta without a and b.
with the process gains, we calculate 
# meeting 

# our state 0 

orignal way: 1d case, gain=g, action_t=[vt]
state_0=[position,v]=[0,0]
state_1=[g*v0,g*v0]  or [g*v0,null]
position: position is correct 
v: this v_1 is incorrect. 

should be v0=1 and state_0=[0,null]
or, state_1=[1,null] because action_1 is not givent yet.


## under this, we cannot run update in 1 step while follow markov
because:
say b(belief mean)0 =[0,0] same as state0
predict b_1 = [0,0], as we use previous corrected v to update new b
updated b_1= [0, corrected(g*v0)]
while state_1=[g*v0,g*v0]

predict b_2= [corrected(g*v0), corrected(g*v0)]
updated b_2 =[corrected(g*v0), corrected(g*v1)]
while state_2=[g*v0+g*v1,g*v1]

the position part of bt is lagged by 1 timestep

## if we use the action not previous corrected action to predict:

b_0=[0,0]
b_1 predict=[g*v0,null]
we dont have the v1 yet, and we cannot and should not update/assign values to v part at b1
the idea is, if we want to use the corrected v, we should not the action v directly to predict
ofcouse we can update with the correct v for the difference, in 2 steps

## run kalman filter with b0?
state_0_init=[0,null]
b_0_init=[0,null]
state_0=[0,g*v0]
predictb_0=[0,g*v0]
update b_0=[0,corrected(g*v0)]

state_1=[g*v0,null], then [g*v0,g*v_1]
predictb_1=[corrected(g*v0),g*v1]
update b_1=[corrected(g*v0),corrected(g*v1)]

state_2=[g*(v_0+v_1), g*v_2]
predict_b_2=[corrected(g*v0+g*v1),g*v2]
update b_2=[corrected(g*v0+g*v1),corrected(g*v2)]
this is the 1 step version of seperate update:
final vt = predict then update v
final position_t = predict using updated v


## assign the action to state 0 then run state dynamics?
state_0_init=[0,null]
b_0_init=[0,null]
state_0=[0,g*v_0]

b_0=[0,g*v_0]
predict_b_1=[g*v_0,null] null could be 0
update_b_1=[?] again here we should not update because it will result in b_1=[position_b,v_b] not on same timestep



# some reward functions

# curriculum
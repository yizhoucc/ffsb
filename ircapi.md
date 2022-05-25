# api for the irc pacakge

## overview
infer the parameters of the subject's model, given subject's behavioral data.
(also given the task enviroment, parameterized policy)

## input

### subject's data
#### actions
list array. list of subject's actions per trial, in format of a t x action_dim array


#### states
optinal but usually needed for perturbation trials
list array. list of subject's states per trial, in format of a t x state_dim array

#### tasks
list vect. list of subject's target per trial in format of a target vector. 
could include other things needed to define a trial condition, such as optic flow density.


### task environment
defines the task dynamics.
vect next_state, task_function(vect cur_state, vect cur_action)

### internal task enviorment, belief
define the belief dynamics.
vect next_belief, belief_function(vect cur_belief, vect cur_action, vect cur_theta)
where the theta is the parameter defining a subject's model, the one we hope to infer.

note, using gym.Env to define the task and belief together is easier for forward training and inverse.
but using gym.Env is not necesssary.
because calculations such as computing reward etc are not required during the inverse, we suggest to make a seperate simplified function for the inverse.

### parameterized model family
defines the rational policy.
vect cur_action, policy(vect cur_belief)

and that's it!
other minor things that could be relavent are:
inital theta (where we want to start), number of samples for marginalization.
other hyperparameters are relatively less important.




# lab meeting

# last time, we covered the big picture of the ff project

# but, we have found some mistakes
forcous on the current right version, let ppl see if suggestions
not where the orignial is wrong.

the core idea of the ff belief is kf.
core of kf, dynamics, prediction, update.

we have state, of position and velocity.
we use the kf to update velocity, as the observation is only on velocity part.
then, we use the udpated velocity to predict the next position
with the predicted next position and updated velocity, this is our next belief mean

as for cov, cov updated by apa+q, integrating uncertainties in velocity.
a is jacobian that each row is partial derviative of the corresponding var.

in this way, in state dynamics, we have to do 2 steps.
that is, st+1=A*st+B*at 
where the A use the velocity in state, to predict position of next state
B assign the action into the next state velocity part.
in other words, st=(position@t, velocity@t)
for same timestamp t, the position is before the action is taken.

# the timestamp problem, or not a problem.

this is following the mdp, so correct.
but ugly when dealing with giving reward.
because, reward is given when stoped, v<some value.
and if the agent stops, the position should still be updated using the previous v,
which is already in the state.
it create a problem when dt is large, agent may step to far.

in the decision info, we gave relative position, and v@1 (basicaly we gave the st hat)
so, optimally, agent should not directly use the relative position,
but instead, use the predicting next position from relative position@t, v@1
would this be a problem?

# next steps

representation learning, how to represent the belief with less byte

perturbation, that following the mdp

eyetracking, could be a potential source of 'side' information.



# plots and ideas

action distribution of one same task


converged_theta:
tensor([[0.8204],
        [2.6666],
        [0.0667],
        [0.0830],
        [0.9744],
        [2.8752],
        [0.3291],
        [0.0311],
        [0.0496]])
true theta:
tensor([[0.8142],
        [2.2705],
        [0.1336],
        [0.0753],
        [0.7418],
        [2.7885],
        [0.1469],
        [0.0745],
        [0.0530]])

        converged_theta:
tensor([[0.4162],
        [1.4452],
        [0.0948],
        [0.0631],
        [0.3783],
        [2.5912],
        [0.1617],
        [0.0323],
        [0.1445]])
true theta:
tensor([[0.4159],
        [1.2195],
        [0.1053],
        [0.0494],
        [0.2873],
        [2.4017],
        [0.0789],
        [0.1063],
        [0.1433]])

        converged_theta:
tensor([[0.4892],
        [2.8059],
        [0.1592],
        [0.1020],
        [0.6997],
        [1.4232],
        [0.2614],
        [0.1059],
        [0.1997]])
true theta:
tensor([[0.5020],
        [3.1152],
        [0.2094],
        [0.1667],
        [0.5652],
        [2.7202],
        [0.2444],
        [0.1347],
        [0.1987]])

        converged_theta:
tensor([[ 3.8118e-01],
        [ 1.4859e+00],
        [ 1.6396e-01],
        [ 2.5945e-01],
        [ 1.0813e+00],
        [ 2.1964e+00],
        [ 1.9984e-01],
        [-1.1796e-03],
        [ 6.6747e-02]])
true theta:
tensor([[0.4008],
        [2.9999],
        [0.0551],
        [0.0556],
        [0.7186],
        [1.6986],
        [0.1389],
        [0.0752],
        [0.0582]])
converged_theta:
tensor([[0.3865],
        [1.8780],
        [0.0838],
        [0.0861],
        [1.3809],
        [2.5786],
        [0.4487],
        [0.0353],
        [0.0663]])
true theta:
tensor([[0.4008],
        [2.9999],
        [0.0551],
        [0.0556],
        [0.7186],
        [1.6986],
        [0.1389],
        [0.0752],
        [0.0582]])
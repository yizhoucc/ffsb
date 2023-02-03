# save the belief trajectory -------------

_, _, beliefs, covs=run_trials(agent=agent, env=env, phi=phi, theta=theta,task=thetask,ntrials=ntrial,pert=None,given_obs=None,return_belief=True, given_action=actions[ind], given_state=states[ind])
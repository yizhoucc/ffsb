## questions

in gym visualization, the goal is always at center and agent comes from different directions trying to reach the goal. green ring is goal radius so once in range and stop, reward is given. if in range but not stop, goal will be 1 but no reward will be given untill it 'stops'. orange circle is belief xy and gold star is actual xy. am i right?  

what is ln and bn stands for? are pro/obs_varience the only perturbable parameters?    

is pro_gain/noise stands for process gain/noise? is it somehow action gain/noise?  

during initializing, self.b=x,self.P. why P0 here is a small values eye(5) matrix?  


## not that important questions

the task itself is cpu intensive, except when doing update every 500 episode. because its offpolicy, is it possible to run 500 episode at same time and push to memory and then update?  

random seed from arg.seednumber is predictable, but for single agent it is ok, am i right?  

have you try gain/noise independent of episode? or smoothly changing gain/noise instead of random? just curious about this    
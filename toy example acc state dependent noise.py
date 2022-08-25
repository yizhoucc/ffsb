print('''toy example acc state dependent noise''')

def geta(tau,dt=0.01):
    a = np.exp(-dt/tau)
    return a

def getprocessnoisescalar(tau):
    a=geta(tau)
    s=(a**2-a*2+1)
    return s

import numpy as np
from matplotlib import pyplot as plt
from plot_ult import quickleg, quickspine, quicksave





class DynamicSystem:
    '''
    the system
    x' = Ax + Bu + np
        x, state
        ', next
        A, state transition matrix
        B, control matrix
        np, process noise
    '''
    def __init__(self, A, B, initalstate, dt=0.1) -> None:
        self.A=A
        self.B=B
        self.dt=dt
        self.x=[initalstate]
    
    def step(self,x, u, A=None, B=None):
        if A is None:
            A=self.A
        if B is None:
            B=self.B
        return self.A@x + self.B@u


    def step_(self,u, process_noise,t=-1):
        nextx=self.A@self.x[t] + self.B@u + process_noise 
        self.x.append(nextx)


print(''' this is an example acc system.
when use max control, the velocity accerate to a vmax, and distance is increasing accordingly.
''')
A=np.array([[1,0.1],[0,0.4]])
B=np.array([[0],[0.6]])
x0=np.array([[0],[0]])
ds=DynamicSystem(A,B,x0)
# ds.x
# ds.A@ds.x[-1]
# ds.B@np.ones((1,1))
for _ in range(20):
    ds.step_(np.ones((1,1))*1, 0.1) # the 0.1 here is a noise
plt.plot([d[0] for d in ds.x], label='distance')
plt.plot([d[1] for d in ds.x], label='velocity')
plt.legend()
plt.xlabel('time [dt]')
plt.ylabel('distance and velocity')



print(''' lets say we try to reach a point at 2.5
we can use bangbang control
''')
A=np.array([[1,0.1],[0,0.4]])
B=np.array([[0],[0.6]])
x0=np.array([[0],[0]])
ds=DynamicSystem(A,B,x0)
for _ in range(10):
    ds.step_(np.ones((1,1))*1, 0.1) # the 0.1 here is a noise
for _ in range(10):
    ds.step_(-1*np.ones((1,1))*1, 0.1) # the 0.1 here is a noise
plt.plot([d[0] for d in ds.x], label='distance')
plt.plot([d[1] for d in ds.x], label='velocity')
plt.legend()
plt.xlabel('time [dt]')
plt.ylabel('distance and velocity')




print(''' the system is the wolrd model.
the subjectg has a copy of that, as the internal model
the internal model integrates prediction and observation,
and estimate the state x in the world model.
''')


class Belief:
    '''
    the observer
    y = Cx + no 
        y, observation
        C, observation matrix,
        no, observation noise

    the regulator
    x_hat' = K* (y - C(Ax_hat+Bu) ) + x_hat
                            error       
    P' = (I - KC)* (APA.t + Q)

    where K = (APA.t + Q)@(C.t())@S-1
    C@((APA.t + Q))@(C.t()) + R

    x_hat, estimated x
    P, uncertainty of the x_hat
    ', next
    K, Kalman_gain
    Q, process noise
    R, obs noise
    '''
    def __init__(self, dynamic_system, C, horizon=100, PN=None,ON=None) -> None:
        self.system=dynamic_system
        self.C=C
        self.horizon = horizon
        self.PN,self.ON=[],[]
        if PN is None:
            PN=np.random.random(size=(horizon,1,1))
        if ON is None:
            ON=np.random.random(size=(horizon,1,1))
        for t in range(horizon): # convert to variance
            self.PN.append(PN[t] @ PN[t].T)
            self.ON.append(ON[t] @ ON[t].T)
        self.P = [np.zeros_like(dynamic_system.A)] #  P_0, inital uncertainty is zero
        self.S=[]
        self.Kf = []
        self.y=[self.C@dynamic_system.x[0]]
        self.x=[dynamic_system.x[0]]


    def observe(self, t=-1):
        return self.C@self.system.x[t]

    # the belief, linear quadratic estimator (Kalman filter) 
    def lqe(self,t=-1):
        C=self.C
        A=self.system.A
        self.Kf.append(self.P[t] @ C.T @ np.linalg.pinv(C @ self.P[t] @ C.T + self.ON[t]))
        self.S.append(self.P[t] - self.Kf[t] @ C @ self.P[t])
        if t < self.horizon - 1:
            self.P.append(A @ self.S[t] @ A.T + self.PN[t]) # this is P_t+1
    
    def step(self,u):
        y=self.observe()
        self.y.append(y)
        self.lqe() # get the current kalman gain
        k=self.Kf[-1]
        predicted_x=self.system.step(self.x[-1],u)
        err=(y-self.C@predicted_x)
        estimated_x=predicted_x+k@(err)
        self.x.append(estimated_x)


print(''' this is an example monkey model, monkey can predict his location and velocity pretty well.
''')


A=np.array([[1,0.1],[0,0.4]])
B=np.array([[0],[0.6]])
x0=np.array([[0],[0]])
world=DynamicSystem(A,B,x0)
C=np.array([[0,1]])
monkey=Belief(world,C)
for _ in range(monkey.horizon-1):
    process_noise=np.random.uniform(-1,1)
    control=np.random.normal(0,1)
    world.step_(np.ones((1,1))*control, process_noise) # the 0.1 here is a noise
    monkey.step(np.ones((1,1))*control)

plt.plot([d[0] for d in monkey.x], label='monkey distance')
[p[0,0]**0.5 for p in monkey.P]
monkey_ds=np.array([d[0,0] for d in monkey.x])
monkey_ds_uncertainty=np.array([p[0,0]**0.5 for p in monkey.P])
plt.fill_between(list(range(monkey.horizon)), monkey_ds-monkey_ds_uncertainty,monkey_ds+monkey_ds_uncertainty, alpha=0.4)
# plt.errorbar(list(range(monkey.horizon)), [d[0,0] for d in monkey.x], yerr=[p[0,0]**0.5 for p in monkey.P])
plt.plot([d[1] for d in monkey.x], label='monkey velocity')
monkey_vs=np.array([d[1,0] for d in monkey.x])
monkey_vs_uncertainty=np.array([p[1,1]**0.5 for p in monkey.P])
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
plt.plot([d[0] for d in world.x], label='actual distance')
plt.plot([d[1] for d in world.x], label='actual velocity')
plt.legend()
plt.xlabel('time [dt]')
plt.ylabel('distance and velocity')


'''

x'=Ax+Bu+ process noise
y=Cx+ observation noise

if noise is added:
observation noise+v = noisy observation = actual_v + noise ~N(0,sigma2)
process noise = 0
process noise + state = noisy state = noisy state velocity = control*control_gain + noise~N(0,sigma2)
in acc control:
process noise + state = noisy state = noisy state velocity = 
if we think noise is added to the state velocity per dt,
v' = v*a + b*u
noisy v' = a*v + b*u + noise



if noise is multiplicated:

yizhou: obs noisy v = (actual_v+noise)*scalar, where noise~N(0,siugma2)
akis: obs noisy v = 
        process noisy v = v*(1+noisy) where noise~some dynaimcs
        rewrite:
        v'= a*v +b*u 
        noise' = _a*noise + _b*new noise, where new noise ~N(0,sigma2)
        noisyv' = a*v +b*u + v*dynamic noise
        state 
        [d
        v
        noise]

        A [ 1, dt, 0
            0, a, 0
            0,, 0, _a]

         B 
         [ 0, b, 0]

        x' = Ax + Bu + _b*process noise(new noise~N(0,sigma2))


state x = [distance, velocity]


x' = Ax + Bu + noise
clean_v' = a*clean_v + (1-a)*u
noisy_v' = clean_v + clean_v*dynamic_noise
where dynamic_noise' = a_noise*dynamic_noise + (1-a_noise)*noise, and noise~N(0,sigma^2)



noisyv' = v' + v'*dynamic_noise
noisyv' = a*v + b*u + (a*v + b*u)*dynamic_noise

noisyv' = v + v*dynamic+_noise
noisyv' = v + v*(_a*prevnoise+b*newnoise)
        = v*(1+_a*prevnoise ) +v*b*newnoise


state 
[d
v
noise
noisy v
]

A [ 1, dt, 0 ,0
    0, a, 0 ,0 
    0,, 0, _a, 0
    0, 1, 0, ]

    B 
    [ 0, b, 0]

x' = Ax + Bu + _b*process noise(new noise~N(0,sigma2))






'''


print('''
if there is a state dependent noise (eg, n~N(0,sigma2)*v instead of n~N(0,sigma2)) (the multiplicative noise)
we can augumented the state to have a specific variable for observeraion directly

that is, 
state=[d, v, noisyv, noise]
and our observation matrix C, can be [0,0,1,0] (monkey is only allowed to observe the noisyv part)
the state transition can be written in this way:
d'  = d+dt*v
v' =a*v+b*u
noisyv' =  v*scalar + noise*scalar
noise' = ~N(0,sigma)

this can be futher simlifyied into:
state=[d, v, noise]
d'  = d+dt*v
v' =a*v+b*u
noise' = ~N(0,sigma)
and
C = [0,scalar,scalar] 
obs = scalar*v + scalar* noise

going back to the dynamic equaltion:
x'=Ax+bu+noise
the noise here is usually termed as 'process noise'.
but here, we use it for both conventional process noise AND our observation noise.
The noise is usually called Q matrix in lqg
here Q = [ [0, 0, 0  ] ,
            [0,  noise_var_process_v, 0]
            [0, 0,  noise_var_obs  ] ]
to make things easier to read, Q is a diagnal matrix where the each entry means the variance of the noise on state variables.
here, its [noisevar_d, noisevar_v, noisevar_noise]
there is no distance perturbations, so noisevar_d is 0
there could be velocity perturbations, so when modeling, this is a parameter, and this is the conventional process noise.
the noisevar_noise is the observation noise. we augumented it into the state for the multiplicative noise calculation.
specifically, noise_var_obs = sigma^2 * scalar^2
The Q is used when solving the optimal solution but it is not used here.

''')



print('''
x'=Ax+Bu+noise
d'= d+dt*v
v'= a*v + (bu)
noise'=0*d +0*v + 0*noise
''')
A=np.array([[1,0.1,0],[0,0.4,0],[0,0,0]])
print('''
x'=Ax+Bu+noise
d'= (d+dt*v) we cannot directly control distance
v'= (a*v) + bu
noise'=(0*d +0*v + 0*noise)
''')
B=np.array([[0],[0.6],[0]])
x0=np.array([[0],[0],[0]])
world=DynamicSystem(A,B,x0)

obs_noise_scalar=0.5 # noisyv = this_scalar*(actual_v + noise~N(0,sigma2))
C=np.array([[0,obs_noise_scalar,obs_noise_scalar]])
monkey=Belief(world,C)

for _ in range(monkey.horizon-1):
    process_noise=np.random.uniform(-1,1)*np.array([[0],[0],[1]])
    control=np.random.uniform(-1,1)
    world.step_(np.ones((1,1))*control, process_noise) # the 0.1 here is a noise
    monkey.step(np.ones((1,1))*control)

plt.plot([d[0] for d in monkey.x], label='monkey distance')
[p[0,0]**0.5 for p in monkey.P]
monkey_ds=np.array([d[0,0] for d in monkey.x])
monkey_ds_uncertainty=np.array([p[0,0]**0.5 for p in monkey.P])
plt.fill_between(list(range(monkey.horizon)), monkey_ds-monkey_ds_uncertainty,monkey_ds+monkey_ds_uncertainty, alpha=0.4)
# plt.errorbar(list(range(monkey.horizon)), [d[0,0] for d in monkey.x], yerr=[p[0,0]**0.5 for p in monkey.P])
plt.plot([d[1] for d in monkey.x], label='monkey velocity')
monkey_vs=np.array([d[1,0] for d in monkey.x])
monkey_vs_uncertainty=np.array([p[1,1]**0.5 for p in monkey.P])
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)

plt.plot([d[0] for d in world.x], label='actual distance')
plt.plot([d[1] for d in world.x], label='actual velocity')
plt.legend()
plt.xlabel('time [dt]')
plt.ylabel('distance and velocity')





print('''
update on 7.22 2022
some mis communications going on. 
met with akis and confirmed with xaq, the goal here is to finish the forward model for akis exp, to get ready for inverse.
the forward model has two parts, 1, world model. this is pretty easy.
2, the internal model. this is actually a copy of the world model, but with uncertainty.
because the dynamic multiplicative noise is a new thing, we are not sure the previous belief modeling can be applied here.
so my goal here is to try to model this.
right now, think observation as a point measuremnt.
''')



print('''
lets do the world model first
''')

class DynamicSystem:
    '''
    the system
    x' = Ax + Bu + np
        x, state [d, cleanv, actualv, dynamicnoise]
        ', next
        A, state transition matrix (vary with time)
        B, control matrix
        np, process noise
    '''
    def __init__(self, B, initalstate, a=0.4, anoise=0.2,dt=0.1) -> None:
        self.B=B
        self.dt=dt
        self.a=a
        self.anoise=anoise
        self.x=[initalstate]
        initA=self.computeA(a, anoise, initalstate[1,0])
        self.A=[initA]
    
    def step_(self,u, process_noise,t=-1):
        nextx=self.A[t]@self.x[t] + self.B@u + process_noise 
        self.A.append(self.computeA(self.a,self.anoise,self.x[t][1,0]))
        self.x.append(nextx)

    def step(self,x, u, A=None, B=None,t=-1): # prediction
        if A is None:
            A=self.A[t]
        if B is None:
            B=self.B
        return A@x + B@u


    def computeA(self,a, anoise, cleanv):
        # x = [d, cleanv, actualv, dynamicnoise]
        A=np.array([
            [1, 0, self.dt,0],
            [0, a, 0, 0],
            [0,1,0,cleanv],
            [0,0,0,anoise]
        ])
        return A

# x = [d, cleanv, actualv, dynamicnoise]
B=np.array([[0],[0.6], [0], [0]])
x0=np.array([[0],[0],[0],[0]])
ds=DynamicSystem(B,x0,a=0.4, anoise=0.4, dt=0.1)
for _ in range(20):
    noise=np.random.normal(0,0.1)
    ds.step_(np.ones((1,1))*1, noise) 
for _ in range(10):
    noise=np.random.normal(0,0.1)
    ds.step_(-1*np.ones((1,1))*1, noise) 

plt.plot([d[0] for d in ds.x], label='distance')
plt.plot([d[1] for d in ds.x], label='clean velocity')
plt.plot([d[2] for d in ds.x], label='actual noisy velocity')
plt.plot([d[3] for d in ds.x], label='dynamic noise')
plt.legend()
plt.xlabel('time [dt]')
plt.ylabel('distance and velocity')
print('''
so akis is right, this kind of process noise set up offers a smooth perturbvation. espaeically the noise time constant is large (changes slowly)
whlie we shoudl be carefule about the noise tau. if the tau is too large, then this could filter out any gaussian noise, and the noise is gone. 
question, should the noise amplitude pair with the noise tau? im sure theres some math relationship we can solve
''')


class Belief:

    def __init__(self, dynamic_system, C, horizon=200, q=0.1,r=0.1,PN=None,ON=None) -> None:
        self.system=dynamic_system
        self.C=C
        self.horizon = horizon
        self.PN,self.ON=[],[]
        self.P = [np.zeros_like(dynamic_system.A[0])]
        # +np.diag(np.diag(np.ones_like(dynamic_system.A[0])*1e-8))] 
        self.S=[np.zeros_like(dynamic_system.A[0])]
        self.Kf = []
        self.y=[self.C@dynamic_system.x[0]]
        self.x=[dynamic_system.x[0]]
        self.Q=np.zeros_like(dynamic_system.A[0])
        self.Q[-1,-1]=q**2
        self.R=r**2
        self.r=r


    def observe(self, t=-1):
        noise=np.random.normal(0,1)*self.r
        return self.C@self.system.x[t]+noise

    # the belief, linear quadratic estimator (Kalman filter) 
    def lqe(self,t=-1):
        C=self.C
        A=self.system.A[t]
        self.Kf.append(self.P[t] @ C.T @ np.linalg.pinv(C @ self.P[t] @ C.T + self.R))
        self.S.append(self.P[t] - self.Kf[t] @ C @ self.P[t])
        if t < self.horizon - 1:
            self.P.append(A @ self.S[t] @ A.T +self.Q) # this is P_t+1
    
    def step(self,u):
        y=self.observe()
        self.y.append(y)
        self.lqe() # get the current kalman gain
        k=self.Kf[-1]
        predicted_x=self.system.step(self.x[-1],u)
        err=(y-self.C@predicted_x)
        estimated_x=predicted_x+k@(err)
        self.x.append(estimated_x)

print(''' this is an example monkey model, monkey can predict his location and velocity pretty well.
''')

def geta(tau,dt=0.1):
    a = np.exp(-dt/tau)
    return a


# x = [d, cleanv, actualv, dynamicnoise]
B=np.array([[0],[0.6], [0], [0]])
x0=np.array([[0],[0],[0],[0]])
world=DynamicSystem(B,x0,a=0.4, anoise=geta(0.5), dt=0.01)

C=np.array([[0,0,1,0]])
processnoiselevel=0.3
obsnoiselevel=0.75
monkey=Belief(world,C,q=processnoiselevel, r=obsnoiselevel)
monkey.processnoiselevel=processnoiselevel

for _ in range(monkey.horizon-1):
    process_noise=np.zeros_like(world.x[-1])
    process_noise[3,0]=np.random.normal(0,1)*processnoiselevel
    control=np.random.normal(0.3,0.5)
    world.step_(np.ones((1,1))*control, process_noise) 
    monkey.step(np.ones((1,1))*control)


plt.plot([d[0] for d in monkey.x], label='monkey distance')
monkey_vs=np.array([d[0,0] for d in monkey.x])
monkey_vs_uncertainty=np.array([p[0,0]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
plt.plot([d[0] for d in world.x], label='actual distance')
plt.xlabel('time [dt]')
plt.ylabel('distance')
ax=plt.gca()
quickleg(ax,bbox_to_anchor=(0,0))
quickspine(ax)


plt.plot([d[3] for d in monkey.x], label='monkey noise')
monkey_vs=np.array([d[3,0] for d in monkey.x])
monkey_vs_uncertainty=np.array([p[3,3]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
plt.plot([d[3] for d in world.x], label='actual noise')
plt.xlabel('time [dt]')
plt.ylabel('distance and velocity')
ax=plt.gca()
quickleg(ax,bbox_to_anchor=(0,0))
quickspine(ax)



monkey_noises_uncertainty=np.array([p[3,3]**0.5 for p in monkey.S])
plt.plot(monkey_noises_uncertainty,label='noise uncertainty')
monkey_vs_uncertainty=np.array([p[2,2]**0.5 for p in monkey.S])
plt.plot(monkey_vs_uncertainty,label='velocity uncertainty')
monkey_ds_uncertainty=np.array([p[0,0]**0.5 for p in monkey.S])
plt.plot(monkey_ds_uncertainty,label='distance uncertainty')
plt.plot([d[2] for d in monkey.x], label='monkey velocity',alpha=0.3)
plt.xlabel('time [dt]')
plt.ylabel('uncertainty')
ax=plt.gca()
quickleg(ax,bbox_to_anchor=(0,0))
quickspine(ax)



plt.plot([d[0] for d in monkey.x], label='monkey distance')
monkey_vs=np.array([d[0,0] for d in monkey.x])
monkey_vs_uncertainty=np.array([p[0,0]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
plt.plot([d[0] for d in world.x], label='actual distance')
plt.xlabel('time [dt]')
plt.ylabel('distance')
ax=plt.gca()
quickleg(ax,bbox_to_anchor=(0,0))
quickspine(ax)



plt.plot([d[1] for d in monkey.x], label='monkey clean velocity')
monkey_vs=np.array([d[1,0] for d in monkey.x])
monkey_vs_uncertainty=np.array([p[1,1]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
plt.plot([d[1] for d in world.x], label='clean velocity')
plt.xlabel('time [dt]')
plt.ylabel('velocity')
ax=plt.gca()
quickleg(ax,bbox_to_anchor=(0,0))
quickspine(ax)



plt.plot([d[2] for d in monkey.x], label='monkey velocity')
monkey_vs=np.array([d[2,0] for d in monkey.x])
monkey_vs_uncertainty=np.array([p[2,2]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
plt.plot([d[2] for d in world.x], label='actual velocity')
plt.xlabel('time [dt]')
plt.ylabel('actual velocity')
ax=plt.gca()
quickleg(ax,bbox_to_anchor=(0,0))
quickspine(ax)



plt.plot([d[3] for d in monkey.x], label='monkey noise')
monkey_vs=np.array([d[3,0] for d in monkey.x])
monkey_vs_uncertainty=np.array([p[3,3]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
plt.plot([d[3] for d in world.x], label='actual noise')
plt.xlabel('time [dt]')
plt.ylabel('distance and velocity')
ax=plt.gca()
quickleg(ax,bbox_to_anchor=(0,0))
quickspine(ax)


print('''
the dynamic noise is filtered
in the long run, the mean of the dynamic noise is 0
so, if the task is long enough, it could be that the optimal strategy is fully trust the control.
use control to predict clean v, and use clean v to calcualte distance.
lets test it and compare
''')

# x = [d, cleanv, actualv, dynamicnoise]
B=np.array([[0],[0.6], [0], [0]])
x0=np.array([[0],[0],[0],[0]])
world=DynamicSystem(B,x0,a=0.4, anoise=0.8, dt=0.1)

C=np.array([[0,0,1,0]])
processnoiselevel=0.3
obsnoiselevel=0.75
monkey=Belief(world,C,q=processnoiselevel, r=obsnoiselevel)
monkey.processnoiselevel=processnoiselevel

for _ in range(monkey.horizon-1):
    process_noise=np.zeros_like(world.x[-1])
    process_noise[3,0]=np.random.normal(0,1)*processnoiselevel
    control=np.random.normal(0.3,0.5)
    world.step_(np.ones((1,1))*control, process_noise) 
    monkey.step(np.ones((1,1))*control)

plt.plot([d[0] for d in monkey.x], label='monkey distance')
monkey_vs=np.array([d[0,0] for d in monkey.x])
monkey_vs_uncertainty=np.array([p[0,0]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
plt.plot([d[0] for d in world.x], label='actual distance')
plt.xlabel('time [dt]')
plt.ylabel('distance')

plt.plot(np.cumsum([d[1] for d in monkey.x])*monkey.system.dt
, label='distance predicted from monkey clean velocity')

ax=plt.gca()
quickleg(ax,bbox_to_anchor=(0,0))
quickspine(ax)


print('''
yes, fully trust the control is a good way.
but is this also true for different noise time constant?
i remember akis said the noise changes much slower than the control acc.
lets try
''')

for anoise in (1-np.power(1e-3,np.linspace(0,1,10))):
    # x = [d, cleanv, actualv, dynamicnoise]
    B=np.array([[0],[0.6], [0], [0]])
    x0=np.array([[0],[0],[0],[0]])
    world=DynamicSystem(B,x0,a=0.4, anoise=anoise, dt=0.1)

    C=np.array([[0,0,1,0]])
    processnoiselevel=0.3
    obsnoiselevel=0.75
    monkey=Belief(world,C,q=processnoiselevel, r=obsnoiselevel)
    monkey.processnoiselevel=processnoiselevel

    for _ in range(monkey.horizon-1):
        process_noise=np.zeros_like(world.x[-1])
        process_noise[3,0]=np.random.normal(0,1)*processnoiselevel
        control=np.random.normal(0.3,0.5)
        world.step_(np.ones((1,1))*control, process_noise) 
        monkey.step(np.ones((1,1))*control)

    plt.plot([d[0] for d in monkey.x], label='monkey distance')
    monkey_vs=np.array([d[0,0] for d in monkey.x])
    monkey_vs_uncertainty=np.array([p[0,0]**0.5 for p in monkey.S])
    plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
    plt.plot([d[0] for d in world.x], label='actual distance')
    plt.xlabel('time [dt]')
    plt.ylabel('distance')

    plt.plot(np.cumsum([d[1] for d in monkey.x])*monkey.system.dt
    , label='distance predicted from monkey clean velocity')

    ax=plt.gca()
    quickleg(ax,bbox_to_anchor=(0,0))
    quickspine(ax)
    plt.title(anoise)
    plt.show()



''' filtering the noise twice'''
# filtering once
dynamicnoise_next = dynamicnoise*a + newnoise*(1-a)


# filer twice
dynamicnoise_next = (dynamicnoise*a + newnoise*(1-a))*(1-a) + dynamicnoise*a 


class DynamicSystem:
    '''
    the system
    x' = Ax + Bu + np
        x, state [d, cleanv, actualv, dynamicnoise]
        ', next
        A, state transition matrix (vary with time)
        B, control matrix
        np, process noise
    '''
    def __init__(self, B, initalstate, a=0.4, anoise=0.2,dt=0.1) -> None:
        self.B=B
        self.dt=dt
        self.a=a
        self.anoise=anoise
        self.x=[initalstate]
        initA=self.computeA(a, anoise, initalstate[1,0])
        self.A=[initA]
    
    def step_(self,u, process_noise,t=-1):
        nextx=self.A[t]@self.x[t] + self.B@u + process_noise 
        self.A.append(self.computeA(self.a,self.anoise,self.x[t][1,0]))
        self.x.append(nextx)

    def step(self,x, u, A=None, B=None,t=-1): # prediction
        if A is None:
            A=self.A[t]
        if B is None:
            B=self.B
        return A@x + B@u


    def computeA(self,a, anoise, cleanv):
        # x = [d, cleanv, actualv, dynamicnoise]
        A=np.array([
            [1, 0, self.dt,0],
            [0, a, 0, 0],
            [0,1,0,cleanv],
            [0,0,0,2*anoise-anoise**2]
        ])
        return A


class Belief:

    def __init__(self, dynamic_system, C, horizon=200, q=0.1,r=0.1) -> None:
        self.system=dynamic_system
        self.C=C
        self.horizon = horizon
        self.PN,self.ON=[],[]
        self.P = [np.zeros_like(dynamic_system.A[0])]
        # +np.diag(np.diag(np.ones_like(dynamic_system.A[0])*1e-8))] 
        self.S=[np.zeros_like(dynamic_system.A[0])]
        self.Kf = []
        self.y=[self.C@dynamic_system.x[0]]
        self.x=[dynamic_system.x[0]]
        self.Q=np.zeros_like(dynamic_system.A[0])
        self.Q[-1,-1]=q**2
        self.R=r**2
        self.r=r


    def observe(self, t=-1):
        noise=np.random.normal(0,1)*self.r
        return self.C@self.system.x[t]+noise

    # the belief, linear quadratic estimator (Kalman filter) 
    def lqe(self,t=-1):
        C=self.C
        A=self.system.A[t]
        self.Kf.append(self.P[t] @ C.T @ np.linalg.pinv(C @ self.P[t] @ C.T + self.R))
        self.S.append(self.P[t] - self.Kf[t] @ C @ self.P[t])
        if t < self.horizon - 1:
            self.P.append(A @ self.S[t] @ A.T +self.Q) # this is P_t+1
    
    def step(self,u):
        y=self.observe()
        self.y.append(y)
        self.lqe() # get the current kalman gain
        k=self.Kf[-1]
        predicted_x=self.system.step(self.x[-1],u)
        err=(y-self.C@predicted_x)
        estimated_x=predicted_x+k@(err)
        self.x.append(estimated_x)


# x = [d, cleanv, actualv, dynamicnoise]
noisetau=0.25
B=np.array([[0],[0.6], [0], [0]])
x0=np.array([[0],[0],[0],[0]])
world=DynamicSystem(B,x0,a=0.4, anoise=geta(noisetau), dt=0.01)

C=np.array([[0,0,1,0]])
processnoiselevel=11
obsnoiselevel=0.75
monkey=Belief(world,C,q=processnoiselevel*getprocessnoisescalar(noisetau), r=obsnoiselevel)
# monkey.processnoiselevel=processnoiselevel*getprocessnoisescalar(noisetau)

for _ in range(monkey.horizon-1):
    process_noise=np.zeros_like(world.x[-1])
    process_noise[3,0]=np.random.normal(0,1)*processnoiselevel*getprocessnoisescalar(noisetau)
    control=np.random.normal(0.3,0.5)
    world.step_(np.ones((1,1))*control, process_noise) 
    monkey.step(np.ones((1,1))*control)


plt.plot([d[3] for d in monkey.x], label='monkey noise')
monkey_vs=np.array([d[3,0] for d in monkey.x])
monkey_vs_uncertainty=np.array([p[3,3]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
plt.plot([d[3] for d in world.x], label='actual noise')
plt.xlabel('time [dt]')
plt.ylabel('distance and velocity')
ax=plt.gca()
quickleg(ax,bbox_to_anchor=(0,0))
quickspine(ax)



plt.plot([d[0] for d in monkey.x], label='monkey distance')
monkey_vs=np.array([d[0,0] for d in monkey.x])
monkey_vs_uncertainty=np.array([p[0,0]**0.5 for p in monkey.S])
plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
plt.plot([d[0] for d in world.x], label='actual distance')
plt.xlabel('time [dt]')
plt.ylabel('distance')
ax=plt.gca()
quickleg(ax,bbox_to_anchor=(0,0))
quickspine(ax)




ts=np.linspace(0,2,200)
initnoise=0
dnoise=[initnoise]
ddnoise=[initnoise]
a=geta(0.25)
for _ in ts:
    newnoise=np.random.normal(0,1)

    # filtering twice, once per step
    tmp=dnoise[-1]*a+(1-a)*newnoise
    filerednoise=(1-a)*tmp + dnoise[-1]*a
    dnoise.append(filerednoise)

    # filtering twice, all in once
    filterreddirectlynoise=(2*a-a**2)*ddnoise[-1] + newnoise*(a**2-a*2+1)
    ddnoise.append(filterreddirectlynoise)

plt.plot(dnoise)

plt.plot(ddnoise)






print('7.31')
print('''
now try to add obs noise.

think about the problem in general
regardless of the type of obs noise, there are 2 optic flows
one is actual v, one is actual v with obs noise.

there are two optic flows
we dont know which one is what
they have different dynamics, how should we integrate them, ideally optimally or near optimally?

there are 3 things. prediction, o1, o2.
if o1 is true:
    we should integrate o1 and p
else, we shoudl integrate o2 and p


lets assume for now there are some way to tell if one optic flow is true, with some probability
(we dont want this to happend, but assume it for now)
then, the optimal integration is:
    res=prob1true*(o1 int p) + (1-prob1true)*(o2 int p)
the problem becomes if we can find a prob1true, the probability of o1 being true optic flow
if not, then we can think about if we should just integrate p and o1 and o2 all together.


Q: if we can find the probability of o1 being true optic flow
there are cases that we cannot have any infomation about this prob
eg, when no control, v+process noise <0 (moving backwards) and v+process noise+obs noise >0 (forward), then theres no way to tell them apart
however, because the observation is always added on top of process noise + true v, then the optic flow with less variation is the one we should trust.
so, the prob will be increasing during the trial, starting from 1/2 to 1-.
so far we think there is a prob. and potentially solvable in simple cases.

think about when the noises are dynamic.
if process noise tau > obs noise tau, then the previous conclusion holds. 
because obs noise changes faster than process noise.
if the diff in tau is large enough, then we will be able to see the large trend as process noise, and the small jitter as obs noise.

what if the obs tau > process tau?
we still trust the smaller variance optic flow


v+pn
v+pn+on

''')


def getnext(prev,u=0, tau=0.25, dt=0.01):
    a=geta(tau,dt=dt)
    cur=a*prev+u*(1-a)
    return cur

vs=[0]
tau=0.1
ustd=0.1
for i in range(nts):
    u=np.random.normal(0,1)*ustd+0.5
    vs.append(getnext(vs[-1],u, tau))

print('''
when ptau > otau, we shouldl trust the optic flow with larger variance''')
nts=100
pnoise=[0]
ptau=0.25
pstd=0.3
for i in range(nts):
    u=np.random.normal(0,1)*pstd
    pnoise.append(getnext(pnoise[-1],u, ptau))

onoise=[0]
otau=0.1
ostd=0.3
for i in range(nts):
    u=np.random.normal(0,1)*ostd
    onoise.append(getnext(onoise[-1],u, otau))

pnoise,onoise=np.array(pnoise),np.array(onoise)
plt.plot(pnoise)
plt.plot(pnoise+onoise)



print('''
when ptau < otau, we shouldl trust the optic flow with larger variance''')
nts=100
pnoise=[0]
ptau=0.1
pstd=0.3
for i in range(nts):
    u=np.random.normal(0,1)*pstd
    pnoise.append(getnext(pnoise[-1],u, ptau))

onoise=[0]
otau=0.3
ostd=0.3
for i in range(nts):
    u=np.random.normal(0,1)*ostd
    onoise.append(getnext(onoise[-1],u, otau))

pnoise,onoise=np.array(pnoise),np.array(onoise)
plt.plot(pnoise)
plt.plot(pnoise+onoise)




print('''
so far, we've seen that we shoudl always trust the smaller variance optic flow for the additive case.
we confirmed that this is true for multipliticave case too.
''')


vs=[0]
tau=0.1
ustd=0.1
for i in range(nts):
    u=np.random.normal(0,1)*ustd+0.5
    vs.append(getnext(vs[-1],u, tau))

nts=100
pnoise=[0]
ptau=0.25
pstd=0.3
for i in range(nts):
    u=np.random.normal(0,1)*pstd
    pnoise.append(getnext(pnoise[-1],u, ptau))

onoise=[0]
otau=0.1
ostd=0.3
for i in range(nts):
    u=np.random.normal(0,1)*ostd
    onoise.append(getnext(onoise[-1],u, otau))

vs,pnoise,onoise=np.array(vs),np.array(pnoise),np.array(onoise)
plt.plot(pnoise*vs)
plt.plot((pnoise+onoise)*vs)
plt.title('ptau>otau')
plt.show()


nts=100
pnoise=[0]
ptau=0.4
pstd=6
for i in range(nts):
    u=np.random.normal(0,1)*pstd
    pnoise.append(getnext(pnoise[-1],u, ptau))

onoise=[0]
otau=0.6
ostd=5
for i in range(nts):
    u=np.random.normal(0,1)*ostd
    onoise.append(getnext(onoise[-1],u, otau))

vs,pnoise,onoise=np.array(vs),np.array(pnoise),np.array(onoise)
plt.plot(pnoise*vs)
plt.plot((pnoise+onoise)*vs)
plt.title('ptau<otau')
plt.show()


print('''
we shoudl always trust the smaller variance optic flow.
but this is hard, there is no information at the begining to know the variance
only as time goes we can start to know the variance.
so, the optimal strategy could be, initialize prob as 1/2, meaning that we take the average of the two optic flow at the very begining of the trial
as time goes, based on the prob, we trust one optic flow more.
this could be a special function. 
''')


print('''
how to estimate the variance?
known the variance is important for this approach.
we should assume the animal only has limited memory, and he estimate the variance from this buffer.
after we have the variance, we will again have some filter function to measure the confidence of the variance.
overall this is doable, but could be complicated.
since we are not target for true optimal here, we decide to hold on to this approach.
''')




print('''
another branch
i have not played the task, but xaq said he cannot distinguish the two optic flow.
although theoritically it is optimal to trust the larger variance one, it seems not possible to do so when doing the task. 
i think it would be approate to say, it is so hard to tell them apart that we can safely assume they are not distinguishable.
so, we could just treat there are two optic flow that we need to integrate.

''')




print('''
to be more specific, we are not integrate 3 things at once. 
instead, we first integrate the 2 optic flow, then integrate the prediciton
when integrating the 2 optic flow, the question we ask is:
given cleanv and unlabeled o1 o2, solve for processv.

think in simple gassiuan addictive noise first.
given x, unlabeld x+p, x+p+o with uncertaintys, return x+p. where p and o are gassian.
because p and o are both centered on 0, so there is no way for us to know which is which.
instead, we do baysien integration:
ov = s1/s * o2 + s2/s * o1, where s is total variance (uncertainty).

when there is dynamic, things could be complicated.
the brain could running two computations at the same time, resulting in a gaussian mixture multimodel belief.
eg, treat o1 as vp, vs treat o2 as vp, then combine the two as our belief.
but for now assume the two beliefs as no communication.
the monkey track the optic flow and do not allow to regret and swich.


ok, we just want a simple model for now. 
ideally its optimal, but we can do apporximation.
just get a model thats working and we can see if we need further improvement.
''')




class DynamicSystem:

    def __init__(self, B, initalstate, a=0.4, pa=0.2, oa=0.2,dt=0.01) -> None:
        self.B=B
        self.dt=dt
        self.a, self.pa,self.oa=a,pa,oa
        self.x=[initalstate]
        initA=self.computeA(initalstate[1,0],initalstate[2,0])
        self.A=[initA]
    
    def step_(self,u, newnoise, t=-1):
        nextx=self.A[t]@self.x[t] + self.B@u + newnoise 
        self.A.append(self.computeA(self.x[t][1,0],self.x[t][2,0]))
        self.x.append(nextx)

    def step(self,x, u, A=None, B=None,t=-1): # prediction
        if A is None:
            A=self.A[t]
        if B is None:
            B=self.B
        return A@x + B@u

    def computeA(self, v, pv):
        # x = [d, v, pv, pn, ov, on]
        a,pa,oa=self.a,self.pa,self.oa
        A=np.array([
            [1, 0, self.dt,0,0,0],
            [0, a, 0, 0,0,0],
            [0,1,0,v,0,0],
            [0,0,0,2*pa-pa**2,0,0],
            [0,0,1,0,0,pv],
            [0,0,0,0,0,2*oa-oa**2]
        ])
        return A

tau,ptau,otau=0.4,0.3,0.6
noiselevel=1
a,pa,oa=geta(tau),geta(ptau),geta(otau)
B=np.array([[0],[1-a],[0], [0],[0], [0]])
x0=np.array([[0],[0],[0],[0],[0], [0]])
world=DynamicSystem(B, x0,a=a,pa=pa,oa=oa)


for _ in range(99):
    newnoise=np.zeros_like(world.x[-1]).astype('float32')
    newnoise[3,0]=np.random.normal(0,1)*noiselevel*getprocessnoisescalar(ptau)
    newnoise[5,0]=np.random.normal(0,1)*noiselevel*getprocessnoisescalar(otau)
    # control=1
    control=np.random.normal(0.7,0.1)
    world.step_(np.ones((1,1))*control, newnoise) 

# xs=np.array(world.x)[:,:,0].T
# plt.plot(xs[0])

# plt.plot(xs[1]) # v
# plt.plot(xs[2]) # pv
# plt.plot(xs[4]) # ov

# plt.plot(xs[3]) # pn
# plt.plot(xs[5]) # on


class Belief:
    # x = [d, v, pv, pn, ov, on]
    def __init__(self, dynamic_system, C, horizon=200, qp=0.1, qo=0.1,r=0.1) -> None:
        self.system=dynamic_system
        self.C=C
        self.horizon = horizon
        self.PN,self.ON=[],[]
        self.P = [np.zeros_like(dynamic_system.A[0])]
        # +np.diag(np.diag(np.ones_like(dynamic_system.A[0])*1e-8))] 
        self.S=[np.zeros_like(dynamic_system.A[0])]
        self.Kf = []
        self.y=[self.C@dynamic_system.x[0]]
        self.x=[dynamic_system.x[0]]
        self.Q=np.zeros_like(dynamic_system.A[0])
        self.Q[3,3]=qp**2
        self.Q[5,5]=qo**2
        self.R=r**2
        self.r=r


    def observe(self, t=-1):
        noise=np.zeros_like(self.C@self.system.x[t])
        noise[0,0]=np.random.normal(0,1)*self.r[0,0]
        noise[1,0]=np.random.normal(0,1)*self.r[1,1]
        return self.C@self.system.x[t]+noise

    # the belief, linear quadratic estimator (Kalman filter) 
    def lqe(self,t=-1,C=None):
        C=self.C if C is None else C
        A=self.system.A[t]
        self.Kf.append(self.P[t] @ C.T @ np.linalg.pinv(C @ self.P[t] @ C.T + self.R))
        self.S.append(self.P[t] - self.Kf[t] @ C @ self.P[t])
        if t < self.horizon - 1:
            self.P.append(A @ self.S[t] @ A.T +self.Q) # this is P_t+1
    
    def step(self,u,C=None):
        y=self.observe()
        self.y.append(y)
        self.lqe(C=C) # get the current kalman gain
        k=self.Kf[-1]
        predicted_x=self.system.step(self.x[-1],u)
        err=(y-self.C@predicted_x)
        estimated_x=predicted_x+k@(err)
        self.x.append(estimated_x)


statenames=[
    'd',
    'v',
    'pv',
    'pn',
    'ov',
    'on',
]
tau,ptau,otau=1.,0.3,0.6
noiselevel=100
a,pa,oa=geta(tau),geta(ptau),geta(otau)
B=np.array([[0],[1-a],[0], [0],[0], [0]])
x0=np.array([[0],[0],[0],[0],[0], [0]])
world=DynamicSystem(B, x0,a=a,pa=pa,oa=oa)
C=np.array([[0,0,1,0,0,0],[0,0,0,0,1,0]])
obsgassiannoise=np.array([[0.3,0],[0,0.3]])

monkey=Belief(world,C,qp=noiselevel*getprocessnoisescalar(ptau), qo=noiselevel*getprocessnoisescalar(otau),r=obsgassiannoise)
for t in range(monkey.horizon-1):
    newnoise=np.zeros_like(world.x[-1]).astype('float32')
    newnoise[3,0]=np.random.normal(0,1)*noiselevel*getprocessnoisescalar(ptau)
    newnoise[5,0]=np.random.normal(0,1)*noiselevel*getprocessnoisescalar(otau)
    # control=0.3
    if t<150:
        control=1
    else: control=-1
    # control=np.random.normal(0.3,0.2)
    world.step_(np.ones((1,1))*control, newnoise) 
    monkey.step(np.ones((1,1))*control)



print('''
when the model knows which optic flow is which, it is very optimal
''')
for i in range(len(world.x[-1])):
    plt.plot([d[i] for d in monkey.x], label='belief')
    monkey_vs=np.array([d[i,0] for d in monkey.x])
    monkey_vs_uncertainty=np.array([p[i,i]**0.5 for p in monkey.S])
    plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
    plt.plot([d[i] for d in world.x], label='actual')
    plt.legend()
    plt.xlabel(statenames[i])
    plt.show()




print('''
when the model mistake the two optic flow, there will be a mismatch in state estimation.
''')

tau,ptau,otau=1,0.3,0.6
noiselevel=100
a,pa,oa=geta(tau),geta(ptau),geta(otau)
B=np.array([[0],[1-a],[0], [0],[0], [0]])
x0=np.array([[0],[0],[0],[0],[0], [0]])
assumedworld=DynamicSystem(B, x0,a=a,pa=oa,oa=pa) # assume wrong dynamics
world=DynamicSystem(B, x0,a=a,pa=pa,oa=oa)
C=np.array([[0,0,1,0,0,0],[0,0,0,0,1,0]])
obsgassiannoise=np.array([[0.3,0],[0,0.3]])
# monkey 1 with swapped optic flow
mistakenmodel=Belief(assumedworld,C,qp=noiselevel*getprocessnoisescalar(ptau), qo=noiselevel*getprocessnoisescalar(otau),r=obsgassiannoise)
# monkey 2 with true optic flow
correctmodel=Belief(world,C,qp=noiselevel*getprocessnoisescalar(ptau), qo=noiselevel*getprocessnoisescalar(otau),r=obsgassiannoise)
for t in range(monkey.horizon-1):
    newnoise, flippednoise=np.zeros_like(world.x[-1]).astype('float32'),np.zeros_like(world.x[-1]).astype('float32')
    thep, theo=np.random.normal(0,1),np.random.normal(0,1)
    newnoise[3,0]=thep*noiselevel*getprocessnoisescalar(ptau)
    newnoise[5,0]=theo*noiselevel*getprocessnoisescalar(otau)
    #
    flippednoise[3,0]=thep*noiselevel*getprocessnoisescalar(ptau)
    flippednoise[5,0]=theo*noiselevel*getprocessnoisescalar(otau)
    if t<150:
        control=1
    else: control=-1
    # control=0.3
    # control=np.random.normal(0.3,0.2)
    world.step_(np.ones((1,1))*control, newnoise) 
    assumedworld.step_(np.ones((1,1))*control, flippednoise) 
    mistakenmodel.step(np.ones((1,1))*control)
    correctmodel.step(np.ones((1,1))*control)

for i in range(len(world.x[-1])):
    plt.plot([d[i] for d in mistakenmodel.x], label='mistaken belief')
    monkey_vs=np.array([d[i,0] for d in mistakenmodel.x])
    monkey_vs_uncertainty=np.array([p[i,i]**0.5 for p in mistakenmodel.S])
    plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
    plt.plot([d[i] for d in correctmodel.x], label='belief')
    monkey_vs=np.array([d[i,0] for d in correctmodel.x])
    monkey_vs_uncertainty=np.array([p[i,i]**0.5 for p in correctmodel.S])
    plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)

    # plt.plot([d[i] for d in world.x], label='actual')
    # plt.plot([d[i] for d in assumedworld.x], label='assumed')
    plt.legend()
    plt.xlabel(statenames[i])
    plt.show()


plt.plot([x[3] for x in world.x])
plt.plot([x[3] for x in assumedworld.x])

plt.plot([x[5]*x[2] for x in world.x])
plt.plot([x[5]*x[2] for x in assumedworld.x])


print('''
when the model mistake the two optic flow, there will be a mismatch in state estimation.
assumed world:
    process tau = otau
    obs tau = ptau
monkey mistake the two noise include their dynamics (mis identify the source, but still predicting the dyanmic noise correct)
monkey will also get the 2 obs swapped.

''')

class World:
    def __init__(self, B, initalstate, a=0.4, pa=0.2, oa=0.2,dt=0.01) -> None:
        self.B=B
        self.dt=dt
        self.a, self.pa,self.oa=a,pa,oa
        self.x=[initalstate]
        initA=self.computeA(initalstate[1,0],initalstate[2,0])
        self.A=[initA]
    
    def step_(self,u, newnoise, t=-1):
        newnoise[3,0]=newnoise[3,0]*getprocessnoisescalar(self.pa)
        newnoise[5,0]=newnoise[5,0]*getprocessnoisescalar(self.oa)
        nextx=self.A[t]@self.x[t] + self.B@u + newnoise 
        self.A.append(self.computeA(self.x[t][1,0],self.x[t][2,0]))
        self.x.append(nextx)

    def step(self,x, u, A=None, B=None,t=-1): # prediction
        if A is None:
            A=self.A[t]
        if B is None:
            B=self.B
        return A@x + B@u

    def computeA(self, v, pv):
        # x = [d, v, pv, pn, ov, on]
        a,pa,oa=self.a,self.pa,self.oa
        A=np.array([
            [1, 0, self.dt,0,0,0],
            [0, a, 0, 0,0,0],
            [0,1,0,v,0,0],
            [0,0,0,2*pa-pa**2,0,0],
            [0,0,1,0,0,pv],
            [0,0,0,0,0,2*oa-oa**2]
        ])
        return A

    def observe(self,C, r, t=-1):
        noise=np.zeros_like(C@self.x[t])
        noise[0,0]=np.random.normal(0,1)*r[0,0]
        noise[1,0]=np.random.normal(0,1)*r[1,1]
        return C@self.x[t]+noise

class Belief:
    # x = [d, v, pv, pn, ov, on]
    def __init__(self, dynamic_system, C, horizon=200, qp=0.1, qo=0.1,r=0.1) -> None:
        self.system=dynamic_system
        self.C=C
        self.horizon = horizon
        self.PN,self.ON=[],[]
        self.P = [np.zeros_like(dynamic_system.A[0])]
        # +np.diag(np.diag(np.ones_like(dynamic_system.A[0])*1e-8))] 
        self.S=[np.zeros_like(dynamic_system.A[0])]
        self.Kf = []
        self.y=[self.C@dynamic_system.x[0]]
        self.x=[dynamic_system.x[0]]
        self.Q=np.zeros_like(dynamic_system.A[0])
        self.Q[3,3]=qp**2
        self.Q[5,5]=qo**2
        self.R=r**2
        self.r=r


    # the belief, linear quadratic estimator (Kalman filter) 
    def lqe(self,t=-1,C=None):
        C=self.C if C is None else C
        A=self.system.A[t]
        self.Kf.append(self.P[t] @ C.T @ np.linalg.pinv(C @ self.P[t] @ C.T + self.R))
        self.S.append(self.P[t] - self.Kf[t] @ C @ self.P[t])
        if t < self.horizon - 1:
            self.P.append(A @ self.S[t] @ A.T +self.Q) # this is P_t+1
    
    def step(self,u,y=None, C=None):
        y=self.system.observe(self.C, self.r) if y is None else y
        self.y.append(y)
        self.lqe(C=C) # get the current kalman gain
        k=self.Kf[-1]
        predicted_x=self.system.step(self.x[-1],u)
        err=(y-self.C@predicted_x)
        estimated_x=predicted_x+k@(err)
        self.x.append(estimated_x)

tau,ptau,otau=1,0.3,0.6
noiselevel=10
a,pa,oa=geta(tau),geta(ptau),geta(otau)
B=np.array([[0],[1-a],[0], [0],[0], [0]])
x0=np.array([[0],[0],[0],[0],[0], [0]])
C=np.array([[0,0,1,0,0,0],[0,0,0,0,1,0]])
obsgassiannoise=np.array([[0.3,0],[0,0.3]])

world=World(B, x0,a=a,pa=pa,oa=oa)
correctmodel=Belief(world,C,qp=noiselevel*getprocessnoisescalar(ptau), qo=noiselevel*getprocessnoisescalar(otau),r=obsgassiannoise)

assumedworld=World(B, x0,a=a,pa=oa,oa=pa)
mistakenmodel=Belief(assumedworld,C,qp=noiselevel*getprocessnoisescalar(otau), qo=noiselevel*getprocessnoisescalar(ptau),r=obsgassiannoise)

ys1,ys2=[],[]
for t in range(monkey.horizon-1):
    newnoise, flippednoise=np.zeros_like(world.x[-1]).astype('float32'),np.zeros_like(world.x[-1]).astype('float32')
    thep, theo=np.random.normal(0,1),np.random.normal(0,1)
    newnoise[3,0]=thep*noiselevel
    newnoise[5,0]=theo*noiselevel
    flippednoise[5,0]=thep*noiselevel
    flippednoise[3,0]=theo*noiselevel
    if t<150:
        control=1
    else: control=-1
    world.step_(np.ones((1,1))*control, newnoise) 
    assumedworld.step_(np.ones((1,1))*control, flippednoise) 
    y=world.observe(mistakenmodel.C, mistakenmodel.r)
    ys1.append(y)
    flippedy=np.array([[0,1],[1,0]])@y # flip the y
    ys2.append(flippedy)
    mistakenmodel.step(np.ones((1,1))*control,y=flippedy)
    correctmodel.step(np.ones((1,1))*control,y=y)

for i in range(len(world.x[-1])):
    plt.plot([d[i] for d in mistakenmodel.x], label='belief (mistaken 2 optic flows)')
    monkey_vs=np.array([d[i,0] for d in mistakenmodel.x])
    monkey_vs_uncertainty=np.array([p[i,i]**0.5 for p in mistakenmodel.S])
    plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
    plt.plot([d[i] for d in correctmodel.x], label='belief (correct)')
    monkey_vs=np.array([d[i,0] for d in correctmodel.x])
    monkey_vs_uncertainty=np.array([p[i,i]**0.5 for p in correctmodel.S])
    plt.fill_between(list(range(monkey.horizon)), monkey_vs-monkey_vs_uncertainty,monkey_vs+monkey_vs_uncertainty, alpha=0.4)
    plt.plot([d[i] for d in world.x], label='actual world', color='r')
    plt.plot([d[i] for d in assumedworld.x], label='assumed world', color='b')
    plt.legend()
    plt.xlabel(statenames[i])
    plt.show()



plt.plot(np.array(ys1)[:,:,0])
plt.plot(np.array(ys2)[:,:,0])


# validation. same pv
plt.plot([x[3] for x in world.x])
plt.plot([x[5] for x in assumedworld.x])
# validation. same ov
plt.plot([x[5]*x[2] for x in world.x])
plt.plot([x[3]*x[4] for x in assumedworld.x])
# x = [d, v, pv, pn, ov, on]

'''
pn, on

model 1:
gaussian mixture belief
option1, pn, on
option2, on, pn (swapped, mistaken)
each option takes 50% probabiliy

model 2:
weigthed integration of the option 1 and option 2
weight is from variance.
model a buffer, remember some optic flow velecoity data for some time.
use the data from buffer region, to estimate the variance.
have some function to calculate the weigth
if variance1 = variance2:
    we cant really tell which one is what.
but as time goes, we have enough evidence to estimate variance
till variance1>variance2, 
    weight2 = variance 1/total variance














'''
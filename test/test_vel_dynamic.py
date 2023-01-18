# test vel dynamic
import matplotlib.pyplot as plt
import torch

def a_(tau, dt=0.2):
    return torch.exp(-dt/tau)


def vm_(tau, x=400, T=8.5):
    vm=x/2/tau*(1/(torch.log(torch.cosh(T/2/tau))))
    if vm==0: # too much velocity control rather than acc control
        vm=x/T*torch.ones(1)
    return vm


def b_(tau, dt=0.2):
    return vm_(tau)*(1-a_(tau))

tau=5*torch.ones(1)
gain=1*torch.ones(1)
u=1*torch.ones(1)
ts=50
v=0
vs=[]
a=a_(tau)
vm=vm_(tau)
b=b_(tau)

for t in range(ts):
    vs.append(v)
    v=a*v+b*gain*u
 
plt.plot(vs)
vs[-1]


param=torch.rand(50,3)
X=[]
for i in range(50):
    a=param[i][0]
    b=param[i][1]
    v=0
    vs=[]
    for j in range(100):
        v=a*v+b
        vs.append(v)
    X.append(vs)

X=torch.tensor(X)
X=X[:,:40]
X.shape
u,s,v=torch.svd(X)
plt.plot(s[:5])
plt.plot(X.t())


from sklearn.cross_decomposition import CCA
cca=CCA(n_components=3)
cca.fit(param,X)
xc,yc=cca.transform(param, X)
plt.plot(xc)
plt.plot(yc)


class Agent():
    def __init__(self,a,b,q,r):
        self.reset(a,b,q,r)
    def reset(self, a,b,q,r):
        self.x=torch.ones(1)
        self.x_=torch.ones(1)
        self.P=torch.ones(1)*1e-8
        self.A=torch.ones(1)*a
        self.B=torch.ones(1)*b
        self.Q=torch.ones(1)*q**2
        self.R=torch.ones(1)*r**2
        self.data={
            'x':[],
            'x_':[],
            'P':[],
            'K':[],
            'S':[],
            'err':[],
            'x_p':[],
            'P_':[],
        }
    def dynamic(self,u,noise=None):
        if noise is None:
            self.x=self.A*self.x+self.B*u+torch.distributions.normal.Normal(0,1).sample()*self.R**0.5
        else:
            self.x=self.A*self.x+self.B*u+noise
        self.data['x'].append(self.x)
        self.kf(u)
    def kf(self,u):
        #predict
        x_=self.A*self.x_+self.B*u
        P_=self.A*self.P*self.A.t()+self.Q
        # prepare update
        err=self.x-x_
        S=P_+self.R
        K=P_*1/S
        # final update
        self.x_=x_+K*err
        self.P=(torch.ones(1)-K)*P_
        self.data['x_'].append(self.x_)
        self.data['x_p'].append(x_)
        self.data['P'].append(self.P)
        self.data['P_'].append(P_)
        self.data['S'].append(S)
        self.data['err'].append(err)
        self.data['K'].append(K)


tau=1
a=a_(tau)
b=b_(tau)

expert=Agent(a,b,0.3,0.15)
agent=Agent(a,b,0.1,0.3)
for i in range(10):
    expert.dynamic(torch.ones(1))
    agent.dynamic(torch.ones(1))

plt.plot(agent.data['x_'])
# plt.plot(agent.data['x'])
low=[x_-P for x_, P in zip(agent.data['x_'],agent.data['P'])]
high=[x_+P for x_, P in zip(agent.data['x_'],agent.data['P'])]
plt.fill_between(list(range(len(low))),low,high, alpha=0.5)
# plt.plot(expert.data['x'])
plt.plot(expert.data['x_'])
low=[x_-P for x_, P in zip(expert.data['x_'],expert.data['P'])]
high=[x_+P for x_, P in zip(expert.data['x_'],expert.data['P'])]
plt.fill_between(list(range(len(low))),low,high, alpha=0.5)


agent.data['P']
agent.data['P_']

plt.plot(agent.data['P'])
plt.plot(expert.data['P'])

plt.plot(agent.data['err'])
plt.plot(expert.data['err'])

plt.plot(agent.data['K'])
plt.plot(expert.data['K'])

loss=torch.sum([ax-ex for ax, ex in zip(agent.data['x_'],expert.data['x_'])])
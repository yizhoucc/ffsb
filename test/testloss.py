import torch
policy=8
class Agent():
    def __init__(self,a,b,q,r):
        self.reset(a,b,q,r)
    def reset(self, a,b,q,r):
        self.x=torch.ones(1)*0
        self.x_=torch.ones(1)*0
        self.P=torch.ones(1)*1e-8
        self.A=torch.ones(1)*a
        self.B=torch.ones(1)*b
        self.Q=torch.ones(1)*q**2
        self.R=torch.ones(1)*r**2
        self.T=30
        self.data_={
            'x':[],
            'x_':[],
            'P':[],
            'K':[],
            'S':[],
            'err':[],
            'x_p':[],
            'P_':[],
        }
        self.data=[]
    def dynamic(self,u,noise=None):
        if noise is None:
            noise=torch.distributions.normal.Normal(0,1).sample()*self.Q**0.5
        self.x=self.A*self.x+self.B*u+noise
        self.data_['x'].append(self.x)
        self.kf(u)
    def kf(self,u):
        # predict
        x_=self.A*self.x_+self.B*u
        P_=self.A*self.P*self.A.t()+self.Q
        # prepare update
        err=(self.x+torch.distributions.normal.Normal(0,1).sample()*self.R**0.5)-x_
        S=P_+self.R
        K=P_*1/S
        # final update
        self.x_=x_+K*err
        self.P=(torch.ones(1)-K)*P_
        self.data_['x_'].append(self.x_)
        self.data_['x_p'].append(x_)
        self.data_['P'].append(self.P)
        self.data_['P_'].append(P_)
        self.data_['S'].append(S)
        self.data_['err'].append(err)
        self.data_['K'].append(K)
    def get_data(self,policy):
        i=0
        while i<self.T:
            self.dynamic(torch.ones(1))
            self.data.append(torch.stack([self.x_,self.P]))
            i+=1
        self.data=torch.stack(self.data)[:,:,0]

class PV():
    def __init__(self,a,b,q,r):
        self.reset(a,b,q,r)
    def reset(self, a,b,q,r):
        self.x=torch.tensor([[0.],[0.]])
        self.x_=torch.tensor([[0.],[0.]])
        self.P=torch.tensor([[1.,0.],[0.,1.]])*1e-8
        self.A=torch.tensor([[1.,0.1],[0.,0]])+torch.tensor([[0.,0.],[0.,1.]])*a
        self.B=torch.tensor([[0.],[1.]])*b
        self.Q=torch.tensor([[0.,0.],[0.,1.]])*q**2
        self.R=torch.tensor([[1.]])*r**2
        self.T=30
        self.data_={
            'x':[],
            'x_':[],
            'P':[],
            'K':[],
            'S':[],
            'err':[],
            'x_p':[],
            'P_':[],
            'obs':[],
        }
        self.data=[]
    def dynamic(self,u,noise=None):
        if noise is None:
            noise=torch.tensor([[0.],[1.]])*torch.distributions.normal.Normal(0,1).sample()*self.Q[1,1]**0.5
        self.x=self.A@self.x+self.B@u+noise
        self.data_['x'].append(self.x)
        self.kf(u)
    def kf(self,u):
        # predict
        x_=self.A@self.x_+self.B@u
        P_=self.A@self.P@self.A.t()+self.Q
        # prepare update
        # print(self.x[1,0])
        # print(x_)
        o=(self.x[1,0]+torch.distributions.normal.Normal(0,1).sample()*self.R**0.5)
        err=o-x_[1,0]
        H=torch.tensor([[0.,1.]])
        # print(H)
        # print(self.R)
        # print(P_)
        S=H@P_@H.t()+self.R
        K=P_@H.t()@torch.inverse(S)
        # final update
        self.x_=x_+K@err
        self.P=(torch.tensor([[1.,0.],[0.,1.]])-K@H)@P_
        self.data_['x_'].append(self.x_)
        self.data_['x_p'].append(x_)
        self.data_['P'].append(self.P)
        self.data_['P_'].append(P_)
        self.data_['S'].append(S)
        self.data_['err'].append(err)
        self.data_['K'].append(K)
        self.data_['obs'].append(o)
    def get_data(self,policy):
        i=0
        while i<self.T:
            self.dynamic(torch.tensor([[1.]]))
            self.data.append(self.x_[1,0])
            i+=1
        self.data=torch.stack(self.data)
        self.data_['obs']=torch.stack(self.data_['obs'])[:,0,0]
        self.data_['P']=torch.stack(self.data_['P'])[:,1,1]
        self.data_['x']=torch.stack(self.data_['x'])[:,1,0]


# sys=PV(a[0],a[1],a[2],a[3])
# sys.get_data(policy)
# data1=[i[1].data for i in sys.data_['x']]
# plt.plot(data1)
# data2=[i[1].data for i in sys.data_['x_']]
# plt.plot(data2)

a =torch.tensor([0.5,0.5,0.1,0.1])
# a_=torch.tensor([0.6,0.4,.2,.2])
a_=torch.tensor([0.5,0.5,0.1,0.1])
a_=torch.nn.Parameter(a_)
opt=torch.optim.Adam([a_],lr=0.001)
while True:
    # sys1=Agent(a[0],a[1],a[2],a[3])
    # sys1.get_data(policy)
    fits=[]
    data=[]
    obs=[]
    var=[]
    truex=[]
    for i in range(200):
        sys1=PV(a[0],a[1],a[2],a[3])
        sys1.get_data(9)
        data.append(sys1.data)
        sys2=PV(a_[0],a_[1],a_[2],a_[3])
        sys2.get_data(9)
        fits.append(sys2.data)
        obs.append(sys2.data_['obs'])
        var.append(sys2.data_['P'])
        truex.append(sys2.data_['x'])
    data=torch.stack(data)
    fits=torch.stack(fits)
    obs=torch.stack(obs)
    var=torch.stack(var)
    truex=torch.stack(truex)
    # loss=abs(sys1.data-sys2.data)
    obsvar=a_.clone().detach()[3]**2
    transition=a_.clone().detach()[0]
    loss=mle(data, fits, obs,obsvar,transition,var,truex)
    # loss=torch.sum(loss)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(a_,loss.data)

# plt.plot(sys1.data_['err'])
# plt.plot(sys1.data_['x_'])

# class Sys():
#     def __init__(self,a,b):
#         self.T=10
#         self.data=[]
#         self.s=0*torch.ones(1)
#         self.a=a
#         self.b=b
#     def get_data(self,policy):
#         i=0
#         while i<self.T:
#             self.forward()
#             self.data.append((self.s))
#             i+=1
#         self.data=torch.stack(self.data)
#     def forward(self):
#         self.s=self.s*self.a+self.b+torch.distributions.normal.Normal(0,1).sample()*self.Q**0.5

class Policy(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(n_in, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, n_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

# policy=Policy(1,1)
# a_=torch.nn.Parameter(a_)
# opt=torch.optim.Adam([a_],lr=0.001)
# while True:
#     sys1=Sys(a,b)
#     sys1.get_data(policy)
#     sys2=Sys(a_[0],a_[1])
#     sys2.get_data(policy)
#     loss=abs(sys1.data-sys2.data)
#     loss=torch.sum(loss)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     print(a_)

# a =torch.tensor([0.5,0.5,0.1,0.1])
# a_=torch.tensor([0.5,0.5,.2,.2])
# a_=torch.nn.Parameter(a_)
# opt=torch.optim.Adam([a_],lr=0.001)
# while True:
#     # sys1=Agent(a[0],a[1],a[2],a[3])
#     # sys1.get_data(policy)
#     fits=[]
#     data=[]
#     for i in range(100):
#         sys1=Agent(a[0],a[1],a[2],a[3])
#         sys1.get_data(policy)
#         data.append(sys1.data)
#         sys2=Agent(a_[0],a_[1],a_[2],a_[3])
#         sys2.get_data(policy)
#         fits.append(sys2.data)
#     data=torch.stack(data)
#     fits=torch.stack(fits)
#     # loss=abs(sys1.data-sys2.data)
#     loss=lossfnwarp_(data, fits)
#     # loss=torch.sum(loss)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     print(a_,loss.data)
def mle(data,fits,obs,obsvar,transition,var,truex):
    index=0
    maxlen=len(data)
    totalloss=[]
    while index < maxlen:
        try:
            tdata=data[index,:]
            tfit=fits[index,:]
            tobs=obs[index,:]
            bvar=var[index,:]
            truext=truex[index,:]
            tloss=logll(tdata, tfit, tobs,obsvar,transition,bvar,truext)
            totalloss.append(tloss)
        except:
            index+=1
            continue
        index+=1
    totalloss=torch.stack(totalloss)
    totalloss=torch.sum(totalloss)
    return totalloss

def logll(tdata, tfit, tobs,obsvar,transition,bvar,truext):
    obsloss=1/obsvar/torch.sqrt(torch.tensor([6.28]))*(torch.exp(-0.5*(tobs-truext/obsvar)**2))
    # var=torch.var(tfit)*transition**2
    bloss=1/bvar/torch.sqrt(torch.tensor([6.28]))*(torch.exp(-0.5*((tdata-tfit)/bvar)**2))
    loss=-torch.sum(torch.log(bloss+1e-3))-torch.sum(torch.log(obsloss+1e-3))
    return loss

def lossfnwarp_(data,fits): # many to many
    index=0
    maxlen=len(data)
    totalloss=[]
    while index < maxlen:
        try:
            i=0
            tdata=data[index,:]
            tfit=fits[index,:]
            tloss=lossfn(tdata,tfit,torch.var(tfit))
            totalloss.append(tloss)
        except:
            index+=1
            continue
        index+=1
    totalloss=torch.stack(totalloss)
    totalloss=torch.sum(totalloss)
    return totalloss

def lossfnwarp(data,fit): # many to one
    index=0
    maxlen=len(data)
    totalloss=[]
    while index < maxlen:
        # print(index)
        tfit=[]
        try:
            i=0
            tdata=data[index][i]
            tfit=[t[index][i] for t in fit]
            tfit=torch.stack(tfit)
            tloss=lossfn(tdata,tfit,torch.var(tfit))
            totalloss.append(tloss)
            # i=1
            # tdata=data[index][i]
            # tfit=[t[index][i] for t in fit]
            # tfit=torch.stack(tfit)
            # tloss=torch.sum(torch.abs(tfit-tdata))
            # totalloss.append(tloss)
        except:
            index+=1
            continue
        index+=1
    totalloss=torch.stack(totalloss)
    totalloss=torch.sum(totalloss)
    return totalloss

def lossfn_(data, fit, var):
    loss=torch.abs(data-fit)
    return loss

def lossfn(data, fit, var):
    loss=1/var/torch.sqrt(torch.tensor([6.28]))*(torch.exp(-0.5*((data-fit)/var)**2))
    loss=-torch.sum(torch.log(loss+1e-3))
    return loss

# expert=[]
# for i in range(10):
#     sys1=Agent(a[0],a[1],a[2],a[3])
#     sys1.get_data(policy)
#     expert.append(sys1.data)
# agent=[]
# for i in range(10):
#     sys2=Agent(a_[0],a_[1],a_[2],a_[3])
#     sys2.get_data(policy)
#     agent.append(sys2.data)

# for i in expert:
#     plt.plot(i)
# for i in agent:
#     plt.plot(i.data)


# plt.plot(sys2.data_['x'])  
# plt.plot(sys2.data_['x_'])  
# # plt.plot(sys2.data_['err'])  
# plt.plot(sys1.data_['x'])  
# plt.plot(sys1.data_['x_'])  


expert=[]
for i in range(100):
    sys1=PV(a[0],a[1],a[2],a[3])
    sys1.get_data(policy)
    expert.append(sys1.data)
agent=[]
for i in range(100):
    sys2=PV(a_[0],a_[1],a_[2],a_[3])
    sys2.get_data(policy)
    agent.append(sys2.data)
for i in expert:
    plt.plot(i,'r',alpha=0.1)
for i in agent:
    plt.plot(i.data,'b',alpha=0.1)

# plt.hist([i[8].data for i in agent])

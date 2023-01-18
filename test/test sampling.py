import torch
import matplotlib.pyplot as plt
sample=100
length=30

param=0.5
data=[]
for i in range(sample):
    data.append(torch.distributions.normal.Normal(0,1).sample()*param)
plt.hist(data)

param=0
fit1=[]
for i in range(sample):
    fit1.append(torch.distributions.normal.Normal(0,1).sample()*param)
plt.hist(fit1)

param=0.5
fit2=[]
for i in range(sample):
    fit2.append(torch.distributions.normal.Normal(0,1).sample()*param)
plt.hist(fit2)

lossfit1=torch.sum(torch.abs(torch.tensor(data)-torch.tensor(fit1)))
lossfit2=torch.sum(torch.abs(torch.tensor(data)-torch.tensor(fit2)))

lossfit1=sum([abs(d-f) for d,f in zip(data,fit1)])


groundtruth=torch.tensor([0.,1.])
initial=torch.tensor([2.,2.])
initial=torch.nn.Parameter(initial)
opt=torch.optim.Adam([initial])
while True:
    alldata=[]
    for j in range(sample):
        data=[]
        for i in range(length):
            data.append(torch.distributions.normal.Normal(0,1).sample()*groundtruth[1]+groundtruth[0])
        data=torch.stack(data)
        alldata.append(data)
    # alldata=torch.stack(alldata)
    allfit1=[]
    for j in range(sample):
        fit1=[]
        for i in range(length):
            fit1.append(torch.distributions.normal.Normal(0,1).sample()*initial[1]+initial[0])
        fit1=torch.stack(fit1)
        allfit1.append(fit1)
    # allfit1=torch.stack(allfit1)
    # lossfit1=torch.sum(torch.abs(data-fit1))
    # lossfit1=lossfn((data),(fit1),torch.var((fit1)))
    totalloss=lossfnwarp(alldata,allfit1)
    opt.zero_grad()
    totalloss.backward()
    opt.step()
    print(initial, lossfit1)

initial.grad

class Policy(torch.nn.Module):
    def __init__(self, n_in, n_out):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(n_in, 80)
        self.fc3 = torch.nn.Linear(80, n_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        return x 

policy=Policy(30,30)




def lossfnwarp(data,fit):
    index=0
    maxlen=max([len(trial) for trial in data])
    totalloss=[]
    while index < maxlen:
        tdata=[]
        tfit=[]
        try:
            tdata=[t[index] for t in data]
            tfit=[t[index] for t in fit]
            tdata=torch.stack(tdata)
            tfit=torch.stack(tfit)
            tloss=lossfn(tdata,tfit,torch.var(tfit))
            totalloss.append(tloss)
        except:
            continue
        index+=1
    totalloss=torch.stack(totalloss)
    totalloss=torch.sum(totalloss)
    return totalloss
    



def lossfn(data, fit, var):
    loss=1/var*torch.exp(-0.5*((data-fit)/var)**2)
    loss=-torch.sum(torch.log(loss))
    return loss

loss=1/initial[1]*torch.exp(-0.5*((data-fit1)/initial[1])**2)

plt.hist(policy(data).data)
plt.hist(policy(fit1).data)
plt.hist(loss.data)

torch.var(policy(fit1))
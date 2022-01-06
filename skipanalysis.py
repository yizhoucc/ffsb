from scipy.ndimage.measurements import label
from IRCpaperplots import xy2pol
from os import stat
import pickle
import numpy as np
from matplotlib import collections
# from monkey_functions import datawash, monkey_data_downsampled
import matplotlib.pyplot as plt
from numpy.core.defchararray import encode
from numpy.core.fromnumeric import size
print('loading data')
with open("C:/Users/24455/Desktop/victor_normal_downsample",'rb') as f:
        df = pickle.load(f)

# validation check
# import pandas as pd
# with open("C:/Users/24455/Desktop/bruno_normal_trajectory.pkl",'rb') as f:
#         df = pd.read_pickle(f)
plt.scatter(df[df.category=='skip'].target_x,df[df.category=='skip'].target_y)

# value of all tasks        
from monkey_functions import *
factor=0.0025
tasks=[]
index=0
while index<df.shape[0]:
    trial=df.iloc[index]
    task=[convert_unit(trial.target_y,factor=factor),-convert_unit(trial.target_x,factor=factor)]
    tasks.append(task)
    index+=1



def getvalue(task):
    task=torch.tensor(task).float()
    env.reset(goal_position=task,theta=theta,phi=phi)
    value=critic(torch.cat([env.wrap_decision_info().flatten(),agent(env.wrap_decision_info(),).flatten()]))
    return value

with torch.no_grad():
    values=[]
    for task in tasks:
        values.append(getvalue(task))

distances,angles=[],[]
for task in tasks:
    d,a=xy2pol(task,rotation=False)
    distances.append(d)
    angles.append(abs(a))

duration=list(df.trial_dur)

trialtype=df.category.tolist()
uniques=set()
for a in trialtype:
    if a not in uniques:
        uniques.add(a)

nunique=len(uniques)

trialtypeencoding={}
defaultencoding=0
for a in uniques:
    trialtypeencoding[a]=defaultencoding
    defaultencoding+=1
vocab=[None]*nunique
for k,v in trialtypeencoding.items():
    vocab[v]=k
encodedtrialtype=[]
for a in trialtype:
    encodedtrialtype.append(trialtypeencoding[a])

dataset=np.zeros([len(encodedtrialtype),nunique])
for i in range(len(encodedtrialtype)):
    dataset[i,encodedtrialtype[i]]=1

rnndataset=[]
for i in range(len(encodedtrialtype)):
    onehot=torch.zeros(len(vocab))
    onehot[encodedtrialtype[i]]=1
    rnndataset.append(torch.cat([onehot,torch.ones(1)*distances[i],torch.ones(1)*angles[i],torch.ones(1)*duration[i]]))

# now we have the dataset of onehot encoding. n by 5 matrix
import torch
from torch import nn
import torch.nn.functional as F
T=len(df) # total data
# time=torch.arange(1,T+1,dtype=torch.float32)
tau=3 # use previous 3 steps to predict the next type
features=torch.zeros((T-tau,tau,nunique)) # [20291, 3, 5]
for i in range(T-tau):
    features[i,:,:]=torch.tensor(dataset[i:i+tau])
labels=dataset[tau:]



'''
first, we analyze the raw data

if we cannot model the transition easily, we can analyze pre and cur are correlated
in the data, there are many consectives. 
we are interested in, how many skips on avg before a normal trial. (2 to 1)

details, 
find 2->1 transition, expend from center to find left 2 and right 1
'''
vocab
encodedtrialtype

transitions=[]
pre=vocab['normal']
cur=vocab['skip']
for i in range(len(encodedtrialtype)-1):
    if encodedtrialtype[i]==pre and encodedtrialtype[i+1]==cur:
        transitions.append(i)
lefts,rights=[],[]
for t in transitions:
    # print(encodedtrialtype[t-10:t+10])
    # expend left
    l=0
    while len(encodedtrialtype)-l>=0 and encodedtrialtype[t-l]==pre:
        l+=1
    lefts.append(l)
    # expend right
    r=0
    while r<len(encodedtrialtype)-2 and encodedtrialtype[t+1+r]==cur:
        r+=1
    rights.append(r)    


plt.hist(lefts,bins=99)
plt.xlabel('num trials')
plt.ylabel('num occurance')
plt.title('num {} trial before a {} trial'.format(vocab[pre],vocab[cur]))

plt.hist(rights,bins=99)
plt.xlabel('num trials')
plt.ylabel('num occurance')
plt.title('num {} trial after a {} trial'.format(vocab[cur],vocab[pre]))


# check max consectives, validation
l,r=0,0
maxlen=0
while r<len(encodedtrialtype):
    if encodedtrialtype[l]==pre:
        r=l
        while encodedtrialtype[r]==pre:
            r+=1
            maxlen=max(maxlen,r-l)
    l+=1



# value of skipped trials and normal trials
skipvalues=[]
for i in range(len(encodedtrialtype)):
    if encodedtrialtype[i]==trialtypeencoding['skip']:
        skipvalues.append(values[i].item())

normalvalues=[]
for i in range(len(encodedtrialtype)):
    if encodedtrialtype[i]==trialtypeencoding['normal']:
        normalvalues.append(values[i].item())

min(normalvalues),max(normalvalues),np.mean(normalvalues)
min(skipvalues),max(skipvalues),np.mean(skipvalues)
min(transitionsvalues),max(transitionsvalues),np.mean(transitionsvalues)


# transition values
transitionsvalues=[]
for i in transitions:
    transitionsvalues.append(values[i+1].item())

plt.hist(skipvalues,alpha=0.5,density=True,label='skip')
plt.hist(normalvalues,alpha=0.5,density=True,label='normal')
# plt.hist(transitionsvalues,alpha=0.5,density=True,label='transition trials')
plt.title('skip values vs normal values')
plt.xlabel('target values')
plt.ylabel('probability')
plt.legend()



# distance and angles instead of value
skipds=[]
for i in range(len(encodedtrialtype)):
    if encodedtrialtype[i]==trialtypeencoding['skip']:
        skipds.append(distances[i].item())

normalds=[]
for i in range(len(encodedtrialtype)):
    if encodedtrialtype[i]==trialtypeencoding['normal']:
        normalds.append(distances[i].item())

plt.hist(skipds,alpha=0.5,density =True,label='skip')
plt.hist(normalds,alpha=0.5,density =True,label='normal')
plt.title('skip distances vs normal distances')
plt.xlabel('target distances')
plt.ylabel('probability')
plt.legend()


skipas=[]
for i in range(len(encodedtrialtype)):
    if encodedtrialtype[i]==trialtypeencoding['skip']:
        skipas.append(angles[i].item())

normalas=[]
for i in range(len(encodedtrialtype)):
    if encodedtrialtype[i]==trialtypeencoding['normal']:
        normalas.append(angles[i].item())

plt.hist(skipas,alpha=0.5,density =True,label='skip')
plt.hist(normalas,alpha=0.5,density =True,label='normal')
plt.title('skip angles vs normal angles')
plt.xlabel('target |angles|')
plt.ylabel('probability')
plt.legend()


plt.scatter(normalds,normalas,label='normal',alpha=0.1)
plt.scatter(skipds,skipas,label='skip',alpha=0.1)
plt.title('normal trials vs skip trials')
plt.xlabel('distance')
plt.ylabel('angles')
plt.legend()



skipxys=[]
for i in range(len(encodedtrialtype)):
    if encodedtrialtype[i]==trialtypeencoding['skip']:
        skipxys.append(tasks[i])

normalxys=[]
for i in range(len(encodedtrialtype)):
    if encodedtrialtype[i]==trialtypeencoding['normal']:
        normalxys.append(tasks[i])

skipxys,normalxys=np.array(skipxys),np.array(normalxys)
plt.scatter(normalxys[:,0],normalxys[:,1],label='normal',alpha=0.2,s=0.2)
plt.scatter(skipxys[:,0],skipxys[:,1],label='skip',alpha=0.8,s=0.9)


plt.hist(df[df.category=='normal'].trial_dur,bins=30)





'''
s - s - s - s - s - s
|   |   |   |   |   |
o   o   o   o   o   o

where s is the hidden state, and o is the obs of trial type that we can observe
without o (actually mk's action) affecting the next state, we have a transition matrix and emmission readout that saying lazy trials lead to lazy trials, and all others lead to normal trials.
this is expected, since lazy trials are likely to in group, while all other trials suggesting mk is paying attention and most of those trials are normal trials.


'''
import numpy as np
from hmmlearn import hmm

n_states=len(vocab)

observations = ['lazy', 'normal', 'skip', 'crazy', 'wrong_target']
n_observations = len(observations)
model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=20, tol=0.01)

model=hmm.MultinomialHMM(n_components=n_states, n_iter=100)
x = np.array(encodedtrialtype).reshape(-1,1)
x.dtype=int

model.fit(x)
model.monitor_.converged

np.set_printoptions(precision=2)

readout={}
for row in range((model.emissionprob_.shape[0])):
    readout[row]=np.argmax(model.emissionprob_[row])

l,r=0,10
print(x[l:r])
print([readout[x] for x in model.predict(x[l:r])])

plt.imshow(model.transmat_)
plt.imshow(model.emissionprob_)

plt.imshow(model.transmat_@model.emissionprob_)
plt.title('HMM transition readout')
plt.xlabel('input state')
plt.ylabel('ouput state')

'''
s - s - s - s - s - s
|/  |/  |/  |/  |/  |
a   a   a   a   a   a

if thiking the observed actions are actions and affecting the hidden state, we have the hmm like above.
use rnn to approximate the hmm


result, rnn cannot predict types well, seems it stucks at local minima and predicting all trials are normal.
this is kind of expected because normal trials are much more than other types.
specificly, states that tend to stay
'lazy', 
'normal', 
'skip', 
'crazy', 

'wrong_target' transition into normal

probabilty strongly favorate normal and crazy

'''

# d2l example
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size,num_steps=32,35
train_iter,vocab=d2l.load_data_time_machine(batch_size,num_steps)
num_hidden=64
rnn_layer=nn.RNN(len(vocab),num_hidden)

state=torch.zeros((1,batch_size,num_hidden))
state.shape

x=torch.rand(size=(num_steps,batch_size,len(vocab)))
x.shape

y,state_new=rnn_layer(x,state)
y.shape,state_new.shape

class RNNModel(nn.Module):
    """循环神经网络模型。"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.floaXt32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device))

device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)

num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)


prefix='time traveller'
num_preds=10
state = net.begin_state(batch_size=1, device=device)
outputs = [vocab[prefix[0]]]
get_input = lambda: d2l.reshape(
    d2l.tensor([outputs[-1]]), (1, 1))

for y in prefix[1:]:  # Warm-up period
    _, state = net(get_input(), state)
    outputs.append(vocab[y])

for _ in range(num_preds):  # Predict `num_preds` steps
    y, state = net(get_input(), state)
    outputs.append(int(y.argmax(axis=1).reshape(1)))

''.join([vocab.idx_to_token[i] for i in outputs])



# use the d2l rnn example on trial types
import torch
import math
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from torch.utils.data import DataLoader
from collections import deque
def grad_clipping(net, theta):
    """Clip the gradient.

    Defined in :numref:`sec_rnn_scratch`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return 1,1

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    loss = nn.CrossEntropyLoss()
    # animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
    #                         legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    # predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        _, _ = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        # if (epoch + 1) % 10 == 0:
            # print(predict('time traveller'))
            # animator.add(epoch + 1, [ppl])
    # print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    # print(predict('time traveller'))
    # print(predict('traveller'))


batch_size,num_steps=32,30
xwindow=deque()
ywindow=deque()
x,y=[],[]
for t in encodedtrialtype:
    if len(xwindow)<num_steps:
        xwindow.append(t)
    elif len(ywindow)<num_steps:
        ywindow.append(t)
    else:
        x.append(list(xwindow))
        y.append(list(ywindow))
        xwindow.popleft()
        xwindow.append(ywindow.popleft())
        ywindow.append(t)
newx,newy=[],[]
for i in range(len(x)):
    if x[i]!=y[i]:
        newx.append(x[i])
        newy.append(y[i])

data=[[torch.tensor(xx),torch.tensor(yy)] for xx,yy in zip(newx,newy)]
training_data=data[:(len(data)//batch_size-2)*(batch_size)]
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
train_iter=iter(train_dataloader)

# train_iter,vocab=d2l.load_data_time_machine(batch_size,num_steps)
num_hidden=256
rnn_layer=nn.RNN(7,num_hidden)

class RNNModel(nn.Module):
    """循环神经网络模型。"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        # step, batch, features
        X = F.one_hot(inputs[0].T.long(), self.vocab_size)
        X2=inputs[1]
        X=torch.cat([X,X2],-1)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device))
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=7)
net = net.to(device)
# d2l.predict_ch8('time traveller', 10, net, vocab, device)

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, device)


# predict 
start,end=2300,2400
num_preds=20
prefix=encodedtrialtype[start:end]


prefix=str(trialtypeencoding['skip'])*100
state = net.begin_state(batch_size=1, device=device)

for x in prefix:
    x=torch.tensor(int(x)).view(1,1)
    pred,state=net(x,state)
y=int(pred.argmax(dim=1))
output=[]
for _ in range(num_preds):
    y=torch.tensor(int(y)).view(1,1)
    pred,state=net(y, state)
    y=int(pred.argmax(dim=1))
    print(pred)
    output.append(int(pred.argmax(dim=1).reshape(1)))
print('out ',''.join(str(vocab[i])+' ' for i in  output))


print('true',''.join(str(vocab[i])+' ' for i in  encodedtrialtype[end:end+num_preds]))























# rnn, add trial other features

xwindow=deque()
ywindow=deque()
x,y=[],[]
for t in rnndataset:
    if len(xwindow)<num_steps:
        xwindow.append(t)
    elif len(ywindow)<1:
        ywindow.append(t)
    else:
        x.append(torch.stack(list(xwindow)))
        y.append(torch.stack(list(ywindow))[:,:4])
        xwindow.popleft()
        xwindow.append(ywindow.popleft())
        ywindow.append(t)
# newx,newy=[],[]
# for i in range(len(x)):
#     if x[i]!=y[i]:
#         newx.append(x[i])
#         newy.append(y[i])

data=[[xx,yy] for xx,yy in zip(x,y)]
training_data=data[:(len(data)//batch_size-2)*(batch_size)]
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
train_iter=iter(train_dataloader)






batch_size,num_steps=32,30
num_hidden=64
rnn_layer=nn.RNN(7,num_hidden)


state=torch.zeros((1,batch_size,num_hidden))
state.shape

x=torch.rand(size=(num_steps,batch_size,7))
x.shape

y,newstate=rnn_layer(x,state)
y.shape,newstate.shape


class RNNModel(nn.Module):
    """循环神经网络模型。"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        # step, batch, features
        Y, state = self.rnn(inputs, state)
        output = self.linear(Y)
        self.state=state
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device))

device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=4)
net = net.to(device)

out,newstate=net(x,state)
out.shape,newstate.shape
state=newstate


X,Y=next(train_iter)
X.shape
X.transpose_(0,1)
X.shape

y_hat, state = net(X, state)


num_epochs, lr = 500, 1
train_ch8(net, train_dataloader, vocab, lr, num_epochs, device)




net(x,state)




def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation
                for s in state:
                    s.detach_()
        X.transpose_(0,1)
        Y.transpose_(0,1)
        X, Y = X.to(device), Y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat[0], torch.argmax(Y.reshape((-1, Y.shape[-1])),dim=1))
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        # metric.add(l * d2l.size(y), d2l.size(y))
    return 1,1

def train_ch8(net, train_dataloader, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8).

    Defined in :numref:`sec_rnn_scratch`"""
    loss = nn.CrossEntropyLoss()
    # animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
    #                         legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    # predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        print(epoch)
        train_iter=iter(train_dataloader)
        _, _ = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        # if (epoch + 1) % 10 == 0:
            # print(predict('time traveller'))
            # animator.add(epoch + 1, [ppl])
    # print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    # print(predict('time traveller'))
    # print(predict('traveller'))




train_ch8(net, train_dataloader, vocab, lr, num_epochs, device)


# predict 

start,end=2300,2330
X=rnndataset[start:end]
state = net.begin_state(batch_size=1, device=device)

for i,x in enumerate(rnndataset):
    pred,state=net(x.view(1,1,7),state)
    truecat=vocab[encodedtrialtype[i+1]]
    cathat=vocab[torch.argmax(pred)]
    # if cathat!='normal':
    if cathat==truecat:
        print('predict:',vocab[torch.argmax(pred)])
        print('true   :',vocab[encodedtrialtype[i+1]])



print('predict:',vocab[torch.argmax(pred)])
print('true   :',vocab[encodedtrialtype[i+1]])


start,end=2300,2400
num_preds=20
prefix=encodedtrialtype[start:end]


state = net.begin_state(batch_size=1, device=device)

for x in prefix:
    x=torch.tensor(int(x)).view(1,1)
    pred,state=net(x,state)
y=int(pred.argmax(dim=1))
output=[]
for _ in range(num_preds):
    y=torch.tensor(int(y)).view(1,1)
    pred,state=net(y, state)
    y=int(pred.argmax(dim=1))
    print(pred)
    output.append(int(pred.argmax(dim=1).reshape(1)))
print('out ',''.join(str(vocab[i])+' ' for i in  output))


print('true',''.join(str(vocab[i])+' ' for i in  encodedtrialtype[end:end+num_preds]))


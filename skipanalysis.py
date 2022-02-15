from scipy.ndimage.measurements import label
from torch.nn.modules.linear import Linear
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


# get tasks
from monkey_functions import *
factor=0.0025
tasks=[]
index=0
while index<df.shape[0]:
    trial=df.iloc[index]
    task=[convert_unit(trial.target_y,factor=factor),-convert_unit(trial.target_x,factor=factor)]
    tasks.append(task)
    index+=1


# validation check
# import pandas as pd
# with open("C:/Users/24455/Desktop/bruno_normal_trajectory.pkl",'rb') as f:
#         df = pd.read_pickle(f)
plt.scatter(df[df.category=='skip'].target_x,df[df.category=='skip'].target_y)
plt.scatter([t[1]*400 for t in tasks],[t[0]*400 for t in tasks])


# value of all tasks   
def getvalue(task):
    task=torch.tensor(task).float()
    env.reset(goal_position=task,theta=theta,phi=phi)
    value=critic(torch.cat([env.wrap_decision_info().flatten(),agent(env.wrap_decision_info(),).flatten()]))
    return value

with torch.no_grad():
    values=[]
    for task in tasks:
        values.append(getvalue(task))


# get distance and angles
distances,angles=[],[]
for task in tasks:
    d,a=xy2pol(task,rotation=False)
    distances.append(d)
    angles.append(abs(a))

# get trial dur
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
pre=trialtypeencoding['normal']
cur=trialtypeencoding['skip']
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


plt.scatter(normalds,normalas,label='normal',alpha=0.9,s=0.5)
plt.scatter(skipds,skipas,label='skip',alpha=0.9,s=2)
plt.title('normal trials vs skip trials')
plt.xlabel('distance')
plt.ylabel('angles')
plt.legend()


plt.hist([a/d for a,d in zip(skipas,skipds)],alpha=0.5,density =True,label='skip')
plt.hist([a/d for a,d in zip(normalas,normalds)],alpha=0.5,density =True,label='normal')
plt.title('skip vs normal, angle/distance')
plt.xlabel('target angle/distance')
plt.ylabel('probability')
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



plt.hist(df[df.category=='skip'].trial_dur,bins=30,label='skip',alpha=0.7,density =True)
plt.hist(df[df.category=='normal'].trial_dur,bins=30,label='normal',alpha=0.7,density =True)
plt.title('normal trials vs skip trials')
plt.xlabel('time dt')
plt.ylabel('probability')
plt.legend()






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


# basic param
batch_size,num_steps=32,30
num_hidden=64

# make data

rnndataset=[]
for i in range(len(encodedtrialtype)):
    onehot=torch.zeros(len(vocab))
    onehot[encodedtrialtype[i]]=1
    # one hot only
    rnndataset.append(onehot)
    # distance
    # rnndataset.append(torch.cat([onehot, torch.ones(1)*distances[i]]))
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

data=[[xx,yy] for xx,yy in zip(x,y)]
training_data=data[:(len(data)//batch_size-2)*(batch_size)]
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
train_iter=iter(train_dataloader)

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

rnn_layer=nn.RNN(rnndataset[0].shape[0],num_hidden)
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=4)
net = net.to(device)
num_epochs, lr = 500, 1

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
    loss = nn.CrossEntropyLoss(weight=classw)
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

import sklearn
classw=sklearn.utils.class_weight.compute_class_weight('balanced',np.unique([torch.argmax(yy).item() for yy in y]),[torch.argmax(yy).item() for yy in y])
classw=torch.tensor(classw).float()



# predict and retrain
for i in range(9):
    train_ch8(net, train_dataloader, vocab, lr=0.01,num_epochs=3,device=device)
    count=0
    testtrue,testpred=[],[]
    # warmup
    state = net.begin_state(batch_size=1, device=device)
    for x in rnndataset[(len(data)//batch_size-2)*(batch_size)-num_steps:(len(data)//batch_size-2)*(batch_size)]:
        pred,state=net(x.view(1,1,rnndataset[0].shape[0]),state)
    # collect out
    for i,x in enumerate(rnndataset[(len(data)//batch_size-2)*(batch_size):-1]):
    # for i,x in enumerate(rnndataset[400:500]):        
        pred,state=net(x.view(1,1,rnndataset[0].shape[0]),state)
        print(pred)
        truecat=vocab[torch.argmax(x[:4])]
        cathat=vocab[torch.argmax(pred)]
        testtrue.append(torch.argmax(x[:4]))
        testpred.append(torch.argmax(pred))
        # if cathat!='normal':
        if cathat==truecat:
            count+=1
            # print('predict:',vocab[torch.argmax(pred)])
            # print('true   :',vocab[encodedtrialtype[i+1]])
    print(count/(len(dataset)-(len(data)//batch_size-2)*(batch_size)))



plt.plot(testtrue,label='true')
plt.plot(testpred,label='pred')
plt.title('true vs pred on testset',fontsize=16)
plt.xlabel('trial index')
plt.ylabel('one hot encoding')
plt.legend()
plt.text(0,0,str(trialtypeencoding),fontsize=14)


# validation on training set
l,r=200,300
count=0
testtrue,testpred=[],[]
# warmup
state = net.begin_state(batch_size=1, device=device)
for x in rnndataset[l-num_steps:l]:
    pred,state=net(x.view(1,1,rnndataset[0].shape[0]),state)
# collect out
for i,x in enumerate(rnndataset[l:r]):
# for i,x in enumerate(rnndataset[400:500]):        
    pred,state=net(x.view(1,1,rnndataset[0].shape[0]),state)
    truecat=vocab[torch.argmax(x[:4])]
    cathat=vocab[torch.argmax(pred)]
    testtrue.append(torch.argmax(x[:4]))
    testpred.append(torch.argmax(pred))
    # if cathat!='normal':
    if cathat==truecat:
        count+=1
        # print('predict:',vocab[torch.argmax(pred)])
        # print('true   :',vocab[encodedtrialtype[i+1]])
print(count/(len(dataset)-(len(data)//batch_size-2)*(batch_size)))


plt.plot(testtrue,label='true')
plt.plot(testpred,label='pred')
plt.title('true vs pred on trainingset',fontsize=16)
plt.xlabel('trial index')
plt.ylabel('one hot encoding')
plt.legend()
plt.text(0,0,str(trialtypeencoding),fontsize=14)


# predict multiple steps
l=int(torch.randint(low=num_steps,high=len(rnndataset)-num_steps,size=(1,)))
numpred=10
r=l+numpred
count=10
testtrue,testpred=[],[]
with torch.no_grad():
    # warmup
    state = net.begin_state(batch_size=1, device=device)
    for x in rnndataset[l-num_steps:l]:
        pred,state=net(x.view(1,1,rnndataset[0].shape[0]),state)
    # collect out
    out=[F.one_hot(torch.argmax(pred),len(vocab))]
    for i,x in enumerate(rnndataset[l:r]):     
        pred,state=net(out[-1].float().view(1,1,rnndataset[0].shape[0]),state)
        print(pred)
        truecat=vocab[torch.argmax(x[:4])]
        cathat=vocab[torch.argmax(out[-1])]
        testtrue.append(torch.argmax(x[:4]))
        testpred.append(torch.argmax(out[-1]))
        out.append(F.one_hot(torch.argmax(pred),len(vocab)))

plt.plot(testtrue,label='true')
plt.plot(testpred,label='pred')
plt.title('true vs multiple predictions on trainingset',fontsize=16)
plt.xlabel('trial index')
plt.ylabel('one hot encoding')
plt.legend()
plt.text(0,0,str(trialtypeencoding),fontsize=14)


# multiple predictions of 1 s
numpred=100
count=0
testtrue,testpred=[],[]
testtruepre=[]
for ipred in range(numpred):
    l=int(torch.randint(low=num_steps,high=len(rnndataset)-num_steps,size=(1,)))
    with torch.no_grad():
        # warmup
        state = net.begin_state(batch_size=1, device=device)
        for x in rnndataset[l-num_steps:l]:
            pred,state=net(x.view(1,1,rnndataset[0].shape[0]),state)
        # collect out
        testpred.append(torch.argmax(pred))
        testtrue.append(torch.argmax(rnndataset[l][:4]))
        testtruepre.append(torch.argmax(x[:4]))
        if torch.argmax(pred)==torch.argmax(rnndataset[l]):
            count+=1
print(count/numpred)

plt.plot(testpred)
plt.plot(testtrue)
plt.plot(testtruepre)

# if predict last type
count=0
numpred=len(testtrue)
for a,b in zip(testtrue,testtruepre):
    if a==b:
        count+=1
print(count/numpred)

# if predict all normal
count=0
numpred=len(testtrue)
for a,b in zip(testtrue,testtruepre):
    if a==trialtypeencoding['normal']:
        count+=1
print(count/numpred)

# if predict last type, but use skip prob on next trial
count=0
numpred=len(testtrue)
pmultiskip=(len(df[df.category=='skip'])-rights.count(1))/len(df[df.category=='skip'])
for pre,true in zip(testtruepre,testtrue):
    if pre!=trialtypeencoding['skip']:
        count+=0.89
    elif pre==trialtypeencoding['skip']:
        count+=0.1
print(count/numpred)



# testing 
startingvalue=2.5
factor=0.1
simulation=[startingvalue,startingvalue]
while simulation[-1]>0.0001:
    simulation.append(simulation[-1]*factor)
plt.plot(simulation)
plt.xlim(0,9)
plt.title('fitted prob of skip|skip')
plt.ylabel('p')
plt.xlabel('number of trials')
plt.hist(rights,bins=99,density=True)


startingvalue=0.17
factor=0.89
simulation=[startingvalue]
while simulation[-1]>0.001:
    simulation.append(simulation[-1]*factor)
plt.plot(simulation)
plt.xlim(0,65)
plt.title('fitted prob of normal|normal')
plt.ylabel('p')
plt.xlabel('number of trials')
plt.hist(lefts,bins=99,density=True)





# logistic regression
'''
we think the new type depends on a set of variables.
then, P(new type = normal)=(sumexp wx +b)/(sumexp .... +1),
where w are weights and b is bias, x are depended variables
some of the variables:
prev type
prev prev type 
cur distance
cur value
'''
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
lrdata=[]
lrdata2=[]
for i in range(len(encodedtrialtype)):
    onehot=[0]*(len(vocab))
    onehot[encodedtrialtype[i]]=1
    lrdata.append(onehot+[distances[i],angles[i]])
    lrdata2.append([vocab[encodedtrialtype[i]], distances[i],angles[i]])

num_steps=1
x=lrdata[:-1*num_steps]
y=[np.argmax(yy[:len(vocab)]) for yy in lrdata[num_steps:]]
x=np.array(x)
y=np.array(y)


# use distance and angle to predict cur type
xx=pd.DataFrame(lrdata2)
xxx = xx.select_dtypes(exclude=['number']) \
                .apply(LabelEncoder().fit_transform) \
                .join(xx.select_dtypes(include=['number']))
splitpoint=np.random.randint(0,len(lrdata2)-100)
train=xxx[:splitpoint].append(xxx[splitpoint+100:])
test=xxx[splitpoint:splitpoint+100]
# use distance and angle to predict cur type
model = LogisticRegression(fit_intercept=False)
model.fit(train.drop(0,1), train[0])
print("acc: %.2f" % model.score(test.drop(0,1), test[0], sample_weight=None))


# use previous types to predict
rewarded=[1 if eachrewarded else 0 for eachrewarded in list(df.rewarded)]
num_steps=1

lrdata=[]
for i in range(num_steps-1, len(encodedtrialtype)-num_steps):
    onehots=[]
    onehotsrewarded=[]
    for j in range(num_steps):
        onehot=[0]*(len(vocab))
        onehot[encodedtrialtype[i-j]]=1
        onehots=onehots+onehot
        onehotrewarded=[rewarded[i-j]]
        onehotsrewarded=onehotsrewarded+onehotrewarded
    lrdata.append(onehots+onehotsrewarded+distances[i-num_steps:i+1]+[distances[i+1]])
x=lrdata
y=encodedtrialtype[num_steps:][:len(lrdata)]
x=np.array(x)
y=np.array(y)

totalacc=0
for i in range(10):
    splitpoint=np.random.randint(0,len(lrdata)-100)
    trainx=np.concatenate((x[:splitpoint],x[splitpoint+100:]),axis=0)
    trainy=np.concatenate((y[:splitpoint],y[splitpoint+100:]),axis=0)
    testx=x[splitpoint:splitpoint+100]
    testy=y[splitpoint:splitpoint+100]
    # use distance and angle to predict cur type
    model = LogisticRegression(fit_intercept=False)
    model.fit(trainx,trainy)
    # print("acc: %.2f" % model.score(testx, testy, sample_weight=None))
    totalacc+=model.score(testx, testy, sample_weight=None)
# print('mean acc ',totalacc/10 )

plt.plot(testy,label='true')
plt.plot(model.predict(testx),label='pred')
plt.title('true vs multiple predictions on testset',fontsize=16)
plt.xlabel('trial index',fontsize=16)
plt.ylabel(str(trialtypeencoding),fontsize=14)
plt.legend()


model.coef_.shape

# see if acc increase with more samples
accs=[]
accsstd=[]
for num_steps in range(2,30):
    lrdata=[]
    for i in range(num_steps, len(encodedtrialtype)-num_steps):
        onehots=[]
        onehotsrewarded=[]
        for j in range(num_steps):
            onehot=[0]*(len(vocab))
            onehot[encodedtrialtype[i-j]]=1
            onehots=onehots+onehot
            onehotrewarded=[rewarded[i-j]]
            onehotsrewarded=onehotsrewarded+onehotrewarded
        lrdata.append(onehots+onehotsrewarded+distances[i-num_steps:i+1]+[distances[i+1]])
    x=lrdata
    y=encodedtrialtype[num_steps+1:][:len(lrdata)]
    x=np.array(x)
    y=np.array(y)

    totalacc=[]
    for i in range(30):
        splitpoint=np.random.randint(0,len(lrdata)-100)
        trainx=np.concatenate((x[:splitpoint],x[splitpoint+100:]),axis=0)
        trainy=np.concatenate((y[:splitpoint],y[splitpoint+100:]),axis=0)
        testx=x[splitpoint:splitpoint+100]
        testy=y[splitpoint:splitpoint+100]
        # use distance and angle to predict cur type
        model = LogisticRegression(fit_intercept=False)
        model.fit(trainx,trainy)
        # print("acc: %.2f" % model.score(testx, testy, sample_weight=None))
        totalacc.append(model.score(testx, testy, sample_weight=None))
    accs.append(sum(totalacc)/30)
    accsstd.append(np.std(totalacc))
    print(accs)

plt.errorbar([i for i in range(1,len(accs)+1)],accs,yerr=accsstd)
plt.title('trial type prediction acc vs num prev steps',fontsize=16)
plt.xlabel('number of previous steps',fontsize=16)
plt.ylabel('prediction accuracy',fontsize=14)



# skip analysis on session level
data_path="D:\mkdata\Bruno_density"
import os
sessions=os.listdir(data_path)


marker_memo = {'file_start': 1, 'trial_start': 2, 'trial_end': 3,
                            'juice' :4, 'perturb_start': 8, 'perturb_start2': 5}
y_offset = 32.5
            

import neo
seg_reader = neo.io.Spike2IO(filename=file_name).read_segment()

        
def bcm_extract_smr():
    channel_signal_all = []
    marker_all = []
    for idx, file_name in enumerate(smr_full_file_path):
        seg_reader = neo.io.Spike2IO(filename=file_name).read_segment()
        
        if idx == 0: # only get sampling rate once
            SAMPLING_RATE = seg_reader.analogsignals[0].sampling_rate.item()
            
        # Sometimes the length across channels varies a bit
        analog_length = min([i.size for i in seg_reader.analogsignals])
        channel_signal = np.ones((analog_length, seg_reader.size['analogsignals']))
        
        channel_names = []
        # Do not read the last channel as it has a unique shape.
        for ch_idx, ch_data in enumerate(seg_reader.analogsignals[:-1]):
            channel_signal[:, ch_idx] = ch_data.as_array()[:analog_length].T
            channel_names.append(ch_data.annotations['channel_names'][0])
        
        # Add a time channel
        channel_signal[:, -1] = seg_reader.analogsignals[0].times[:analog_length]
        channel_names.append('Time') 
        
        channel_signal_all.append(pd.DataFrame(channel_signal,columns=channel_names))
        
        marker_channel_idx = [idx for idx, value 
                                in enumerate(seg_reader.events)
                                if value.name == 'marker'][0]
        marker_key, marker_time = (
            seg_reader.events[marker_channel_idx].get_labels().astype('int'),
            seg_reader.events[marker_channel_idx].as_array())
        marker = {'key': marker_key, 'time': marker_time}
        marker_all.append(marker)
        
        channel_signal_all = channel_signal_all
        marker_all = marker_all
            
def bcm_extract_log():
        log_data_all = []
        
        for file_name in self.log_full_file_path:
            with open(file_name, 'r', encoding='UTF-8') as content:
                log_content = content.readlines()
                
            floor_density = []
            perturb_vpeak = []; perturb_wpeak = []
            perturb_start_time_ori = []
            full_on = []
            for line in log_content:
                if 'Joy Stick Max Velocity' in line:
                    gain_v = float(line.split(': ')[1])
                    
                if 'Joy Stick Max Angular Velocity' in line:
                    gain_w = float(line.split(': ')[1])
                    
                if 'Perturb Max Velocity' in line:
                    perturb_vpeakmax = float(line.split(': ')[1])
                    
                if 'Perturb Max Angular Velocity' in line:
                    perturb_wpeakmax = float(line.split(': ')[1])
                    
                if 'Perturbation Sigma' in line:
                    perturb_sigma = float(line.split(': ')[1])
                    
                if 'Perturbation Duration' in line:
                    perturb_dur = float(line.split(': ')[1])
                    
                if 'Floor Density' in line:
                    content_temp = float(line.split(': ')[1])
                    floor_density.append(content_temp)
                    
                if 'Perturbation Linear Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_vpeak.append(content_temp)
                    
                if 'Perturbation Angular Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_wpeak.append(- content_temp)
                    
                if 'Perturbation Delay Time' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_start_time_ori.append(content_temp / 1000)
                    
                if 'Firefly Full ON' in line:
                    content_temp = bool(int(line.split(': ')[1]))
                    full_on.append(content_temp)
            
            if len(full_on) == 0: # Quigley's perturbation sessions
                full_on = [False] * len(floor_density)
            
            log_data_all.append({'gain_v': gain_v, 'gain_w': gain_w, 
                                 'perturb_vpeakmax': perturb_vpeakmax, 'perturb_wpeakmax': perturb_wpeakmax,
                                 'perturb_sigma': perturb_sigma, 'perturb_dur': perturb_dur,
                                 'floor_density': floor_density, 
                                 'perturb_vpeak': perturb_vpeak, 'perturb_wpeak': perturb_wpeak,
                                 'perturb_start_time_ori': perturb_start_time_ori,
                                 'full_on': full_on})
            self.log_data_all = log_data_all
            
def bcm_segment(lazy_threshold=4000, skip_threshold=400, skip_r_threshold=30, 
                    crazy_threshold=200,
                    medfilt_kernel=5, v_threshold=1, reward_boundary=65,
                    target_r_range=[100, 400], target_theta_range=[55, 125], 
                    target_tolerance=1, perturb_corr_threshold=100):
        # lazy_threshold (data points): Trial is too long.
        # skip_threshold (data points): Trial is too short.
        # skip_r_threshold (cm): Monkey did not move a lot.
        # crazy_threshold (cm): Monkey stopped too far.
        # medfilt_kernel (data points): Remove spikes from raw data.
        # v_threshold (cm/s): Threshold for end point.
        # reward_boundary (cm): Rewarded when stop inside this circular boundary.
        # target_r_range (cm): Radius of target distribution.
        # target_theta_range (deg): Angle of target distribution.
        # target_tolerance (cm or deg): Max tolerance for targets out of distribution.
        # perturb_corr_threshold (data points): Corrected perturbation start index should not be too biased.

        gain_v = []; gain_w = []; perturb_vpeakmax = []; perturb_wpeakmax = []
        perturb_sigma = []; perturb_dur = []; perturb_vpeak = []; perturb_wpeak = []
        perturb_v = []; perturb_w = []; perturb_v_gauss = []; perturb_w_gauss = []
        perturb_start_time = []; perturb_start_time_ori = []
        floor_density = []; pos_x = []; pos_y = []
        head_dir = []; head_dir_end = []; pos_r = []; pos_theta = []; 
        pos_r_end = []; pos_theta_end = []
        pos_v = []; pos_w = []; target_x = []; target_y = []
        target_r = []; target_theta = []; full_on = []; rewarded = []
        relative_radius = []; relative_angle = []; time = []; 
        trial_dur = []; action_v = []; action_w = []; 
        relative_radius_end = []; relative_angle_end = []; category = []

        for session_idx, session_data in enumerate(self.channel_signal_all):
            log_data = self.log_data_all[session_idx]
            marker_data = self.marker_all[session_idx]
            start_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_start']]
            end_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_end']]
            perturb_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['perturb_start']]
            if perturb_marker_times.size == 0:
                perturb_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['perturb_start2']]

            # segment trials
            for trial_idx in range(end_marker_times.size):
                trial_data = session_data[np.logical_and(
                    session_data.Time > start_marker_times[trial_idx],
                    session_data.Time < end_marker_times[trial_idx])].copy()

                # Use median filter kernel size as 5 to remove spike noise first.
                trial_data['ForwardV'] = medfilt(trial_data['ForwardV'], medfilt_kernel)
                trial_data['AngularV'] = medfilt(trial_data['AngularV'], medfilt_kernel)
                
                # cut non-moving head and tail
                moving_period = np.where(trial_data['ForwardV'].abs() > v_threshold)[0]
                if moving_period.size > 0:
                    start_idx = moving_period[0]
                    end_idx = moving_period[-1] + 2
                else:
                    start_idx = 0
                    end_idx = None
                    
                # store trial data
                trial_data = trial_data.iloc[start_idx : end_idx]
                trial_data['AngularV'] = - trial_data['AngularV']
                trial_data['MonkeyYa'] = np.cumsum(trial_data['AngularV']) / self.SAMPLING_RATE + 90
                trial_data['MonkeyX'] = np.cumsum(trial_data['ForwardV']
                                            * np.cos(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                trial_data['MonkeyY'] = np.cumsum(trial_data['ForwardV']
                                            * np.sin(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                
                gain_v.append(log_data['gain_v'])
                gain_w.append(log_data['gain_w'])
                perturb_vpeakmax.append(log_data['perturb_vpeakmax'])
                perturb_wpeakmax.append(log_data['perturb_wpeakmax'])
                perturb_vpeak.append(log_data['perturb_vpeak'][trial_idx])
                perturb_wpeak.append(log_data['perturb_wpeak'][trial_idx])
                perturb_start_time_ori.append(log_data['perturb_start_time_ori'][trial_idx])
                perturb_sigma.append(log_data['perturb_sigma'])
                perturb_dur.append(log_data['perturb_dur'])
                pos_x.append(trial_data['MonkeyX'].values)
                pos_y.append(trial_data['MonkeyY'].values)
                head_dir.append(trial_data['MonkeyYa'].values)
                head_dir_end.append(trial_data['MonkeyYa'].values[-1])
                floor_density.append(log_data['floor_density'][trial_idx])
                full_on.append(log_data['full_on'][trial_idx])
                
                rho, phi = cart2pol(pos_x[-1], pos_y[-1])
                pos_r.append(rho)
                pos_theta.append(np.rad2deg(phi))
                pos_r_end.append(rho[-1])
                pos_theta_end.append(np.rad2deg(phi[-1]))
                
                
                # determine if it is a perturbation trial
                perturb_start_time_temp = perturb_marker_times[(np.logical_and(
                                            perturb_marker_times > start_marker_times[trial_idx],
                                            perturb_marker_times < end_marker_times[trial_idx]))]
                if bool(perturb_start_time_temp.size):
                    assert perturb_start_time_temp.size == 1
                    pos_v.append(trial_data['ForwardV'].values)
                    pos_w.append(trial_data['AngularV'].values)
                
                    # construct perturbation curves
                    perturb_xaxis = np.linspace(0, perturb_dur[-1], round(self.SAMPLING_RATE))
                    perturb_temp = norm.pdf(perturb_xaxis, loc=perturb_dur[-1] / 2, scale=perturb_sigma[-1])
                    perturb_temp /= perturb_temp.max()
                    perturb_v_temp = perturb_temp * perturb_vpeak[-1]
                    perturb_w_temp = perturb_temp * perturb_wpeak[-1]
                    perturb_v_gauss.append(perturb_v_temp)
                    perturb_w_gauss.append(perturb_w_temp)
                    
                    # use the more obvious perturbation curve as a template
                    corrcoef_v = np.correlate(pos_v[-1] - pos_v[-1].mean(), 
                                              perturb_v_temp - perturb_v_temp.mean(),
                                              mode='same') / (pos_v[-1].std() * perturb_v_temp.std())
                    corrcoef_w = np.correlate(pos_w[-1] - pos_w[-1].mean(), 
                                              perturb_w_temp - perturb_w_temp.mean(),
                                              mode='same') / (pos_w[-1].std() * perturb_w_temp.std())
                    if corrcoef_v.max() > corrcoef_w.max():
                        perturb_template = perturb_v_temp
                        original_vel = pos_v[-1]
                    else:
                        perturb_template = perturb_w_temp
                        original_vel = pos_w[-1]
                        
                    # use the template to do cross-correlation to find perturbation start time
                    perturb_start_idx_mark = int((perturb_start_time_temp
                                                - trial_data['Time'].values[0]) * self.SAMPLING_RATE)
                    perturb_start_idx_mark = np.clip(perturb_start_idx_mark, 0, None)
                    perturb_peak_idx = np.correlate(original_vel, perturb_template, mode='same').argsort()[::-1]
                    perturb_start_idx_corr = perturb_peak_idx - perturb_dur[-1] / 2 * self.SAMPLING_RATE
                    mask = (perturb_start_idx_corr > 0) \
                           & (perturb_start_idx_corr > perturb_start_idx_mark) \
                           & (perturb_start_idx_corr - perturb_start_idx_mark < perturb_corr_threshold)
                    
                    if mask.sum() == 0 or original_vel.size < perturb_template.size:
                        perturb_start_idx = np.clip(perturb_start_idx_mark, None, pos_v[-1].size - 1)
                    else:
                        perturb_start_idx = int(perturb_start_idx_corr[mask][0])
                    perturb_start_time.append(perturb_start_idx / self.SAMPLING_RATE)
                    
                    # get pure actions
                    perturb_v_full = np.zeros_like(pos_v[-1])
                    perturb_v_full[perturb_start_idx:perturb_start_idx + perturb_v_temp.size] = \
                                                            perturb_v_temp[:perturb_v_full.size - perturb_start_idx]
                    perturb_w_full = np.zeros_like(pos_w[-1])
                    perturb_w_full[perturb_start_idx:perturb_start_idx + perturb_w_temp.size] = \
                                                            perturb_w_temp[:perturb_w_full.size - perturb_start_idx]
                    
                    perturb_v.append(perturb_v_full); perturb_w.append(perturb_w_full)
                    action_v.append((pos_v[-1] - perturb_v_full).clip(-gain_v[-1], gain_v[-1]) / gain_v[-1])
                    action_w.append((pos_w[-1] - perturb_w_full).clip(-gain_w[-1], gain_w[-1]) / gain_w[-1])
                else:
                    pos_v.append(trial_data['ForwardV'].values.clip(-gain_v[-1], gain_v[-1]))
                    pos_w.append(trial_data['AngularV'].values.clip(-gain_w[-1], gain_w[-1]))
                    perturb_v_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_w_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_start_time.append(np.nan)
                    perturb_v.append(np.zeros_like(pos_v[-1])); perturb_w.append(np.zeros_like(pos_w[-1]))
                    action_v.append(pos_v[-1] / gain_v[-1])
                    action_w.append(pos_w[-1] / gain_w[-1])
                

                trial_data['Time'] -= trial_data['Time'].iloc[0]
                time.append(trial_data['Time'].values)
                trial_dur.append(trial_data['Time'].values[-1])
                
                
                # target position is analog in BCM data, I bin target channels
                # and find the mode of bins
                targetx_bins = np.arange(my_floor(trial_data['FireflyX'].min(), 1),
                                         my_ceil(trial_data['FireflyX'].max(), 1), 0.1)
                targetx_idxes = np.digitize(trial_data['FireflyX'], targetx_bins)
                targetx_hist, _ = np.histogram(trial_data['FireflyX'], targetx_bins)
                try:
                    tar_x = trial_data['FireflyX'][
                                    targetx_idxes == targetx_hist.argmax()+1].mean()
                except: # when start_idx == end_idx, they are bad trials that not matter
                    tar_x = trial_data['FireflyX'].mean()

                targety_bins = np.arange(my_floor(trial_data['FireflyY'].min(), 1),
                                         my_ceil(trial_data['FireflyY'].max(), 1), 0.1)
                targety_idxes = np.digitize(trial_data['FireflyY'], targety_bins)
                targety_hist, _ = np.histogram(trial_data['FireflyY'], targety_bins)
                try:
                    tar_y = trial_data['FireflyY'][
                                    targety_idxes == targety_hist.argmax()+1].mean()
                except:
                    tar_y = trial_data['FireflyY'].mean()

                target_x.append(tar_x)
                target_y.append(- tar_y + self.y_offset)
                tar_rho, tar_phi = cart2pol(target_x[-1], target_y[-1])
                target_r.append(tar_rho)
                target_theta.append(np.rad2deg(tar_phi))

                relative_r, relative_ang = get_relative_r_ang(
                                pos_x[-1], pos_y[-1], head_dir[-1], target_x[-1], target_y[-1])
                relative_radius.append(relative_r)
                relative_angle.append(np.rad2deg(relative_ang))
                relative_radius_end.append(relative_r[-1])
                relative_angle_end.append(np.rad2deg(relative_ang[-1]))
                rewarded.append(relative_r[-1] < reward_boundary)

                #juice_time = marker_data['time'][marker_data['key'] == marker_memo['juice']]
                #j_marker = np.where(np.logical_and(juice_time > start_marker_times[trial_idx],
                #               juice_time < end_marker_times[trial_idx]))[0]

                # Categorize trials
                # Note that few targets in BCM data are out of the distribution
                # for unknown reason, I just label and ignore them.
                if target_r[-1] < target_r_range[0] - target_tolerance or\
                   target_r[-1] > target_r_range[1] + target_tolerance or\
                   target_theta[-1] < target_theta_range[0] - target_tolerance or\
                   target_theta[-1] > target_theta_range[1] + target_tolerance:
                    category.append('wrong_target')
                else:
                    if rewarded[-1]:
                        category.append('normal')
                    else:
                        if trial_data['ForwardV'].size < skip_threshold or\
                           pos_r_end[-1] < skip_r_threshold:
                            category.append('skip')
                        elif trial_data['ForwardV'].size > lazy_threshold:
                            category.append('lazy')
                        elif relative_r[-1] > crazy_threshold:
                            category.append('crazy')
                        else:
                            category.append('normal')

        # Construct a dataframe   
        self.monkey_trajectory = pd.DataFrame().assign(gain_v=gain_v, gain_w=gain_w, 
                                 perturb_vpeakmax=perturb_vpeakmax, perturb_wpeakmax=perturb_wpeakmax,
                                 perturb_sigma=perturb_sigma, perturb_dur=perturb_dur,
                                 perturb_vpeak=perturb_vpeak, perturb_wpeak=perturb_wpeak,
                                 perturb_start_time=perturb_start_time,
                                 perturb_start_time_ori=perturb_start_time_ori,
                                 perturb_v_gauss=perturb_v_gauss, perturb_w_gauss=perturb_w_gauss,
                                 perturb_v=perturb_v, perturb_w=perturb_w,
                                 floor_density=floor_density, pos_x=pos_x,
                                 pos_y=pos_y, head_dir=head_dir, head_dir_end=head_dir_end,
                                 pos_r=pos_r, pos_theta=pos_theta, pos_r_end=pos_r_end,
                                 pos_theta_end=pos_theta_end, pos_v=pos_v, pos_w=pos_w, 
                                 target_x=target_x, target_y=target_y, target_r=target_r,
                                 target_theta=target_theta, full_on=full_on, rewarded=rewarded,
                                 relative_radius=relative_radius, relative_angle=relative_angle,
                                 time=time, trial_dur=trial_dur, 
                                 action_v=action_v, action_w=action_w, 
                                 relative_radius_end=relative_radius_end,
                                 relative_angle_end=relative_angle_end, category=category)



i=sessions[0]
j=data_path+'/'+i
files=os.listdir(j)
k=files[1]
file=j+'/'+k

# .log
with open(file, 'r', encoding='UTF-8') as content:
    log_content = content.readlines()
    for line in log_content:
        print(line)

# .smr
channel_signal_all = []
marker_all = []

seg_reader = neo.io.Spike2IO(filename=file).read_segment()
SAMPLING_RATE = seg_reader.analogsignals[0].sampling_rate.item()
    
# Sometimes the length across channels varies a bit
analog_length = min([i.size for i in seg_reader.analogsignals])
channel_signal = np.ones((analog_length, seg_reader.size['analogsignals']))

channel_data={}
channel_names = []
# Do not read the last channel as it has a unique shape.
for ch_idx, ch_data in enumerate(seg_reader.analogsignals[:-1]):
    channel_data[ch_idx] = ch_data.as_array()[:analog_length].T
    channel_names.append(ch_data.name)

# Add a time channel
channel_data[ch_idx+1] = seg_reader.analogsignals[0].times[:analog_length]
channel_names.append('Time') 


marker_channel_idx = [idx for idx, value 
                        in enumerate(seg_reader.events)
                        if value.name == 'marker'][0]
marker_key, marker_time = (
    seg_reader.events[marker_channel_idx].get_labels().astype('int'),
    seg_reader.events[marker_channel_idx].as_array())
marker = {'key': marker_key, 'time': marker_time}
marker_all.append(marker)

channel_signal_all = channel_signal_all
marker_all = marker_all













# hw code
import torch
from torch import nn
from collections import OrderedDict
from matplotlib import pyplot as plt

nlayers=torch.randint(high=10,low=2,size=(1,))
nnodes=[torch.randint(high=10,low=2,size=(1,)) for i in range(nlayers)]


def makemodel(nnodes):
    arch=[]
    prev=2
    for ind,each in enumerate(nnodes):
        arch=arch+[('linear'+str(ind), nn.Linear(prev,each))]
        prev=each
        arch=arch+[('tanh'+str(ind), nn.Tanh())]
    arch.append(('linearout', nn.Linear(prev,1)))
    arch.append(('tanh',nn.Tanh()))
    model=nn.Sequential(OrderedDict(arch))
    return model

model=makemodel(nnodes)


x=torch.zeros((1000,2)).uniform_(-1,1)

with torch.no_grad():
    y=model(x)

cbar=plt.scatter(x[:,0], x[:,1],c=y.view(-1).tolist(), cmap='bwr')
plt.colorbar(cbar,label='y out')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()



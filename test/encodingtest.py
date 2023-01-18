
import torch
from torch import nn






# example task

class Task(nn.Module):
    
    def __init__(self, goal=None,behavior=True):
        self.goalsize=torch.ones(1)
        self.goalsize=self.goalsize*goal if goal else self.goalsize*torch.rand(1)
        self.behavior=behavior

    def reset(self,x=None):
        self.d=torch.ones(1)
        self.d=self.goalsize*x if x else self.goalsize*torch.rand(1)*20
        self.s=torch.cat(self.d,torch.zeros(1))

    def dynamic(self,a):
        self.s[0]+=self.s[1]
        self.s[1]=a
    
    def obs(self,):
        if self.behavior:
            return self.s
        else:
            return 


# random encoder
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.ConvTranspose1d(1, 6, 5)
        self.conv2 = nn.ConvTranspose1d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)




# example

import scipy as sp
import numpy as np
import pylab as plt

#Constants
C_m  =   1.0 #membrane capacitance, in uF/cm^2"""
g_Na = 120.0 #Sodium (Na) maximum conductances, in mS/cm^2""
g_K  =  36.0 #Postassium (K) maximum conductances, in mS/cm^2"""
g_L  =   0.3 #Leak maximum conductances, in mS/cm^2"""
E_Na =  50.0 #Sodium (Na) Nernst reversal potentials, in mV"""
E_K  = -77.0 #Postassium (K) Nernst reversal potentials, in mV"""
E_L  = -54.387 #Leak Nernst reversal potentials, in mV"""

def poisson_spikes(t, N=100, rate=1.0 ):
    spks = []
    dt = t[1] - t[0]
    for n in range(N):
        spkt = t[np.random.rand(len(t)) < rate*dt/1000.] #Determine list of times of spikes
        idx = [n]*len(spkt) #Create vector for neuron ID number the same length as time
        spkn = np.concatenate([[idx], [spkt]], axis=0).T #Combine tw lists
        if len(spkn)>0:        
            spks.append(spkn)
    spks = np.concatenate(spks, axis=0)
    return spks

N = 100
N_ex = 80 #(0..79)
N_in = 20 #(80..99)
G_ex = 1.0
K = 4

dt = 0.01
t = sp.arange(0.0, 300.0, dt) #The time to integrate over """
ic = [-65, 0.05, 0.6, 0.32]

spks =  poisson_spikes(t, N, rate=10.)

def alpha_m(V):
        return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))

def beta_m(V):
        return 4.0*sp.exp(-(V+65.0) / 18.0)

def alpha_h(V):
        return 0.07*sp.exp(-(V+65.0) / 20.0)

def beta_h(V):
        return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))

def alpha_n(V):
        return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))

def beta_n(V):
        return 0.125*sp.exp(-(V+65) / 80.0)

def I_Na(V, m, h):
        return g_Na * m**3 * h * (V - E_Na)

def I_K(V, n):
        return g_K  * n**4 * (V - E_K)

def I_L(V):
        return g_L * (V - E_L)

def I_app(t):
        return 3

def I_syn(spks, t):
    """
    Synaptic current
    spks = [[synid, t],]
    """
    exspk = spks[spks[:,0]<N_ex] # Check for all excitatory spikes
    delta_k = exspk[:,1] == t # Delta function
    if sum(delta_k) > 0:
        h_k = np.random.rand(len(delta_k)) < 0.5 # p = 0.5
    else:
        h_k = 0

    inspk = spks[spks[:,0] >= N_ex] #Check remaining neurons for inhibitory spikes
    delta_m = inspk[:,1] == t #Delta function for inhibitory neurons
    if sum(delta_m) > 0:
        h_m = np.random.rand(len(delta_m)) < 0.5 #p =0.5
    else:
        h_m = 0

    isyn = C_m*G_ex*(np.sum(h_k*delta_k) - K*np.sum(h_m*delta_m))

    return  isyn


def dALLdt(X, t):
        V, m, h, n = X
        dVdt = (I_app(t)+I_syn(spks,t)-I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m
        dmdt = alpha_m(V)*(1.0-m) - beta_m(V)*m
        dhdt = alpha_h(V)*(1.0-h) - beta_h(V)*h
        dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
        return np.array([dVdt, dmdt, dhdt, dndt])

X = [ic]
for i in t[1:]:
    dx = (dALLdt(X[-1],i))
    x = X[-1]+dt*dx
    X.append(x)

X = np.array(X)    
V = X[:,0]        
m = X[:,1]
h = X[:,2]
n = X[:,3]
ina = I_Na(V, m, h)
ik = I_K(V, n)
il = I_L(V)

plt.figure()
plt.subplot(3,1,1)
plt.title('Hodgkin-Huxley Neuron')
plt.plot(t, V, 'k')
plt.ylabel('V (mV)')

plt.subplot(3,1,2)
plt.plot(t, ina, 'c', label='$I_{Na}$')
plt.plot(t, ik, 'y', label='$I_{K}$')
plt.plot(t, il, 'm', label='$I_{L}$')
plt.ylabel('Current')
plt.legend()

plt.subplot(3,1,3)
plt.plot(t, m, 'r', label='m')
plt.plot(t, h, 'g', label='h')
plt.plot(t, n, 'b', label='n')
plt.ylabel('Gating Value')
plt.legend()

plt.show()
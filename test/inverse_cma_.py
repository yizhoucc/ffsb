

# """    save all sampled points, (theta, logll), to plot the countour plot at the end
#     can skip saving generation attr, mu and cov, they can be calculated from all points"""



from numpy.core.defchararray import array
from FireflyEnv.env_utils import is_pos_def
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import heapq
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt
import time
from stable_baselines3 import TD3
torch.manual_seed(42)
from numpy import pi
from InverseFuncs import *
from monkey_functions import *
from FireflyEnv import ffacc_real
from Config import Config
from test.cma_mpi_helper import run
import ray
ray.init(log_to_driver=False,ignore_reinit_error=True,include_dashboard=True)

# def check_envs():
# 	print("scan..")
# 	for ev in os.environ:
# 	    if "KMP" in ev:
# 	        print(ev, os.environ[ev])
# 	    if "OMP" in ev:
# 	        print(ev, os.environ[ev])
# 	print(".. finished scan")

# ray.available_resources()
# check_envs()
# ray.shutdown()



arg = Config()

import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager, ExitStack
@contextmanager
def suppress(out=True, err=False):
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield
            


env=ffacc_real.FireFlyPaper(arg)
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()

print('loading data')
note='testdcont'
with open("C:/Users/24455/Desktop/bruno_pert_downsample",'rb') as f:
        df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
# df=df[df.target_r>250]
df=df[df.floor_density==0.005]
# floor density are in [0.0001, 0.0005, 0.001, 0.005]
df=df[:-100]
print('process data')
states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
print('done process data')

# misc
phi=torch.tensor([[0.5],
        [pi/2],
        [0.001],
        [0.001],
        [0.001],
        [0.001],
        [0.13],
        [0.001],
        [0.001],
        [0.001],
        [0.001],
])
# init condition, we want to at least cover some dynamic range
init_theta=torch.tensor([[0.5],   
        [1.6],   
        [0.5],   
        [0.5],   
        [0.5],   
        [0.5],   
        [0.13],   
        [0.1],   
        [0.1],   
        [0.1],   
        [0.1]])
dim=init_theta.shape[0]
init_cov=torch.diag(torch.ones(dim))*0.3
cur_mu=init_theta.view(-1)
cur_cov=init_cov

def boundary(theta):
    return theta.clamp(1e-3,999)

def invalidparam(theta):
    return not torch.all(torch.tensor(theta)>0.)

def invaliddegree(p):
    negsum=0.
    for pp in p:
        if pp<0:
            negsum-=pp
    return negsum


def samplingparam(offset,cov,samplesize=10):
    dist=MultivariateNormal(loc=offset, covariance_matrix=cov)
    return dist.sample_n(samplesize)

def stepctrl(oldmu,newmu,oldcov,newcov):
    d=newmu-oldmu
    d=d.clamp(-0.01,0.01)
    newmu=oldmu+d

    # sizeratio=torch.det(newcov)/torch.det(oldcov)
    # if 0.01<sizeratio<0.5:
    #     newcov=newcov*(1/sizeratio*0.5)**0.5
    # elif sizeratio>9:
    #     newcov=newcov*(1/sizeratio*9)**0.5
    return newmu,newcov


# cur_mu=torch.rand(11)
cur_mu= torch.tensor([0.5, 1.57, 0.7, 0.7, 0.7, 0.7, 0.13, 0.1, 0.1, 0.1, 0.1])
# cur_cov=torch.tensor([[ 0.0685,  0.0851,  0.1082,  0.0237, -0.0779, -0.0727, -0.0037, -0.0423,
#          -0.0354, -0.0152,  0.0398],
#         [ 0.0851,  0.1194,  0.1342,  0.0425, -0.0952, -0.0988, -0.0091, -0.0496,
#          -0.0596, -0.0186,  0.0624],
#         [ 0.1082,  0.1342,  0.1811,  0.0315, -0.1234, -0.1144, -0.0042, -0.0690,
#          -0.0540, -0.0376,  0.0635],
#         [ 0.0237,  0.0425,  0.0315,  0.0313, -0.0252, -0.0321, -0.0086, -0.0078,
#          -0.0285, -0.0018,  0.0268],
#         [-0.0779, -0.0952, -0.1234, -0.0252,  0.0939,  0.0819,  0.0038,  0.0531,
#           0.0388,  0.0135, -0.0453],
#         [-0.0727, -0.0988, -0.1144, -0.0321,  0.0819,  0.0847,  0.0063,  0.0442,
#           0.0481,  0.0130, -0.0513],
#         [-0.0037, -0.0091, -0.0042, -0.0086,  0.0038,  0.0063,  0.0028,  0.0011,
#           0.0076, -0.0002, -0.0065],
#         [-0.0423, -0.0496, -0.0690, -0.0078,  0.0531,  0.0442,  0.0011,  0.0387,
#           0.0200,  0.0043, -0.0216],
#         [-0.0354, -0.0596, -0.0540, -0.0285,  0.0388,  0.0481,  0.0076,  0.0200,
#           0.0377,  0.0041, -0.0354],
#         [-0.0152, -0.0186, -0.0376, -0.0018,  0.0135,  0.0130, -0.0002,  0.0043,
#           0.0041,  0.0328, -0.0097],
#         [ 0.0398,  0.0624,  0.0635,  0.0268, -0.0453, -0.0513, -0.0065, -0.0216,
#          -0.0354, -0.0097,  0.0377]])
numworker=15
selection_ratio=0.5 # ratio of evolution
steps=20 # number of updates
samplesize=numworker*4 # population size
ircsample=5 # irc sample per mk trial
action_var=0.01 # assumed monkey action var
batchsize=500 # number of mk trials in batch
numepoch=10
# batch trial count will be: population*batchsize*ircsamplesize


# start solving
log=[]
for epoch in range(numepoch):
    # _batchsize=int(epoch/steps*(len(tasks)-batchsize))+batchsize
    _batchsize=batchsize
    ithstep=0
    for batch_states, batch_actions, batch_tasks in data_iter(_batchsize,states, actions, tasks):
        if ithstep>steps:
            break
        ithstep+=1
        if len(tasks)<=2: # that last bactch of 1 at the end of epoch
            break
        batchlog=[]
        # use the data in batch to evaluate params
        # sample params
        sampledparam=samplingparam(cur_mu,cur_cov,samplesize=samplesize).tolist()

        start=time.time()

        # @ray.remote(num_cpus=numworker)
        @ray.remote
        def getlogll(x):
            bad=invaliddegree(x)
            if invaliddegree(x)>0:
                return bad*999
            with torch.no_grad():
                return  monkeyloss(agent, batch_actions, batch_tasks, phi, torch.tensor(x).t(), env, action_var=action_var,num_iteration=1, states=batch_states, samples=ircsample,gpu=False).item()


        # solve for logll
        print('starting {}th step'.format(ithstep))
        start=time.time()
        with suppress():
            res=ray.get([getlogll.remote(p) for p in sampledparam])
        print('ray finished in {:.1f}'.format(time.time()-start))

        r=[[r,p] for r,p in zip(res, sampledparam)]
        r.sort(key=lambda a: a[0])
        nextgenparam=[p[1] for p in r[:int(selection_ratio*samplesize)]]
        new_cov=torch.tensor(np.cov(torch.tensor(nextgenparam).t())).float()
        if not is_pos_def(new_cov):
            print('non pos def cov')
            new_cov = (new_cov + new_cov.t()) / 2 + 1e-6 * torch.eye(dim)
        
        new_mu=torch.mean(torch.tensor(nextgenparam),0)
        # tmp=cur_cov
        cur_mu,cur_cov=stepctrl(cur_mu,new_mu,cur_cov,new_cov)
        cur_mu[6]=0.13

        print('cur -logll',np.mean((res[:int(selection_ratio*samplesize)])),'\ncurrent mu ',cur_mu, '\ncur cov',np.sqrt(np.diag(cur_cov)).tolist())
        log.append([cur_mu,cur_cov,res])

        import pickle
        with open('cmamyinit', 'wb') as handle:
            pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)

        mulog=[l[0] for l in log]
        lastcov=log[-1][1]
            # trend
        fig=plt.figure(figsize=(20,2))
        for n in range(dim):
            ax=fig.add_subplot(1,dim,n+1,)
            y=[t[n] for t in mulog]
            ax.plot(y)
            # ax.set_ylim([min(parameter_range[n][0],min(y)[0]),max(parameter_range[n][1],max(y)[0])])
        plt.show()

        losslog=[np.mean(l[2]) for l in log]
        plt.plot(losslog)



import pickle
with open('cmaquicknocovctrlrand', 'rb') as f:
    r = pickle.load(f)

finaltheta=r[-1][0]
finalcov=r[-1][1]

import torch
import numpy as np
torch.diag(finalcov)**0.5
torch.eig(finalcov)

import matplotlib.pyplot as plt

theta_names = [ 'pro gain v',
                'pro gain w',
                'pro noise v',
                'pro noise w',
                'obs noise v',
                'obs noise w',
                'goal radius',
                'action cost v',      
                'action cost w',      
                'init uncertainty x',      
                'init uncertainty y',      
                ]

fig=plt.figure()
ax = fig.add_subplot(111)
# Create bars and choose color
ax.bar([i for i in range(11)], finaltheta,yerr=torch.diag(finalcov)**0.5*3,color = 'tab:blue')
# title and axis names
ax.set_ylabel('inferred parameter value')
ax.set_xticks([i for i in range(11)])
ax.set_xticklabels(theta_names, rotation=45, ha='right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


theta_mean=np.array([
    0.5,
    1.57,
    0.5,
    0.5,
    0.5,
    0.5,
    0.13,
    0.5,
    0.5,
    0.5,
    0.5    
])
# plt.bar([i for i in range(11)], torch.diag(finalcov)**0.5*1/theta_mean)
fig=plt.figure()
ax = fig.add_subplot(111)
# Create bars and choose color
ax.bar([i for i in range(11)], torch.diag(finalcov)**0.5*1/theta_mean, color = 'tab:blue')
# title and axis names
ax.set_ylabel('inferred parameter uncertainty (std/mean)')
ax.set_xticks([i for i in range(11)])
ax.set_xticklabels(theta_names, rotation=45, ha='right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)



mu=torch.stack([rr[0] for rr in r])
err=torch.stack([torch.diag(rr[1])**0.5 for rr in r])

for i in range(11):
    plt.plot(mu[:,i],alpha=0.4)
    plt.fill_between([j for j in range(err.shape[0])],y1=mu[:,i]+err[:,i],y2=mu[:,i]-err[:,i],alpha=0.4)
plt.xlabel('number updates')
plt.ylabel('parameter value')

len(r)
r=r[3:]
loglls=[np.array(rr[2]) for rr in r]
logllmu=np.array([np.mean(rr) for rr in loglls])
logllstd=np.array([np.std(rr) for rr in loglls])
for i in range(11):
    plt.plot(logllmu,alpha=1)
    plt.fill_between([j for j in range(len(logllmu))],y1=logllmu+logllstd,y2=logllmu-logllstd,alpha=0.3)
plt.xlabel('number updates')
plt.ylabel('log likelihood')


# covariance
inds=[1,3,5,8,0,2,4,7,6,9,10]
with initiate_plot(5,5,300) as fig:
    ax=fig.add_subplot(1,1,1)
    cov=finalcov
    cov=torch.tensor(cov)
    im=plt.imshow(cov[inds].t()[inds].t(),cmap=plt.get_cmap('bwr'),
        vmin=-torch.max(cov),vmax=torch.max(cov))
    ax.set_title('covariance matrix', fontsize=20)
    x_pos = np.arange(len(theta_names))
    plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
    plt.xticks(x_pos, [theta_names[i] for i in inds],rotation=45,ha='right')
    add_colorbar(im)


# correlation matrix
b=torch.diag(torch.tensor(cov),0)
S=torch.diag(torch.sqrt(b))
Sinv=torch.inverse(S)
correlation=Sinv@cov@Sinv
with initiate_plot(5,5,300) as fig:
    ax=fig.add_subplot(1,1,1)
    im=ax.imshow(correlation[inds].t()[inds].t(),cmap=plt.get_cmap('bwr'),
        vmin=-torch.max(correlation),vmax=torch.max(correlation))
    ax.set_title('correlation matrix', fontsize=20)
    x_pos = np.arange(len(theta_names))
    plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
    plt.xticks(x_pos, [theta_names[i] for i in inds],rotation=45,ha='right')
    add_colorbar(im)



# eigen vector
ev, evector=torch.eig(torch.inverse(finalcov),eigenvectors=True)
ev=ev[:,0]
ev,esortinds=ev.sort(descending=True)
evector=evector[esortinds]
with initiate_plot(5,5,300) as fig:
    ax=fig.add_subplot(1,1,1)
    img=ax.imshow(evector[inds],cmap=plt.get_cmap('bwr'),
            vmin=-torch.max(evector),vmax=torch.max(evector))
    add_colorbar(img)
    ax.set_title('eigen vectors of Hessian')
    x_pos = np.arange(len(theta_names))
    plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')


with initiate_plot(5,1,300) as fig:
    ax=fig.add_subplot(1,1,1)
    x_pos = np.arange(len(theta_names))
    # Create bars and choose color
    ax.bar(x_pos, ev, color = 'blue')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yscale('log')
    # ax.set_ylim(min(plotdata),max(plotdata))
    ax.set_yticks([0.1,100,1e4])
    ax.set_xlabel('eigen values, log scale')
    plt.tick_params(axis='y', which='minor')
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    plt.gca().invert_yaxis()



# validate the inverse on simulation data.


# imports
# --------------------------------------------------------
import os
import pandas as pd
from numpy.lib.npyio import save
os.chdir(r"C:\Users\24455\iCloudDrive\misc\ffsb")
import numpy as np
from cmaes import CMA
import copy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pickle
import numpy as np
import torch
import heapq
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt
import time
from stable_baselines3 import TD3
torch.manual_seed(0)
from numpy import pi
from InverseFuncs import *
from monkey_functions import *
from FireflyEnv import ffacc_real
from Config import Config
# from cma_mpi_helper import run
import ray
from pathlib import Path
arg = Config()
import os
from timeit import default_timer as timer
from plot_ult import initiate_plot, process_inv, quickleg, quickspine, run_trial
# --------------------------------------------------------


# load agent and task
env=ffacc_real.FireFlyPaper(arg)
env.episode_len=50
phi=torch.tensor([[1],
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
agent_=TD3.load('trained_agent/paper.zip')
agent=agent_.actor.mu.cpu()

# create the dataset
nsub=10
n_normaltrial=50
n_perttrial=0
path=Path('Z:/simulation')
datafile=path/'simulation10group150normfixgr'

# individual per sub inverse
try:
    with open(datafile, 'rb') as f:
        allstates, allactions, alltasks, alltruth = pickle.load(f)
    print('use previous simulation trajectory')
except FileNotFoundError as err:
    raise Exception('run group inverse first!') from err


# decide if to continue
for isub in range(nsub):
    ntrialpersub=(len(allstates)//nsub)
    states, actions, tasks=allstates[isub*ntrialpersub: (isub+1)*ntrialpersub], allactions[isub*ntrialpersub: (isub+1)*ntrialpersub], alltasks[isub*ntrialpersub: (isub+1)*ntrialpersub]
    resfile=datafile.parent/'{}invsub{}'.format(isub,datafile.name)
    optimizer=None
    log=[]
    if resfile.is_file():
        print('continue on previous inverse...')
        with open(resfile, 'rb') as f:
            log = pickle.load(f)
        optimizer=log[-1][0]
    else:
        print('starting new inverse ...')

    # use localhost
    ray.init(log_to_driver=False,ignore_reinit_error=True)

    @ray.remote
    def getlogll(x):
        with torch.no_grad():
            return  monkeyloss_sim(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=1e-3,num_iteration=1, states=states, samples=3,gpu=False).item()

    # fresh start
    if not optimizer:
        # init condition, we want to at least cover some dynamic range
        init_theta=torch.tensor([[0.5],   
                [1.6],   
                [0.5],   
                [0.5],   
                [0.5],   
                [0.5],   
                [0.13],   
                [0.5],   
                [0.5],   
                [0.5],   
                [0.5]])
        init_theta=torch.mean(torch.stack(alltruth),axis=0)
        dim=init_theta.shape[0]
        init_cov=torch.diag(torch.ones(dim))*0.3
        cur_mu=init_theta.view(-1)
        cur_cov=init_cov
        optimizer = CMA(mean=np.array(cur_mu), sigma=0.5,population_size=14)
        optimizer.set_bounds(np.array([
        [0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.129, 0.01, 0.01, 0.01, 0.01],
        [2.,2,1,1,1,1,0.131,1,1,1,1]],dtype='float32').transpose())

        # optimizer.set_bounds(np.array([
        # [0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.129, groundtruth[7]-0.01, groundtruth[8]-0.01, groundtruth[9]-0.01, groundtruth[10]-0.01],
        # [1.,2,1,1,1,1,0.131,groundtruth[7]+0.01, groundtruth[8]+0.01, groundtruth[9]+0.01, groundtruth[10]+0.01]],dtype='float32').transpose())


    for generation in range(len(log),len(log)+40):
        start=timer()
        solutions = []
        xs=[]
        for _ in range(optimizer.population_size):
            x = optimizer.ask().astype('float32')
            xs.append(x)
        solutions=ray.get([getlogll.remote(p) for p in xs])
        meanlogll=np.mean(solutions)
        solutions=[[x,s] for x,s in zip(xs,solutions)]
        optimizer.tell(solutions)
        log.append([copy.deepcopy(optimizer), xs, solutions])
        # plt.imshow(optimizer._C)
        # plt.colorbar()
        # plt.show()
        with open(resfile, 'wb') as handle:
            pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('done, ',timer()-start)
        print("generation: ",generation,'-logll: ',meanlogll)
        # print('ground truth ',["{0:0.2f}".format(i) for i in groundtruth])
        print('cur estimatation ',["{0:0.2f}".format(i) for i in optimizer._mean])
        print('cur uncertainty ',["{0:0.2f}".format(i) for i in np.diag(optimizer._C)**0.5])
        if optimizer.should_stop():
            print('stop at {}th generation'.format(str(generation)))




# check res
# ---------------------------
path=Path('Z:/simulation')
datafile=path/'simulation10group150normfixgr'
subtheta,subcov,suberr=[],[],[]

# ground truth
if datafile.is_file():
    print('use previous simulation trajectory')
    with open(datafile, 'rb') as f:
        states, actions, tasks, alltruth = pickle.load(f)
ntrialpersub=(len(alltruth)//nsub)
# sub inverse
for isub in range(nsub):
    resfile=datafile.parent/'{}invsub{}'.format(isub,datafile.name)
    if resfile.is_file():
        theta,cov,err=process_inv(resfile,removegr=False)
        subtheta.append(theta)
        subcov.append(cov)
        err=torch.diag(cov)**0.5
        suberr.append(err)

# group inverse
resfile=datafile.parent/'invblief{}'.format(datafile.name)
grouptheta,groupcov,grouperr=process_inv(resfile,removegr=False)
grouperr=torch.diag(groupcov)**0.5

with initiate_plot(6,4,300) as f:
    ax=f.add_subplot(111)
    xs=np.array(list(range(11)))
    for t in alltruth:
        plt.scatter(xs,t,color='b',alpha=0.2,s=9,label='ground truth')
    for t,e in zip(subtheta, suberr):
        plt.errorbar(xs-0.1,t.view(-1),color='r',yerr=e,alpha=0.1,label='sub inverse',ls='none')
    plt.errorbar(xs+0.1,grouptheta.view(-1),yerr=grouperr,color='b',ls='none',label='group inverse')
    quickleg(ax)
    quickspine(ax)
    ax.set_xticks(list(range(11)))
    ax.set_xticklabels(theta_names_, rotation=45, ha='right')
    ax.set_ylabel('inferred param value')

plt.imshow(groupcov)
plt.imshow(cov)


# covariance heatmap -----------------------------------------------
correlation=correlation_from_covariance(cov)
with initiate_plot(5,5,300) as fig:
    ax=fig.add_subplot(1,1,1)
    im=ax.imshow(correlation[inds].t()[inds].t(),cmap=plt.get_cmap('bwr'),
        vmin=-torch.max(correlation),vmax=torch.max(correlation))
    ax.set_title('correlation matrix', fontsize=20)
    c=plt.colorbar(im,fraction=0.046, pad=0.04)
    c.set_label('correlation')
    x_pos = np.arange(len(theta_names))
    plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
    # plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
    plt.xticks(x_pos, [theta_names[i] for i in inds],rotation=45,ha='right')











import numpy as np
import torch
import heapq
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt

# misc
dim=2
selection_ratio=0.25
steps=20
samplesize=100
# ground truth
trueoffset=torch.ones(2)
x=torch.rand(2,10)
truecov=torch.tensor(np.cov(x)).float()*10

def gaussian(x,offset,cov,noise=0.1):
    noise=torch.zeros(1).uniform_(-1,1)*0.3
    return -1*(-0.5*(x-offset)@torch.inverse(cov)@(x-offset).t())+noise

sampledata=lambda x: gaussian(x,trueoffset,truecov)


# init condition
x=torch.zeros(dim)
cov=torch.diag(torch.ones(dim))*10


# sampling
def samplingparam(offset,cov,samplesize=10):
    dist=MultivariateNormal(loc=offset, covariance_matrix=cov)
    return dist.sample_n(samplesize)



# start solving
log=[]
offset=torch.zeros(2)
cov=torch.diag(torch.ones(2))
for i in range(steps):
    sampledparam=samplingparam(offset,cov,samplesize)
    res=[(sampledata(p),p) for p in sampledparam]
    res.sort(key=lambda a: a[0])
    nextgenparam=[p[1] for p in res[:int(selection_ratio*samplesize)]]
    cov=torch.tensor(np.cov(torch.stack(nextgenparam).t())).float()
    offset=torch.mean(torch.stack(nextgenparam),0)
    log.append([i,offset,cov])
    plt.scatter(sampledparam[:,0],sampledparam[:,1],color='tab:blue')
    plt.scatter(torch.stack(nextgenparam)[:,0],torch.stack(nextgenparam)[:,1],color='orange')
    plt.scatter(torch.stack(nextgenparam)[:,0],torch.stack(nextgenparam)[:,1],color='orange')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.show()

plt.plot(torch.stack([o[1] for o in log]))



# imports, for cma and botorch
import cma
import math
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood




X = torch.rand(20, 2) - 0.5
Y = (torch.sin(2 * math.pi * X[:, 0]) + torch.cos(2 * math.pi * X[:, 1])).unsqueeze(-1)
Y += 0.1 * torch.randn_like(Y)

gp = SingleTaskGP(X, Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)
from botorch.acquisition import UpperConfidenceBound

UCB = UpperConfidenceBound(gp, beta=0.1)
import numpy as np

# get initial condition for CMAES in numpy form
# note that CMAES expects a different shape (no explicit q-batch dimension)
x0 = np.random.rand(2)

# create the CMA-ES optimizer
es = cma.CMAEvolutionStrategy(
    x0=x0,
    sigma0=0.2,
    inopts={'bounds': [0, 1], "popsize": 100},
)

# speed up things by telling pytorch not to generate a compute graph in the background
with torch.no_grad():
    # Run the optimization loop using the ask/tell interface -- this uses 
    # PyCMA's default settings, see the PyCMA documentation for how to modify these
    while not es.stop():
        xs = es.ask()  # as for new points to evaluate

        # convert to Tensor for evaluating the acquisition function
        X = torch.tensor(xs, device=X.device, dtype=X.dtype)

        # evaluate the acquisition function (optimizer assumes we're minimizing)
        Y = - UCB(X.unsqueeze(-2))  # acquisition functions require an explicit q-batch dimension
        y = Y.view(-1).double().numpy()  # convert result to numpy array
        
        es.tell(xs, y)  # return the result to the optimizer

# convert result back to a torch tensor
best_x = torch.from_numpy(es.best.x).to(X)

best_x


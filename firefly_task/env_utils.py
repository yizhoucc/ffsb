import torch
# import torch.nn as nn
# from time import time
import math
import numpy as np
from numpy import pi

def bcov2vec(cov):
    vec=torch.stack([
    cov[0,0],
    cov[1,0],
    cov[1,1],
    cov[2,0],
    cov[2,1],
    cov[2,2],
    cov[3,3],
    cov[4,4]])
    return vec


def row_vector(vector):
    # input, np array or torch tensor, ouput to tensor row vector
    if type(vector) == torch.Tensor:
        return vector.view(1,-1)
    elif type(vector) == np.ndarray:
        return torch.Tensor(vector).view(1,-1)


def col_vector(vector):
    # input, np array or torch tensor, ouput to tensor col vector
    if type(vector) == torch.Tensor:
        return vector.view(-1,1)
    elif type(vector) == np.ndarray:
        return torch.Tensor(vector).view(-1,1)


def is_pos_def(x):
    """
    Check if the matrix is positive definite
    """
    if x.is_cuda:
        x = x.clone().detach().cpu()
        yes=torch.all(torch.eig(x)[0] >= torch.tensor([0.]))
        if not yes:
            print(torch.eig(x))
    else:
        # x = x.detach().numpy()
        # yes=np.all(np.linalg.eigvalsh(x) >= 0)
        yes=torch.all(torch.eig(x)[0] >= torch.tensor([0.]))
        if not yes:
            print(torch.eig(x))

    return yes


def tril_mask(size):
    """
    Returns a lower triangular mask
    (Used to select lower triangular elements)
    """
    mask = torch.tril(torch.ones(size, size, dtype=torch.uint8)) # ByteTensor
    return mask


def vectorLowerCholesky(P):
    """
    Performs the lower cholesky decomposition and returns vectorized output
    P = L L.t()
    """
    L = torch.cholesky(P, upper=False)
    mask = tril_mask(P.size(0))
    return torch.masked_select(L, mask > 0)


def inverseCholesky(vecL):
    """
    Performs the inverse operation to lower cholesky decomposition
    and converts vectorized lower cholesky to matrix P
    P = L L.t()
    """
    size = int(np.sqrt(2 * vecL.size(0)))
    L = torch.zeros(size, size)
    mask = tril_mask(size)
    L[mask == 1] = vecL
    P = L.mm(L.t())
    return P


def range_angle(ang):
    """
    Adjusts the range of angle from -pi to pi
    """
    ang = torch.remainder(ang, 2*pi)
    ang = ang if ang < pi else (ang -2*pi)
    return ang


def pos_init(box=2.):
    """
    Initialize position (polar) coordinates and relative angle of monkey
    """
    #r = torch.rand(1) * box
    r = torch.sqrt(torch.rand(1)+0.25) * box
    ang = torch.zeros(1).uniform_(-pi, pi)
    #ang = torch.ones(1)*pi/2
    #pos = torch.cat((r, ang)) # polar coordinates
    rel_ang = torch.zeros(1).uniform_(-pi/4, pi/4)
    return (r, ang, rel_ang)


def ellipse(mu, cov, n=100, conf_int=5.991, torch=False):
    """
    Returns points on the confidence region boundary for 2D Gaussian
    distribution
    (Used to plot while rendering)
    """
    if torch:
        mu = mu.detach().numpy()
        cov = cov.detach().numpy()
    w, v = np.linalg.eigh(cov)
    max_eig = v[:, -1]
    phi = math.atan2(max_eig[1], max_eig[0])
    phi %= 2 * pi
    chi2_val = np.sqrt(conf_int) # 95% confidence interval
    a =  chi2_val * np.sqrt(w[1])
    b =  chi2_val * np.sqrt(w[0])
    #theta = np.arange(0, 2 * pi, 0.01)
    theta = np.linspace(0, 2 * pi, n)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    R = np.array([[np.cos(phi), np.sin(phi)],
                  [-np.sin(phi), np.cos(phi)]])
    X = np.array([x,y]).T
    X = X.dot(R)
    x = X[:,0] + mu[0]
    y = X[:,1] + mu[1]
    return x, y


def sample_exp(min, max, scale = np.e):
    """sample a random number with exponetial distribution
    the number should be in a range [min, max]
    should be min, max >= 0
    """
    temp = min -100
    while temp < min or temp > max:
        temp = np.random.exponential(scale=scale)
    return temp


def norm_parameter(parameter,param_range, given_range=[0.001,0.999]):
    'normalize the paramter range to a range'
    k=(max(given_range)-min(given_range))/(max(param_range)-min(param_range))
    c=k*min(param_range)-min(given_range)
    parameter=parameter*k-c
    return parameter


def denorm_parameter(parameter,param_range, given_range=[0.001,0.999]):
    'denormalize the paramter range to a range'
    k=(max(given_range)-min(given_range))/(max(param_range)-min(param_range))
    c=k*min(param_range)-min(given_range)
    parameter=(parameter+c)/k
    return parameter


def inverse_sigmoid(parameter):
    return torch.log(parameter/(1-parameter))


def placeholder_func(input):
    return input


# for acc control
def get_a(vm, dt=0.1, T=7, target_d=3, control_gain=1):
    # return max a for given vm
    lowera=0
    uppera=0.99
    a=(lowera+uppera)/2
    
    while uppera-lowera>0.02:
        d=get_d(a, vm=vm, dt=dt, T=T,control_gain=control_gain)
        if d>target_d+0.05:
            lowera=a
            a=(lowera+uppera)/2
        elif d<target_d-0.05:
            uppera=a
            a=(lowera+uppera)/2
        else:
            return a
        # print(a, lowera, uppera)
    return a


def get_d(a, vm, dt=0.1, T=7,control_gain=1):
    # return max d given a and vm
    s=get_s(a, vm, dt=dt, T=T,control_gain=control_gain)
    b=vm*(1-a)/control_gain
    d=0
    v=0
    for i in range(s):
        v=v*a+b*control_gain
        d=d+v*dt
    for i in range(int(T/dt-s)):
        v=v*a-b*control_gain
        d=d+v*dt
    return d


def get_s(a, vm, dt=0.1,T=7,control_gain=1):
    # return switch time given a and vm
    b=vm*(1-a)/control_gain
    totalt=T/dt
    s=int(totalt/2)
    lowers=0
    uppers=totalt
    while uppers-lowers>1:
        v=0
        for i in range(int(s)):
            v=v*a+b*control_gain
        for i in range(int(totalt-s)):
            v=v*a-b*control_gain
        if v>0.01:
            uppers=s
            s=round((lowers+uppers)/2)
        elif v<-0.01:
            lowers=s
            s=round((lowers+uppers)/2)
        else:
            return s
        # print(s)
    return int(s)
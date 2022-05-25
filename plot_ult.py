

import enum
import imp
from pathlib import Path
from tkinter import PhotoImage
import numpy as np
from matplotlib.ticker import MaxNLocator
import scipy.stats
from numpy import pi
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1
from matplotlib.patches import Ellipse, Circle
from InverseFuncs import *
import heapq
from collections import namedtuple
import torch
import os
os.chdir('C:\\Users\\24455\\iCloudDrive\\misc\\ffsb')
import numpy as np
from scipy.stats import tstd
from matplotlib import pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn import random_projection
# from InverseFuncs import trajectory, getLoss
import sys
# from stable_baselines import DDPG,TD3
from FireflyEnv import firefly_action_cost, ffenv_new_cord,firefly_accac, ffac_1d
from Config import Config
arg = Config()
# import policy_torch
from plot_ult import *
import TD3_torch
from importlib import reload
from inspect import EndOfBlock
from pickle import HIGHEST_PROTOCOL
import multiprocessing
from matplotlib.collections import LineCollection
from numpy.core.numeric import NaN
from scipy.ndimage.measurements import label
from torch.nn import parameter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import torch
import warnings
import random
from astropy.convolution import convolve
from numpy import pi
# from numba import njit
from contextlib import redirect_stdout, redirect_stderr, contextmanager, ExitStack
import os
import matplotlib as mpl
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict
import matplotlib.pylab as pl
from plot_ult import *
from InverseFuncs import *
import warnings
warnings.filterwarnings('ignore')
from copy import copy
import time
import random
from stable_baselines3 import SAC,PPO
seed=0
random.seed(seed)
import torch
torch.manual_seed(seed)
import numpy as np
np.random.seed(int(seed))
from numpy import pi
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from InverseFuncs import trajectory, getLoss, reset_theta, theta_range,reset_theta_log, single_theta_inverse
from FireflyEnv import ffacc_real
import TD3_torch
from monkey_functions import *
from Config import Config
arg = Config()

cmaps = OrderedDict()
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
global color_settings
color_settings={
    'v':'tab:blue',
    'w':'orange',
    'cov':'#8200FF', # purple, save as belief, just use low alpha
    'b':'#8200FF', # purple
    's':'#0000FF', # blue
    'o':'#FF0000', # red
    'a':'#008100', # green, slightly dark
    'goal':'#4D4D4D', # dark grey
    'hidden':'#636363',
    'model':'#FFBC00', # orange shifting to green, earth
    '':'',
    'pair':[[1,0,0],[0,0,1]],

}

global theta_names
theta_names = [ 'pro gain v',
                'pro gain w',
                'pro noise v',
                'pro noise w',
                'obs noise v',
                'obs noise w',
                'action cost v',      
                'action cost w',      
                'init uncertainty x',      
                'init uncertainty y',      
                ]

global theta_mean
theta_mean=[
    0.4,
    1.57,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5    
]

global theta_names_
theta_names_ = [ 'pro gain v',
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

global theta_mean_
theta_mean_=[
    0.4,
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
]

global parameter_range
parameter_range=[
    [0.4,0.6],
    [1.2,1.8],
    [1e-3,1],
    [1e-3,1],
    [1e-3,1],
    [1e-3,1],
    [1e-3,0.5],
    [1e-3,1],
    [1e-3,1],
    [1e-3,1],
    [1e-3,1],
]

global phi 
phi=torch.tensor([[0.5],
        [pi/2],
        [0.001],
        [0.001],
        [0.001],
        [0.001],
        [0.13],
        [0.5],
        [0.5],
        [0.001],
        [0.001],
])


def hex2rgb(hexstr):
    if hexstr[0] =='#':
        hexstr=hexstr.lstrip('#')
    return [int(hexstr[i:i+2], 16)/255 for i in (0, 2, 4)]


def colorshift(mu, direction, amount, n):
    mu=np.array(mu)
    direction=np.array(direction)
    deltas=np.linspace(np.clip(mu-direction*amount, 0,1), np.clip(mu+direction*amount,0,1), n)
    res=[ d.tolist() for d in deltas ]
    # for d in deltas:
    #     res.append((mu+d).tolist())
    return res
    


def mean_confidence_interval(data, confidence=0.95):
    # ci of the mean
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# illustrate perterb trial
def pertexample():
    with initiate_plot(3,3,300) as fig:
        ax=fig.add_subplot()
        ax.plot(df.iloc[0].pos_v,color=color_settings['v'],label='real v')
        ax.plot([i*200 for i in df.iloc[0].action_v ],'--',color=color_settings['v'],label='v control')
        ax.plot(df.iloc[0].pos_w ,color=color_settings['w'],label='real w')
        ax.plot([i*200 for i in df.iloc[0].action_w ],'--',color=color_settings['w'],label='w control')
        ax.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time [dt]')
        ax.set_ylabel('velocity [cm/s and degree/s]')

# illustrate density
def densityexample():
    densities=[0.0001, 0.0005, 0.001, 0.005]
    marker = itertools.cycle(('v', '^', '<', '>')) 
    with initiate_plot(8,2,300) as fig:
        for i,density in enumerate(densities):
            ax=fig.add_subplot(1,4,i+1)    
            ax.set_xticks([]);ax.set_yticks([])
            x=torch.zeros(int(10000*1*density)).uniform_()
            y=torch.zeros(int(10000*1*density)).uniform_()
            for xx,yy in zip(x,y):
                plt.plot(xx,yy, markersize=3,marker = next(marker), linestyle='',color='k')
            ax.set_xlabel('density={}'.format(density))

          
def plot_belief_trajectory(model, env, title, ax=None, **kwargs):
    if ax is None:
        f1=plt.figure(figsize=(10,10))
        ax = plt.gca()
        ax.set_ylim(-1.5,1.5)
        ax.set_xlim(-1.5,1.5)


def plot_belief(env,title='title', alpha_factor=1,**kwargs):
    'plot the belief, show the ellipse with goal circle, heading'
    f1=plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.set_ylim(-1.5,1.5)
    ax.set_xlim(-1.5,1.5)
    pos=env.b[:2,0].detach()
    cov=env.P[:2,:2].detach()
    # plt.plot(pos[0],pos[1],'o')
    # print('test',title)
    if kwargs.get('title'):
        print('test',kwargs.get('title'))
        title=kwargs.get('title')
    plt.title(title)
    # real position red dot
    plt.plot(env.s.detach()[0,0],env.s.detach()[1,0],'ro')
    # green line showing mean belief to goal
    plt.plot([pos.detach()[0],pos.detach()[0]+env.decision_info.detach()[0,0]*np.cos(env.b.detach()[2,0]+env.decision_info.detach()[0,1]) ],
    [pos.detach()[1],pos.detach()[1]+env.decision_info.detach()[0,0]*np.sin(env.b.detach()[2,0]+env.decision_info.detach()[0,1])],'g')
    # red arrow of belief heading direction
    plt.quiver(pos.detach()[0], pos.detach()[1],np.cos(env.b.detach()[2,0].item()),np.sin(env.b.detach()[2,0].item()), 
            color='r', scale=10, alpha=0.5*alpha_factor)
    # blue ellipse of belief
    plot_cov_ellipse(cov, pos, alpha_factor=1,nstd=2,ax=ax)
    # yellow goal
    plot_circle(np.eye(2)*env.phi[-1,0].item(),[env.goalx,env.goaly],ax=ax,color='y')
    
    return f1


def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos=[0,0], nstd=2, color=None, ax=None,alpha=1,edgecolor=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        figure=plt.figure(figsize=(10,10))
        ax = plt.gca()
        ax.set_ylim(-1.5,1.5)
        ax.set_xlim(-1.5,1.5)


    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, edgecolor='none',height=height, angle=theta, **kwargs)
    if color is not None:
        ellip.set_color(color)
    if edgecolor:
        ellip.set_color('none')
        ellip.set_edgecolor(edgecolor)
    ellip.set_alpha(alpha)
    ax.add_artist(ellip)
    return ellip


def plot_circle(cov, pos, color=None, ax=None,alpha=1, **kwargs):
    'plot a circle'
    if ax is None:
        figure=plt.figure(figsize=(10,10))
        ax = plt.gca()
        ax.set_ylim(-1.5,1.5)
        ax.set_xlim(-1.5,1.5)
    assert cov[0,0]==cov[1,1]
    r=cov[0,0]
    c = Circle(pos,r)
    if color is not None:
        c.set_color(color)
    c.set_alpha(alpha)
    ax.add_artist(c)
    return c


def overlap_mc(r, cov, mu, nsamples=1000):
    'return the overlaping of a circle and ellipse using mc'
    # xrange=[-cov[0,0],cov[0,0]]
    # yrange=[-cov[1,1],cov[1,1]]
    # xrange=[mu[0]-r*1.1,mu[0]+r*1.1]
    # yrange=[mu[1]-r*1.1,mu[1]+r*1.1]

    check=[]
    xs, ys = np.random.multivariate_normal(-mu, cov, nsamples).T
    # plot_overlap(r,cov,mu,title=None)
    for i in range(nsamples):
        # plt.plot(xs[i],ys[i],'.')
        if (xs[i])**2+(ys[i])**2<=r**2:                
            check.append(1)
        else:
            check.append(0)
    P=np.mean(check)
    return P


def plot_overlap(r,cov,mu,title=None):
    'plot the overlap between a circle and ellipse with cov'
    f1=plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.set_ylim(-1.5,1.5)
    ax.set_xlim(-1.5,1.5)
    if title is not None:
        ax.title.set_text(str(title))
    plot_cov_ellipse(cov,[0,0],nstd=1,ax=ax)
    plot_circle(np.eye(2)*r,mu,ax=ax,color='r')
    return f1


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
    

def inverseCholesky(vecL):
    """
    Performs the inverse operation to lower cholesky decomposition
    and converts vectorized lower cholesky to matrix P
    P = L L.t()
    """
    size = int(np.sqrt(2 * len(vecL)))
    L = np.zeros((size, size))
    mask = np.tril(np.ones((size, size)))
    L[mask == 1] = vecL
    P = L@(L.transpose())
    return P


def policy_surface(belief,torch_model):
    # manipulate distance, reltive angle in belief and plot
    r_range=[0.2,1.0]
    r_ticks=0.01
    r_labels=[r_range[0]]
    a_range=[-pi/4, pi/4]
    a_ticks=0.05
    a_labels=[a_range[0]]
    while r_labels[-1]+r_ticks<=r_range[-1]:
        r_labels.append(r_labels[-1]+r_ticks)
    while a_labels[-1]+a_ticks<=a_range[-1]:
        a_labels.append(a_labels[-1]+a_ticks)
    policy_data_v=np.zeros((len(r_labels),len(a_labels)))
    policy_data_w=np.zeros((len(r_labels),len(a_labels)))
    for ri in range(len(r_labels)):
        for ai in range(len(a_labels)):
            belief[0]=r_labels[ri]
            belief[1]=a_labels[ai]
            policy_data_v[ri,ai]=baselines_mlp_model.predict(belief.view(1,-1).tolist())[0][0].item()
            policy_data_w[ri,ai]=baselines_mlp_model.predict(belief.view(1,-1).tolist())[0][1].item()

    plt.figure(0,figsize=(8,8))
    plt.suptitle('Policy surface of velocity',fontsize=24)
    plt.ylabel('relative distance',fontsize=15)
    plt.xlabel('relative angle',fontsize=15)
    plt.imshow(policy_data_v,origin='lower',extent=[a_labels[0],a_labels[-1],r_labels[0],r_labels[-1]])
    plt.savefig('policy surface v {}.png'.format(name_index))

    plt.figure(1,figsize=(8,8))
    plt.suptitle('Policy surface of angular velocity')
    plt.figure(1,figsize=(8,8))
    plt.suptitle('Policy surface of velocity',fontsize=24)
    plt.ylabel('relative distance',fontsize=15)
    plt.xlabel('relative angle',fontsize=15)
    plt.imshow(policy_data_w,origin='lower',extent=[r_labels[0],r_labels[-1],a_labels[0],a_labels[-1]])

    plt.savefig('policy surface w {}.png'.format(name_index))


def policy_range(r_range=None,r_ticks=None,a_range=None,a_ticks=None):

    r_range=[0.2,1.0] if r_range is None else r_range
    r_ticks=0.01 if r_ticks is None else r_ticks
    r_labels=[r_range[0]] 

    a_range=[-pi/4, pi/4] if a_range is None else a_range
    a_ticks=0.05 if a_ticks is None else a_ticks
    a_labels=[a_range[0]]

    while r_labels[-1]+r_ticks<=r_range[-1]:
        r_labels.append(r_labels[-1]+r_ticks)
    while a_labels[-1]+a_ticks<=a_range[-1]:
        a_labels.append(a_labels[-1]+a_ticks)
    policy_data_v=np.zeros((len(r_labels),len(a_labels)))
    policy_data_w=np.zeros((len(r_labels),len(a_labels)))

    return r_labels, a_labels, policy_data_v, policy_data_w

    # imports and define agent name


def plot_path_ddpg(modelname,env,num_episode=None):
    
    from stable_baselines import DDPG

    num_episode=20 if num_episode is None else num_episode

    agent = DDPG.load(modelname,env=env)

    # create saving vars
    all_ep=[]
    # for ecah episode,
    for i in range(num_episode):
        ep_data={}
        ep_statex=[]
        ep_statey=[]
        ep_belifx=[]
        ep_belify=[]
        # get goal position at start
        decisioninfo=env.reset()
        goalx=env.goalx
        goaly=env.goaly
        ep_data['goalx']=goalx
        ep_data['goaly']=goaly
        # log the actions raw, v and w
        while not env.stop:
            action,_=agent.predict(decisioninfo)
            decisioninfo,_,_,_=env.step(action)
            ep_statex.append(env.s[0,0])
            ep_statey.append(env.s[0,1])
            ep_belifx.append(env.b[0,0])
            ep_belify.append(env.b[0,1])
        ep_data['x']=ep_statex
        ep_data['y']=ep_statey
        ep_data['bx']=ep_belifx
        ep_data['by']=ep_belify
        ep_data['goalx']=env.goalx
        ep_data['goaly']=env.goaly
        ep_data['theta']=env.theta.tolist()
        # save episode data dict to all data
        all_ep.append(ep_data)

    for i in range(num_episode):
        plt.figure
        ep_xt=all_ep[i]['x']
        ep_yt=all_ep[i]['y']
        plt.title(str(['{:.2f}'.format(x) for x in all_ep[i]['theta']]))
        plt.plot(ep_xt,ep_yt,'r-')
        plt.plot(all_ep[i]['bx'],all_ep[i]['by'],'b-')
        # plt.scatter(all_ep[i]['goalx'],all_ep[i]['goaly'])

        circle = np.linspace(0, 2*np.pi, 100)
        r = all_ep[i]['theta'][-1]
        x = r*np.cos(circle)+all_ep[i]['goalx'].item()
        y = r*np.sin(circle)+all_ep[i]['goaly'].item()
        plt.plot(x,y)

        plt.savefig('path.png')


def sort_evals_descending(evals, evectors):
  """
  Sorts eigenvalues and eigenvectors in decreasing order. Also aligns first two
  eigenvectors to be in first two quadrants (if 2D).

  Args:
    evals (numpy array of floats)    : Vector of eigenvalues
    evectors (numpy array of floats) : Corresponding matrix of eigenvectors
                                        each column corresponds to a different
                                        eigenvalue

  Returns:
    (numpy array of floats)          : Vector of eigenvalues after sorting
    (numpy array of floats)          : Matrix of eigenvectors after sorting
  """

  index = np.flip(np.argsort(evals))
  evals = evals[index]
  evectors = evectors[:, index]
  if evals.shape[0] == 2:
    if np.arccos(np.matmul(evectors[:, 0],
                           1 / np.sqrt(2) * np.array([1, 1]))) > np.pi / 2:
      evectors[:, 0] = -evectors[:, 0]
    if np.arccos(np.matmul(evectors[:, 1],
                           1 / np.sqrt(2) * np.array([-1, 1]))) > np.pi / 2:
      evectors[:, 1] = -evectors[:, 1]
  return evals, evectors


def get_sample_cov_matrix(X):
  """
    Returns the sample covariance matrix of data X

    Args:
      X (numpy array of floats) : Data matrix each column corresponds to a
                                  different random variable

    Returns:
      (numpy array of floats)   : Covariance matrix
  """

  # Subtract the mean of X
  X = X - np.mean(X, 0)
  # Calculate the covariance matrix (hint: use np.matmul)
  cov_matrix = cov_matrix = 1 / X.shape[0] * np.matmul(X.T, X)

  return cov_matrix


def plot_data_new_basis(Y):
  """
  Plots bivariate data after transformation to new bases. Similar to plot_data
  but with colors corresponding to projections onto basis 1 (red) and
  basis 2 (blue).
  The title indicates the sample correlation calculated from the data.

  Note that samples are re-sorted in ascending order for the first random
  variable.

  Args:
    Y (numpy array of floats) : Data matrix in new basis each column
                                corresponds to a different random variable

  Returns:
    Nothing.
  """

  fig = plt.figure(figsize=[8, 4])
  gs = fig.add_gridspec(2, 2)
  ax1 = fig.add_subplot(gs[0, 0])
  ax1.plot(Y[:, 0], 'r')
  plt.ylabel('Projection \n basis vector 1')
  ax2 = fig.add_subplot(gs[1, 0])
  ax2.plot(Y[:, 1], 'b')
  plt.xlabel('Projection \n basis vector 1')
  plt.ylabel('Projection \n basis vector 2')
  ax3 = fig.add_subplot(gs[:, 1])
  ax3.plot(Y[:, 0], Y[:, 1], '.', color=[.5, .5, .5])
  ax3.axis('equal')
  plt.xlabel('Projection basis vector 1')
  plt.ylabel('Projection basis vector 2')
  plt.title('Sample corr: {:.1f}'.format(np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]))
  plt.show()


def pca(X):
  """
    Performs PCA on multivariate data.

    Args:
      X (numpy array of floats) : Data matrix each column corresponds to a
                                  different random variable

    Returns:
      (numpy array of floats)   : Data projected onto the new basis
      (numpy array of floats)   : Vector of eigenvalues
      (numpy array of floats)   : Corresponding matrix of eigenvectors

  """

  # Subtract the mean of X
  X = X - np.mean(X, axis=0)
  # Calculate the sample covariance matrix
  cov_matrix = get_sample_cov_matrix(X)
  # Calculate the eigenvalues and eigenvectors
  evals, evectors = np.linalg.eigh(cov_matrix)
  # Sort the eigenvalues in descending order
  evals, evectors = sort_evals_descending(evals, evectors)
  # Project the data onto the new eigenvector basis
  score = np.matmul(X, evectors)

  return score, evectors, evals


def random_projection(X):
  X = X - np.mean(X, axis=0)
  projection_matrix=np.random.random((9,2))
  Y=X@projection_matrix

  newX== X@transformer.components_.transpose()
  X.shape
  projection_matrix.shape
  Y.shape
  X=background_data
  plt.imshow(X)
  plt.imshow(newX)
  plt.imshow(Y)
  plt.imshow(score)
  plt.imshow(score@evectors.transpose())
  plt.imshow(newX@transformer.components_.transpose())
  plt.imshow(X@transformer.components_.transpose())
  plt.imshow(newX@np.linalg.inv( transformer.components_.transpose()))
  Y=newX@np.linalg.inv( transformer.components_.transpose())-X

  X = np.random.rand(20, 9)
  transformer = random_projection.GaussianRandomProjection(n_components=9)    
  transformer.components_
  newX= transformer.fit_transform(X)


def column_feature_data(list_data):
    '''
    input: list of data, index as feature
    output: np array of data, column as feature, row as entry
    '''
    total_rows=len(list_data)
    total_cols=len(list_data[0])
    data_matrix=np.zeros((total_rows,total_cols))
    for i, data in enumerate(list_data):
        data_matrix[i,:]=[d[0] for d in data]
    return data_matrix


def plot_inverse_trajectory(theta_trajectory,true_theta, env,agent,
                      phi=None, method='PCA', background_data=None, 
                      background_contour=False,number_pixels=10,
                      background_look='contour',
                      ax=None, loss_sample_size=100, H=False,loss_function=None):
    '''    
    plot the inverse trajectory in 2d pc space
    -----------------------------
    input:
    theta trajectory: list of list(theta)
    method: PCA or other projections
    -----------------------------
    output:
    figure
    '''
    # plot trajectory
    if ax is None:
        fig = plt.figure(figsize=[8, 8])
        ax = fig.add_subplot()

    # reshape the data into colume as features
    data_matrix=column_feature_data(theta_trajectory)
    if method=='PCA':
      try:
        score, evectors, evals = pca(data_matrix)
      except np.linalg.LinAlgError:
        score, evectors, evals = pca(data_matrix)
      # note, the evectors are arranged in cols. one col, one evector.
      plt.xlabel('Projection basis vector 1')
      plt.ylabel('Projection basis vector 2')
      plt.title('Inverse for theta estimation in a PCs space. Sample corr: {:.1f}'.format(np.corrcoef(score[:, 0], score[:, 1])[0, 1]))

      # plot true theta
      mu=np.mean(data_matrix,0)
      if type(true_theta)==list:
        true_theta=np.array(true_theta).reshape(-1)
        true_theta_pc=(true_theta-mu)@evectors
      elif type(true_theta)==torch.nn.parameter.Parameter:
        true_theta_pc=(true_theta.detach().numpy().reshape(-1)-mu)@evectors
      else:
        true_theta_pc=(true_theta-mu)@evectors

      ax.scatter(true_theta_pc[0],true_theta_pc[1],marker='o',c='',edgecolors='r',zorder=10)
      # plot latest theta
      ax.scatter(score[-1,0],score[-1,1],marker='o',c='',edgecolors='b',zorder=9)

      # plot theta inverse trajectory
      row_cursor=0
      while row_cursor<score.shape[0]-1:
          row_cursor+=1
          # plot point
          ax.plot(score[row_cursor, 0], score[row_cursor, 1], '.', color=[.5, .5, .5])
          # plot arrow
          # fig = plt.figure(figsize=[8, 8])
          # ax1 = fig.add_subplot()
          # ax1.set_xlim([-1,1])
          # ax1.set_ylim([-1,1])
          ax.quiver(score[row_cursor-1, 0], score[row_cursor-1, 1],
                  score[row_cursor, 0]-score[row_cursor-1, 0], score[row_cursor, 1]-score[row_cursor-1, 1],
                  angles='xy',color='g',scale=1, scale_units='xy')
      ax.axis('equal')

      # plot hessian
      if H:
        H=compute_H(env, agent, theta_trajectory[-1], true_theta.reshape(-1,1), phi, trajectory_data=None,H_dim=len(true_theta), num_episodes=loss_sample_size)
        cov=theta_cov(H)
        cov_pc=evectors[:,:2].transpose()@np.array(cov)@evectors[:,:2]
        plot_cov_ellipse(cov_pc,pos=score[-1,:2],alpha_factor=0.5,ax=ax)
        stderr=np.sqrt(np.diag(cov)).tolist()
        ax.title.set_text('stderr: {}'.format(str(['{:.2f}'.format(x) for x in stderr])))


      # plot log likelihood contour
      loss_function=compute_loss_wrapped(env, agent, true_theta.reshape(-1,1), np.array(phi).reshape(-1,1), trajectory_data=None, num_episodes=1000) if loss_function is None else loss_function
      current_xrange=list(ax.get_xlim())
      current_xrange[0]-=0.5
      current_xrange[1]+=0.5
      current_yrange=list(ax.get_ylim())
      current_yrange[0]-=0.5
      current_yrange[1]+=0.5
      xyrange=[current_xrange,current_yrange]
      # ax1.contourf(X,Y,background_data)
      if background_contour:
        background_data=plot_background(ax, xyrange,mu, evectors, loss_function, number_pixels=number_pixels,look=background_look) if background_data is None else background_data

    return background_data


def inverse_trajectory_monkey(theta_trajectory,
                    env=None,
                    agent=None,
                    phi=None, 
                    background_data=None, 
                    background_contour=False,
                    number_pixels=10,
                    background_look='contour',
                    ax=None, 
                    loss_sample_size=100, 
                    H=None,
                    loss_function=None,
                    **kwargs):
    '''    
        plot the inverse trajectory in 2d pc space
        -----------------------------
        input:
        theta trajectory: list of list(theta)
        method: PCA or other projections
        -----------------------------
        output:
        background contour array and figure
    '''
    # plot trajectory
    fig = plt.figure(figsize=[8, 8])
    ax = fig.add_subplot()
    data_matrix=column_feature_data(theta_trajectory)

    try:
        score, evectors, evals = pca(data_matrix)
    except np.linalg.LinAlgError:
        score, evectors, evals = pca(data_matrix)
        # note, the evectors are arranged in cols. one col, one evector.
        plt.xlabel('Projection basis vector 1')
        plt.ylabel('Projection basis vector 2')
        plt.title('Inverse for theta estimation in a PCs space. Sample corr: {:.1f}'.format(np.corrcoef(score[:, 0], score[:, 1])[0, 1]))

    # plot theta inverse trajectory
    row_cursor=0
    while row_cursor<score.shape[0]-1:
        row_cursor+=1
        # plot point
        # ax.plot(score[row_cursor, 0], score[row_cursor, 1], '.', color=[.5, .5, .5])
        # plot arrow
        # fig = plt.figure(figsize=[8, 8])
        # ax1 = fig.add_subplot()
        # ax1.set_xlim([-1,1])
        # ax1.set_ylim([-1,1])
        ax.plot(score[row_cursor-1:row_cursor+1,0],
                score[row_cursor-1:row_cursor+1,1],
                '-',
                linewidth=0.1,
                color='g')
        if row_cursor%20==0 or row_cursor==1:
            ax.quiver(score[row_cursor-1, 0], score[row_cursor-1, 1],
                    score[row_cursor, 0]-score[row_cursor-1, 0], score[row_cursor, 1]-score[row_cursor-1, 1],
                    angles='xy',color='g',scale=0.2,width=1e-2, scale_units='xy')
    ax.scatter(score[row_cursor, 0], score[row_cursor, 1], marker=(5, 1), s=200, color=[1, .5, .5])
    # ax.axis('equal')

    # plot hessian
    if H is not None:
        cov=theta_cov(H)
        cov_pc=evectors[:,:2].transpose()@np.array(cov)@evectors[:,:2]
        plot_cov_ellipse(cov_pc,pos=score[-1,:2],alpha_factor=0.5,ax=ax)
        stderr=np.sqrt(np.diag(cov)).tolist()
        ax.title.set_text('stderr: {}'.format(str(['{:.2f}'.format(x) for x in stderr])))

    # plot log likelihood contour
    loss_function=compute_loss_wrapped(env, agent, true_theta.reshape(-1,1), np.array(phi).reshape(-1,1), trajectory_data=None, num_episodes=1000) if loss_function is None else loss_function
    current_xrange=list(ax.get_xlim())
    current_xrange[0]-=0.5
    current_xrange[1]+=0.5
    current_yrange=list(ax.get_ylim())
    current_yrange[0]-=0.5
    current_yrange[1]+=0.5
    xyrange=[current_xrange,current_yrange]
    # ax1.contourf(X,Y,background_data)
    if background_contour:
        background_data=plot_background(ax, xyrange,mu, evectors, loss_function, number_pixels=number_pixels,look=background_look) if background_data is None else background_data

    return background_data


def load_inverse_data(filename):
  'load the data pickle file, return the data dict'
  sys.path.insert(0, './inverse_data/')
  if filename[-4:]=='.pkl':
      data=torch.load('inverse_data/{}'.format(filename))
  else:
      data=torch.load('inverse_data/{}.pkl'.format(filename))
  return data


def run_inverse(data=None,theta=None,filename=None):
  import os
  import warnings
  warnings.filterwarnings('ignore')
  from copy import copy
  import time
  import random
  seed=time.time().as_integer_ratio()[0]
  seed=0
  random.seed(seed)
  import torch
  torch.manual_seed(seed)
  import numpy as np
  np.random.seed(int(seed))
  from numpy import pi
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  # -----------invser functions-------------
  from InverseFuncs import trajectory, getLoss, reset_theta, theta_range,reset_theta_log, single_inverse
  # ---------loading env and agent----------
  from stable_baselines import DDPG,TD3
  from FireflyEnv import ffenv_new_cord
  from Config import Config
  arg = Config()
  DISCOUNT_FACTOR = 0.99
  arg.NUM_SAMPLES=2
  arg.NUM_EP = 1000
  arg.NUM_IT = 2 # number of iteration for gradient descent
  arg.NUM_thetas = 1
  arg.ADAM_LR = 0.007
  arg.LR_STEP = 2
  arg.LR_STOP = 50
  arg.lr_gamma = 0.95
  arg.PI_STD=1
  arg.goal_radius_range=[0.05,0.2]


  # agent convert to torch model
  import policy_torch
  baselines_mlp_model = TD3.load('trained_agent//TD_95gamma_mc_smallgoal_500000_9_24_1_6.zip')
  agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[128,128])

  # loading enviorment, same as training
  env=ffenv_new_cord.FireflyAgentCenter(arg)
  env.agent_knows_phi=False

  true_theta_log = []
  true_loss_log = []
  true_loss_act_log = []
  true_loss_obs_log = []
  final_theta_log = []
  stderr_log = []
  result_log = []
  number_update=100
  if data is None:
    save_dict={'theta_estimations':[]}
  else:
    save_dict=data


  # use serval theta to inverse
  for num_thetas in range(arg.NUM_thetas):

      # make sure phi and true theta stay the same 
      true_theta = torch.Tensor(data['true_theta'])
      env.presist_phi=True
      env.reset(phi=true_theta,theta=true_theta) # here we first testing teacher truetheta=phi case
      theta=torch.Tensor(data['theta_estimations'][0])
      phi=torch.Tensor(data['phi'])
  

      save_dict['true_theta']=true_theta.data.clone().tolist()
      save_dict['phi']=true_theta.data.clone().tolist()
      save_dict['inital_theta']=theta.data.clone().tolist()


      for num_update in range(number_update):
          states, actions, tasks = trajectory(
              agent, phi, true_theta, env, arg.NUM_EP)
              
          result = single_theta_inverse(true_theta, phi, arg, env, agent, states, actions, tasks, filename, num_thetas, initial_theta=theta)
          
          save_dict['theta_estimations'].append(result.tolist())
          if filename is None:
            savename=('inverse_data/' + filename + "EP" + str(arg.NUM_EP) + "updates" + str(number_update)+"sample"+str(arg.NUM_SAMPLES) +"IT"+ str(arg.NUM_IT) + '.pkl')
            torch.save(save_dict, savename)
          elif filename[:-4]=='.pkl':
            torch.save(save_dict, filename)
          else:
            torch.save(save_dict, (filename+'.pkf'))

          print(result)

  print('done')


def continue_inverse(filename):
  data=load_inverse_data(filename)
  theta=data['theta_estimations'][0]
  run_inverse(data=data,filename=filename, theta=theta)


def _jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                


def _hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)                                             


def compute_H_monkey(env, 
                    agent, 
                    theta_estimation, 
                    phi, 
                    H_dim=11, 
                    num_episodes=2,
                    **kwargs):
  states, actions, tasks=kwargs['monkeydata']
  theta_estimation=torch.nn.Parameter(torch.Tensor(theta_estimation))
  phi=torch.Tensor(phi)
  phi.requires_grad=False
  loss = monkeyloss(agent, actions, tasks, phi, theta_estimation, env, 
                    action_var=0.1,
                    num_iteration=1, 
                    states=states, 
                    samples=num_episodes,
                    gpu=False)
  grads = torch.autograd.grad(loss, theta_estimation, create_graph=True,allow_unused=True)[0]
  H = torch.zeros(H_dim,H_dim)
  for i in range(H_dim):
      print(i)
      H[i] = torch.autograd.grad(grads[i], theta_estimation, retain_graph=True,allow_unused=True)[0].view(-1)
  return H


def compute_H(env, agent, theta_estimation, true_theta, phi, trajectory_data=None,H_dim=7, 
            num_episodes=100,is1d=False):
  states, actions, tasks=trajectory(agent, torch.Tensor(phi), torch.Tensor(true_theta), env, num_episodes,is1d=is1d)
  theta_estimation=torch.nn.Parameter(torch.Tensor(theta_estimation))
  phi=torch.nn.Parameter(torch.Tensor(phi))
  phi.requires_grad=False
  loss = getLoss(agent, actions, tasks, phi, theta_estimation, env,states=states, gpu=False)
  grads = torch.autograd.grad(loss, theta_estimation, create_graph=True,allow_unused=True)[0]
  H = torch.zeros(H_dim,H_dim)
  for i in range(H_dim):
      print(i)
      H[i] = torch.autograd.grad(grads[i], theta_estimation, retain_graph=True,allow_unused=True)[0].view(-1)
  return H


def compute_loss(env, agent, theta_estimation, true_theta, phi, 
  trajectory_data=None, num_episodes=100,is1d=False):
  if trajectory_data is None:
    states, actions, tasks=trajectory(agent, torch.Tensor(phi), torch.Tensor(true_theta), env, num_episodes, is1d=is1d)
  else:
    actions=trajectory_data['actions']
    tasks=trajectory_data['tasks']
  theta_estimation=torch.nn.Parameter(torch.Tensor(theta_estimation))
  loss = getLoss(agent, actions, tasks, torch.Tensor(phi), theta_estimation, env,states=states)
  return loss


def compute_loss_wrapped(env, agent, true_theta, phi, trajectory_data=None, 
  num_episodes=100,is1d=False):
  new_function= lambda theta_estimation: compute_loss(env, agent, theta_estimation, true_theta, phi, num_episodes=100,is1d=is1d)
  return new_function


def inverse_pca(score, evectors, mu):
  return score@evectors.transpose()[:2,:]+mu


def plot_background(ax, xyrange,mu, evectors, loss_function, 
    number_pixels=5, num_episodes=100, method='PCA', look='contour', alpha=0.5):
  background_data=np.zeros((number_pixels,number_pixels))
  X,Y=np.meshgrid(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels),np.linspace(xyrange[1][0],xyrange[1][1],number_pixels))
  if method =='PCA':
    for i,u in enumerate(np.linspace(xyrange[1][0],xyrange[1][1],number_pixels)):
      for j,v in enumerate(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels)):
        score=np.array([u,v])
        reconstructed_theta=score@evectors.transpose()[:2,:]+mu
        if not np.array_equal(reconstructed_theta,reconstructed_theta.clip(1e-4,999)):
          print(reconstructed_theta)
          background_data[i,j]=np.nan
        else:
          reconstructed_theta=reconstructed_theta.clip(1e-4,999)
          print(loss_function(torch.Tensor(reconstructed_theta).reshape(-1,1)))
          background_data[i,j]=loss_function(torch.Tensor(reconstructed_theta).reshape(-1,1))
    # elif method=='v_noise':
  # elif method =='random':
  #   for i,u in enumerate(np.linspace(xyrange[1][0],xyrange[1][1],number_pixels)):
  #     for j,v in enumerate(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels)):
  #       score=np.array([u,v])
  #       distance=score-mu[:2]
  #       reconstructed_theta=mu+distance@np.linalg.inv(evectors)

  #       reconstructed_theta=reconstructed_theta.clip(1e-4,999)
  #       print(loss_function(torch.Tensor(reconstructed_theta).reshape(-1,1)))
  #       background_data[i,j]=loss_function(torch.Tensor(reconstructed_theta).reshape(9,1))
  if look=='contour':
    ax.contourf(X,Y,background_data,alpha=alpha,zorder=1)
  elif look=='pixel':
    im=ax.imshow(X,Y,background_data,alpha=alpha,zorder=1)
    add_colorbar(im)
  return background_data


def theta_cov(H):
  return np.linalg.inv(H)


def stderr(cov):
  return np.sqrt(np.diag(cov)).tolist()


def loss_surface_two_param(mu,loss_function, param_pair,
    number_pixels=5, param_range=[[0.001,0.4],[0.001,0.4]]):
  if type(mu) == torch.Tensor:
    mu=np.array(mu)
  background_data=np.zeros((number_pixels,number_pixels))
  X,Y=np.meshgrid(np.linspace(param_range[0][0],param_range[0][1],number_pixels),np.linspace(param_range[1][0],param_range[1][1],number_pixels))
  for i in range(background_data.shape[0]):
    for j in range(background_data.shape[1]):
      reconstructed_theta=mu.copy()
      reconstructed_theta[param_pair[0]]=[X[i,j]]
      reconstructed_theta[param_pair[1]]=[Y[i,j]]
      # reconstructed_theta=reconstructed_theta.clip(1e-4,999)
      background_data[i,j]=loss_function(torch.Tensor(reconstructed_theta).reshape(-1,1))
  return background_data


def plot_inverse_trajectorys(theta_trajectorys,true_theta, 
                      phi, background_data=None, 
                      background_contour=False,
                      ax=None,number_pixels=20,alpha=0.3):

    # plot trajectory
    if ax is None:
        fig = plt.figure(figsize=[8, 8])
        ax = fig.add_subplot()

    # reshape the data into colume as features
    data_lenths=[0]
    for i, theta_trajectory in enumerate(theta_trajectorys):
      data_lenths.append(data_lenths[-1]+len(theta_trajectory))
      if i==0:
        all_data_matrix=column_feature_data(theta_trajectory)
      else:
        all_data_matrix=np.vstack((all_data_matrix,column_feature_data(theta_trajectory)))
    # Perform PCA on the all data matrix X
    score, evectors, evals = pca(all_data_matrix)
    mu=np.mean(all_data_matrix,0)

    ax.set_xlabel('Projection basis vector 1')
    ax.set_ylabel('Projection basis vector 2')
    ax.set_title('Inverse for theta estimation in a PCs space. Sample corr: {:.1f}'.format(np.corrcoef(score[:, 0], score[:, 1])[0, 1]))


    # plot true theta

    if type(true_theta)==list:
      true_theta=np.array(true_theta).reshape(-1)
      true_theta_pc=(true_theta-mu)@evectors
    elif type(true_theta)==torch.nn.parameter.Parameter:
      true_theta_pc=(true_theta.detach().numpy().reshape(-1)-mu)@evectors
    else:
      true_theta_pc=(true_theta-mu)@evectors
    ax.scatter(true_theta_pc[0],true_theta_pc[1],marker='o',c='',edgecolors='g')

    row_cursor=0
    i=0
    while row_cursor<score.shape[0]-1:
      row_cursor+=1
      if row_cursor in data_lenths:
        i+=1
        continue
      
      # plot point
      ax.plot(score[row_cursor, 0], score[row_cursor, 1], '.', color=[.5, .5, .5],alpha=0.3)
      # plot arrow
      ax.quiver(score[row_cursor-1, 0], score[row_cursor-1, 1],
              score[row_cursor, 0]-score[row_cursor-1, 0], score[row_cursor, 1]-score[row_cursor-1, 1],
              angles='xy',color=colors[i],scale=1, scale_units='xy')

    ax.axis('equal')
    # plot hessian

    # plot log likelihood contour
    if background_contour:
      print(np.array(true_theta).reshape(-1,1), np.array(phi).reshape(-1,1))
      loss_function=compute_loss_wrapped(env, agent, np.array(true_theta).reshape(-1,1), np.array(phi).reshape(-1,1), trajectory_data=None, num_episodes=100)
      current_xrange=ax.get_xlim()
      current_yrange=ax.get_ylim()
      xyrange=[current_xrange,current_yrange]
      # ax1.contourf(X,Y,background_data)
      background_data=plot_background(ax, xyrange,mu, evectors, loss_function, number_pixels=number_pixels,alpha=alpha) if background_data is None else background_data


    return background_data


def diagnose_dict(index,inverse_data, monkeystate=True,num_trials=10,**kwargs):
  inverse_data=load_inverse_data(inverse_data)
  packed={}
  packed['index']=index
  packed['phi']=torch.tensor(inverse_data['phi'])
  packed['theta']=torch.tensor(inverse_data['theta_estimations'][-1])
  if monkeystate:
    packed['estate']=kwargs['states'][index]
  else: 
    packed['estate']=None
    packed['monkeystate']=kwargs['states'][index]
  packed['eaction']=kwargs['actions'][index]
  packed['etask']=kwargs['tasks'][index]
  packed['agent']=kwargs['agent']
  packed['env']=kwargs['env']
  packed['initv']=kwargs['actions'][index][0][0]
  packed['initw']=kwargs['actions'][index][0][1]
  return packed


def diagnose_trial(index, phi, eaction, etask, theta, agent, env, num_trials=10,
    monkeystate=None,estate=None, guided=True, initv=0, initw=0):
  agent_actions=[]
  agent_beliefs=[]
  agent_covs=[]
  agent_states=[]
  with torch.no_grad():
    for trial_i in range(num_trials):
      env.reset(phi=phi,theta=theta,goal_position=etask,vctrl=initv, wctrl=initw)
      epbliefs=[]
      epbcov=[]
      epactions=[]
      epstates=[]
      t=0
      while t<len(eaction):
        action = agent(env.decision_info)[0]
        if estate is None and guided:
          # if t==1: print('mk action with process noise')
          _,done=env(torch.tensor(eaction[t]).reshape(1,-1),task_param=theta) 
        elif estate is None and not guided:
          # if t==1: print('agent by itself')
          env.step(torch.tensor(action).reshape(1,-1)) 
        elif estate is not None and guided:
          if  t+1<(estate.shape[1]):
              # if t==1: 
              # print('at mk situation',estate[:,t+1].view(-1,1))
              _,done=env(torch.tensor(eaction[t]).reshape(1,-1),task_param=theta,state=estate[:,t+1].view(-1,1)) 
        epactions.append(action)
        epbliefs.append(env.b)
        # print(env.b, action)
        epbcov.append(env.P)
        epstates.append(env.s)
        t=t+1
      agent_actions.append(torch.stack(epactions))
      agent_beliefs.append(torch.stack(epbliefs))
      agent_covs.append(epbcov)
      agent_states.append(torch.stack(epstates)[:,:,0])
    astate=torch.stack(agent_states)[0,:,:].t() #used when agent on its own
    if estate is None:
      estate=astate     
    return_dict={
      'agent_actions':agent_actions,
      'agent_beliefs':agent_beliefs,
      'agent_covs':agent_covs,
      'estate':estate,
      'astate':agent_states,
      'eaction':eaction,
      'etask':etask,
      'theta':theta,
      'index':index,
    }
  return return_dict


def diagnose_plot(estate, theta,eaction, etask, agent_actions, agent_beliefs, 
  agent_covs,astate=None,index=None):
  fig = plt.figure(figsize=[16, 16])

  ax1 = fig.add_subplot(221)
  ax1.set_xlabel('t')
  ax1.set_ylabel('v')
  ax1.set_title('v control')
  for i in range(len(agent_actions)):
    ax1.plot(agent_actions[i][:,0],alpha=0.7)
  ax1.set_ylim([-1.1,1.1])
  ax1.plot(torch.tensor(eaction)[:,0], label='monkey', alpha=0.4, linewidth=10)
  ax1.legend()
  # ax1.set_yticks(np.linspace(-30,200,10))
  # ax1.set_xlim([-100,100])
  # ax1.axis('equal')
  
  ax2 = fig.add_subplot(222)
  ax2.set_xlabel('t')
  ax2.set_ylabel('w')
  ax2.set_title('w control')
  for i in range(len(agent_actions)):
    ax2.plot(agent_actions[i][:,1],alpha=0.7)
  ax2.set_ylim([-1.1,1.1])
  ax2.plot(torch.tensor(eaction)[:,1], label='monkey', alpha=0.4, linewidth=10)
  ax2.legend()


  ax3 = fig.add_subplot(223)
  ax3.set_xlabel('world x, cm')
  ax3.set_ylabel('world y, cm')
  ax3.set_title('state')
  ax3.plot(estate[:,0],estate[:,1])
  goalcircle = plt.Circle((etask[0],etask[1]), 0.13, color='y', alpha=0.5)
  ax3.add_patch(goalcircle)
  ax3.set_xlim([-0.1,1.1])
  ax3.set_ylim([-0.6,0.6])

  ax4 = fig.add_subplot(224)
  ax4.set_xlabel('world x, cm')
  ax4.set_ylabel('world y, cm')
  ax4.set_title('belief')
  goalcircle = plt.Circle((etask[0],etask[1]), theta[6,0], color='y', alpha=0.5)
  ax4.add_patch(goalcircle)
  ax4.set_xlim([-0.1,1.1])
  ax4.set_ylim([-0.6,0.6])
  # diaginfo['agent_beliefs'][0][:,:,0][:,0]
  # diaginfo['agent_beliefs'][0][:,:,0][:,1]
  # len(diaginfo['agent_beliefs'][0][:,:,0])
  for i in range(len(agent_beliefs)):
    for t in range(len(agent_beliefs[i][:,:,0])):
      cov=agent_covs[0][t][:2,:2]
      pos=  [agent_beliefs[0][:,:,0][t,0],
              agent_beliefs[0][:,:,0][t,1]]
      plot_cov_ellipse(cov, pos, nstd=2, color=None, ax=ax4,alpha_factor=0.2)
  ax4.plot(astate[0][:,0],astate[0][:,1])


def diagnose_plot_theta(agent, env, phi, theta_init, theta_final,nplots,):
  def sample_trials(agent, env, theta, phi, etask, num_trials=5):
    # print(theta)
    agent_actions=[]
    agent_beliefs=[]
    agent_covs=[]
    agent_states=[]
    agent_dev_costs=[]
    agent_mag_costs=[]
    with torch.no_grad():
      for trial_i in range(num_trials):
        env.reset(phi=phi,theta=theta,goal_position=etask[0],vctrl=torch.zeros(1), wctrl=torch.zeros(1))
        epbliefs=[]
        epbcov=[]
        epactions=[]
        epstates=[]
        t=0
        done=False
        while not done:
          action = agent(env.decision_info)[0]
          _,_,done,_=env.step(torch.tensor(action).reshape(1,-1)) 
          epactions.append(action)
          epbliefs.append(env.b)
          epbcov.append(env.P)
          epstates.append(env.s)
          t=t+1
        agent_dev_costs.append(torch.stack(env.trial_costs))
        # agent_mag_costs.append(torch.stack(env.trial_mag_costs))
        agent_actions.append(torch.stack(epactions))
        agent_beliefs.append(torch.stack(epbliefs))
        agent_covs.append(epbcov)
        agent_states.append(torch.stack(epstates))
      estate=torch.stack(agent_states)[0,:,:,0].t()
      return_dict={
        'agent_actions':agent_actions,
        'agent_beliefs':agent_beliefs,
        'agent_covs':agent_covs,
        'estate':estate,
        # 'eaction':eaction,
        'etask':etask,
        'theta':theta,
        'devcosts':agent_dev_costs,
        'magcosts':agent_mag_costs,
      }
    return return_dict
  delta=(theta_final-theta_init)/(nplots-1)
  fig = plt.figure(figsize=[20, 20])
  # curved trails
  etask=[[0.7,-0.3]]
  for n in range(nplots):
    ax1 = fig.add_subplot(6,nplots,n+1)
    theta=(n-1)*delta+theta_init
    ax1.set_xlabel('world x, cm')
    ax1.set_ylabel('world y, cm')
    ax1.set_title('state plot')
    data=sample_trials(agent, env, theta, phi, etask, num_trials=1)
    estate=data['estate']
    ax1.plot(estate[0,:],estate[1,:], color='r',alpha=0.5)
    goalcircle = plt.Circle((etask[0][0],etask[0][1]), 0.13, color='y', alpha=0.5)
    ax1.add_patch(goalcircle)
    ax1.set_xlim([-0.1,1.1])
    ax1.set_ylim([-0.6,0.6])
    agent_beliefs=data['agent_beliefs']
    for t in range(len(agent_beliefs[0][:,:,0])):
      cov=data['agent_covs'][0][t][:2,:2]
      pos=  [agent_beliefs[0][:,:,0][t,0],
              agent_beliefs[0][:,:,0][t,1]]
      plot_cov_ellipse(cov, pos, nstd=2, color=None, ax=ax1,alpha_factor=0.2)

  # v and w
    ax2 = fig.add_subplot(6,nplots,n+nplots+1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('w')
    ax2.set_title('w control')
    agent_actions=data['agent_actions']
    for i in range(len(agent_actions)):
      ax2.plot(agent_actions[i],alpha=0.7)
    ax2.set_ylim([-1.1,1.1])
  # dev and mag costs
    ax2 = fig.add_subplot(6,nplots,n+nplots*2+1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('costs')
    ax2.set_title('costs')
    for i in range(len(data['devcosts'])):
      ax2.plot(data['devcosts'][0], color='green',alpha=0.7)
      # ax2.plot(data['magcosts'][0], color='violet',alpha=0.7)

  # straight trails
  etask=[[0.7,0.0]]
  for n in range(nplots):
    ax1 = fig.add_subplot(6,nplots,n+nplots*3+1)
    theta=(n-1)*delta+theta_init
    ax1.set_xlabel('world x, cm')
    ax1.set_ylabel('world y, cm')
    ax1.set_title('state plot')
    data=sample_trials(agent, env, theta, phi, etask, num_trials=1)
    estate=data['estate']
    ax1.plot(estate[0,:],estate[1,:], color='r',alpha=0.5)
    goalcircle = plt.Circle((etask[0][0],etask[0][1]), 0.13, color='y', alpha=0.5)
    ax1.add_patch(goalcircle)
    ax1.set_xlim([-0.1,1.1])
    ax1.set_ylim([-0.6,0.6])
    agent_beliefs=data['agent_beliefs']
    for t in range(len(agent_beliefs[0][:,:,0])):
      cov=data['agent_covs'][0][t][:2,:2]
      pos=  [agent_beliefs[0][:,:,0][t,0],
              agent_beliefs[0][:,:,0][t,1]]
      plot_cov_ellipse(cov, pos, nstd=2, color=None, ax=ax1,alpha=0.2)

  # v and w
    ax2 = fig.add_subplot(6,nplots,n+nplots*4+1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('w')
    ax2.set_title('w control')
    data.keys()
    agent_actions=data['agent_actions']
    for i in range(len(agent_actions)):
      ax2.plot(agent_actions[i],alpha=0.7)
    ax2.set_ylim([-1.1,1.1])

    # dev and mag costs
    ax2 = fig.add_subplot(6,nplots,n+nplots*5+1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('costs')
    ax2.set_title('costs')
    for i in range(len(data['devcosts'])):
      ax2.plot(data['devcosts'][0], color='green',alpha=0.7)
      # ax2.plot(data['magcosts'][0], color='violet',alpha=0.7)


def diagnose_plot_theta1d(agent, env, phi, theta_init, theta_final,nplots,initv=0.):
  def sample_trials(agent, env, theta, phi, etask, num_trials=5, initv=0.):
    agent_actions=[]
    agent_beliefs=[]
    agent_covs=[]
    agent_states=[]
    agent_dev_costs=[]
    agent_mag_costs=[]
    with torch.no_grad():
      for trial_i in range(num_trials):
        env.reset(phi=phi,theta=theta,goal_position=etask[0]*torch.ones(1),initv=initv*torch.ones(1))
        epbliefs=[]
        epbcov=[]
        epactions=[]
        epstates=[]
        t=0
        done=False
        while not done:
          action = agent(env.decision_info)[0]
          _,_,done,_=env.step(action,onetrial=True) 
          epactions.append(action)
          epbliefs.append(env.b)
          epbcov.append(env.P)
          epstates.append(env.s)
          t+=1
          # print(t,done)
        agent_dev_costs.append(torch.stack(env.trial_dev_costs))
        agent_mag_costs.append(torch.stack(env.trial_mag_costs))
        agent_actions.append(torch.stack(epactions))
        agent_beliefs.append(torch.stack(epbliefs))
        agent_covs.append(epbcov)
        agent_states.append(torch.stack(epstates))
      estate=torch.stack(agent_states)[0,:,:,0].t()
      return_dict={
        'agent_actions':agent_actions,
        'agent_beliefs':agent_beliefs,
        'agent_covs':agent_covs,
        'estate':estate,
        # 'eaction':eaction,
        'etask':etask,
        'theta':theta,
        'devcosts':agent_dev_costs,
        'magcosts':agent_mag_costs,
      }
    return return_dict
  delta=(theta_final-theta_init)/(nplots-1)
  fig = plt.figure(figsize=[20, 16])
  # curved trails
  etask=[0.7]
  for n in range(nplots):
    ax1 = fig.add_subplot(3,nplots,n+1)
    theta=(n-1)*delta+theta_init
    ax1.set_xlabel('world x, cm')
    ax1.set_ylabel('world y, cm')
    ax1.set_title('state plot')
    data=sample_trials(agent, env, theta, phi, etask, num_trials=1, initv=initv)
    estate=data['estate']
    # if n==1:
    #   print(agent_beliefs[0][:,:,0][:,0])
    ax1.plot(list(range(len(estate[0,:]))),estate[0,:], color='r',alpha=0.5)
    # goalcircle = plt.Circle((etask[0],0.), 0.13, color='y', alpha=0.5)
    # ax1.add_patch(goalcircle)
    # ax1.set_xlim([-0.1,1.1])
    # ax1.set_ylim([-0.6,0.6])
    agent_beliefs=data['agent_beliefs']
    ax1.plot(agent_beliefs[0][:,:,0][:,0])
    # for t in range(len(agent_beliefs[0][:,:,0])):
    #   # cov=data['agent_covs'][0][t][:2,:2]
    #   print(agent_beliefs[0][:,:,0][:,0])
    #   pos=  [agent_beliefs[0][:,:,0][t,0],
    #           agent_beliefs[0][:,:,0][t,1]]
      # plot_cov_ellipse(cov, pos, nstd=2, color=None, ax=ax1,alpha=0.2)
  # v and w
    ax2 = fig.add_subplot(3,nplots,n+nplots+1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('w')
    ax2.set_title('w control')
    agent_actions=data['agent_actions']
    for i in range(len(agent_actions)):
      ax2.plot(agent_actions[i],alpha=0.7)
    ax2.set_ylim([-1.1,1.1])
  # dev and mag costs
    ax2 = fig.add_subplot(3,nplots,n+nplots*2+1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('costs')
    ax2.set_title('costs')
    for i in range(len(data['devcosts'])):
      ax2.plot(data['devcosts'][0], color='green',alpha=0.7)
      ax2.plot(data['magcosts'][0], color='violet',alpha=0.7)


def diagnose_plot_xaqgrant(estate, theta,eaction, etask, agent_actions, agent_beliefs, 
    agent_covs,astate=None,index=None, tasks=None, actions=None):
  fig = plt.figure(figsize=[12, 8])
  ax1 = fig.add_subplot(221)
  ax1.set_xlabel('time, s')
  ax1.set_ylabel('velocity, m/s')
  ax1.set_title('forward control')
  for i in range(len(agent_actions)):
    ax1.plot(agent_actions[i][:,0],alpha=0.7, color='orange')
  ax1.set_ylim([-0.1,1.1])
  indls=similar_trials(index,tasks,actions)
  for i in indls:
    ax1.plot(torch.tensor(actions[i])[:,0],color='steelblue', alpha=0.4)
  ax1.plot(torch.tensor(eaction)[:,0], label='monkey', alpha=0.4)
  ax2 = fig.add_subplot(222)
  ax2.set_xlabel('time, s')
  ax2.set_ylabel('angular velocity, 90 degree/s')
  ax2.set_title('angular control')
  for i in range(len(agent_actions)):
    ax2.plot(agent_actions[i][:,1],alpha=0.7, color='orange')
  ax2.plot(agent_actions[i][:,1],alpha=0.7, color='orange',label='agent')
  ax2.set_ylim([-0.7,0.7])
  ax2.plot(torch.tensor(eaction)[:,1], label='monkey', alpha=0.4)
  for i in indls:
    ax2.plot(torch.tensor(actions[i])[:,1],color='steelblue', alpha=0.4)
  ax2.legend()


def diagnose_stopcompare(estate, theta,eaction, etask, agent_actions, agent_beliefs, agent_covs,
    astate=None,index=None, tasks=None, actions=None):
  fig = plt.figure(figsize=[8, 8])
  ax1 = fig.add_subplot(111)
  ax1.set_xlim([-0.1,1.1])
  ax1.set_ylim([-0.6,0.6])
  ax1.set_xlabel('world x, cm')
  ax1.set_ylabel('world y, cm')
  ax1.set_title('stop positions')
  indls=similar_trials(index,tasks,actions)
  for i in indls:
    goalcircle = plt.Circle((tasks[i][0][0],tasks[i][0][1]), 0.13, color='y', alpha=0.3,fill=False)
    ax1.add_patch(goalcircle)
    ax1.plot(states[i][0,:],states[i][1,:], 'steelblue', alpha=0.2)
    ax1.scatter(states[i][0,-1],states[i][1,-1], color='steelblue', alpha=0.5)
  for a in astate:
    ax1.plot(a[:,0],a[:,1],'orange', alpha=0.2)
    ax1.scatter(a[-1,0],a[-1,1], color='orange', alpha=0.5)


def diagnose_plot_xaqgrantoverhead(estate, theta,eaction, etask, agent_actions, agent_beliefs, 
    agent_covs,astate=None):
  fig = plt.figure(figsize=[10, 8])
  ax3 = fig.add_subplot(223)
  ax3.set_xlabel('world x, cm')
  ax3.set_ylabel('world y, cm')
  ax3.set_title('arena overhead view')
  ax3.plot(estate[0,:],estate[1,:],alpha=0.4,linewidth=2, label='actual positions')
  goalcircle = plt.Circle((etask[0][0],etask[0][1]), 0.13, color='y', alpha=0.4)
  ax3.add_patch(goalcircle)
  ax3.set_xlim([-0.1,1.1])
  ax3.set_ylim([-0.6,0.6])

  for t in range(len(agent_beliefs[0][:,:,0])):
    cov=diaginfo['agent_covs'][0][t][:2,:2]
    pos=  [agent_beliefs[0][:,:,0][t,0],
            agent_beliefs[0][:,:,0][t,1]]
    plot_cov_ellipse(cov, pos, nstd=2, color='orange', ax=ax3,alpha=0.2)
  ax3.plot(0,0, color='orange', linewidth=2, label='estimated belief')
  ax3.legend()


def diagnose_plot_stop(estate, theta,eaction, etask, agent_actions, agent_beliefs, 
    agent_covs,astate=None, index=None):
  fig = plt.figure(figsize=[16, 16])

  ax1 = fig.add_subplot(221)
  ax1.set_xlabel('t')
  ax1.set_ylabel('v')
  ax1.set_title('v control')
  for i in range(len(agent_actions)):
    ax1.plot(agent_actions[i][:,0],alpha=0.7)
  ax1.set_ylim([-1.1,1.1])
  ax1.plot(torch.tensor(eaction)[:,0], label='monkey', alpha=0.4, linewidth=10)
  ax1.legend()
  
  ax2 = fig.add_subplot(222)
  ax2.set_xlabel('t')
  ax2.set_ylabel('w')
  ax2.set_title('w control')
  for i in range(len(agent_actions)):
    ax2.plot(agent_actions[i][:,1],alpha=0.7)
  ax2.set_ylim([-1.1,1.1])
  ax2.plot(torch.tensor(eaction)[:,1], label='monkey', alpha=0.4, linewidth=10)
  ax2.legend()

  ax3 = fig.add_subplot(223)
  ax3.set_xlabel('world x, cm')
  ax3.set_ylabel('world y, cm')
  ax3.set_title('state')
  for i in range(len(agent_actions)):
    ax3.scatter(astate[i][-1,0],astate[i][-1,1])
  goalcircle = plt.Circle((etask[0][0],etask[0][1]), 0.13, color='y', alpha=0.5)
  ax3.add_patch(goalcircle)
  ax3.set_xlim([-0.1,1.1])
  ax3.set_ylim([-0.6,0.6])

  ax4 = fig.add_subplot(224)
  ax4.set_xlabel('world x, cm')
  ax4.set_ylabel('world y, cm')
  ax4.set_title('belief')
  goalcircle = plt.Circle((etask[0][0],etask[0][1]), theta[6,0], color='y', alpha=0.5)
  ax4.add_patch(goalcircle)
  ax4.set_xlim([-0.1,1.1])
  ax4.set_ylim([-0.6,0.6])
  for i in range(len(agent_actions)):
    # for t in range(len(agent_beliefs[0][:,:,0])):
      cov=diaginfo['agent_covs'][i][-1][:2,:2]
      pos=  [agent_beliefs[i][:,:,0][-1,0],
              agent_beliefs[i][:,:,0][-1,1]]
      plot_cov_ellipse(cov, pos, nstd=2, color=None, ax=ax4,alpha=0.1)


def similar_trials(ind, tasks, actions):
  indls=[]
  for i in range(len(tasks)):
      if tasks[i][0]>tasks[ind][0]-0.05 \
      and tasks[i][0]<tasks[ind][0]+0.05 \
      and tasks[i][1]>tasks[ind][1]-0.03 \
      and tasks[i][1]<tasks[ind][1]+0.03 \
      and actions[i][0][0]>actions[ind][0][0]-0.1 \
      and actions[i][0][1]<actions[ind][0][1]+0.1:
          indls.append(i)
  return indls

# check if simulated state dynamic matches with actual. only 1cm for 1m arena.
def diagnose_dynamic_wrapped():
  def diagnose_dynamic(index, phi, eaction, etask, theta, agent, env, num_trials=10,monkeystate=None,estate=None, guided=True, initv=0):
    agent_actions=[]
    agent_beliefs=[]
    agent_covs=[]
    agent_states=[]
    with torch.no_grad():
      for trial_i in range(num_trials):
        env.reset(phi=phi,theta=theta,goal_position=etask[0],initv=initv)
        epbliefs=[]
        epbcov=[]
        epactions=[]
        epstates=[]
        t=0
        while t<len(eaction):
          action = agent(env.decision_info)[0]
          _,done=env(torch.tensor(eaction[t]).reshape(1,-1),task_param=theta) # this
          epactions.append(action)
          epbliefs.append(env.b)
          epbcov.append(env.P)
          epstates.append(env.s)
          t=t+1
        agent_actions.append(torch.stack(epactions))
        agent_beliefs.append(torch.stack(epbliefs))
        agent_covs.append(epbcov)
        agent_states.append(torch.stack(epstates)[:,:,0])
      astate=torch.stack(agent_states)[0,:,:].t() #used when agent on its own 
      return_dict={
        'agent_actions':agent_actions,
        'agent_beliefs':agent_beliefs,
        'agent_covs':agent_covs,
        'estate':estate,
        'astate':agent_states,
        'eaction':eaction,
        'etask':etask,
        'theta':theta,
        'index':index,
      }
    return return_dict
  ind=torch.randint(low=100,high=7000,size=(1,))
  pac=diagnose_dict(ind,'1_21_58', monkeystate=True)
  pac['initv']=actions[ind][0][0]
  pac['guided']=True
  pac['num_trials']=2
  pac['theta']=torch.tensor([[0.4],
          [1.57],
          [0.5],
          [0.5],
          [0.5],
          [0.5],
          [0.13],
          [0.0550],
          [0.6],
          [0.6],
          [0.1],
          [0.1]])
  diaginfo=diagnose_dynamic(**pac)
  for a in diaginfo['astate']:
    plt.plot(a[:,0],a[:,1],'green')
    plt.title(((diaginfo['estate'][0,-1]-a[-1,0])**2+(a[-1,1]-diaginfo['estate'][1,-1])**2)**0.5)
  plt.plot(diaginfo['estate'][0,:],diaginfo['estate'][1,:])


# assume the wrong gain and agent makes sense 
def agent_double_gain():
    phi=torch.tensor([[0.4000],
            [1.57],
            [0.01],
            [0.01],
            [0.01],
            [0.01],
            [0.13],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
    ])
    theta1=torch.tensor([[0.4000],
            [1.57],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
            [0.13],
            [0.9],
            [0.9],
            [0.1],
            [0.1],
    ])
    # double gain theta
    theta2=torch.tensor([[0.8000],
            [1.57],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
            [0.13],
            [0.9],
            [0.9],
            [0.1],
            [0.1],
    ])
    input={
        'agent':agent,
        'theta':theta1,
        'phi':phi,
        'env': env,
        'num_trials':100,
        # 'task':[0.7,-0.3],
        'mkdata':{},
        'use_mk_data':False,
    }

    with suppress():
        res1=trial_data(input)
    errs1=[]
    goalds1=[]
    for trialind in range(res1['num_trials']):
        err=torch.norm(torch.tensor(res1['task'][trialind])-res1['agent_states'][trialind][-1][:2])
        errs1.append(err)
        goalds1.append(torch.norm(torch.tensor(res1['task'][trialind])))
    input={
        'agent':agent,
        'theta':theta2,
        'phi':phi,
        'env': env,
        'num_trials':100,
        'task':res1['task'],
        'mkdata':{},
        'use_mk_data':False,
    }
    with suppress():
        res2=trial_data(input)
    errs2=[]
    goalds2=[]
    for trialind in range(res2['num_trials']):
        err=torch.norm(torch.tensor(res2['task'][trialind])-res2['agent_states'][trialind][-1][:2])
        errs2.append(err)
        goalds2.append(torch.norm(torch.tensor(res2['task'][trialind])))

    errs1=[e.item() for e in errs1]
    goalds1=[e.item() for e in goalds1]
    errs2=[e.item() for e in errs2]
    goalds2=[e.item() for e in goalds2]

    # error scatter plot
    with initiate_plot(3,3,300):
        plt.scatter(goalds1,errs1, alpha=0.3,edgecolors='none')
        plt.scatter(goalds2,errs2, alpha=0.3,edgecolors='none')
        plt.title('error vs goal distance', fontsize=20)
        plt.ylabel('error')
        plt.xlabel('goal distance')

    # overhead view plot
    ax=plotoverheadcolor(res1)
    plotoverheadcolor(res2,linewidth=1,alpha=0.3,ax=ax)
    ax.get_figure()


# large goal radius
def agent_double_goalr():
    theta2=torch.tensor([[0.4],
            [1.57],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
            [0.4],
            [0.9],
            [0.9],
            [0.1],
            [0.1],
    ])
    input={
        'agent':agent,
        'theta':theta1,
        'phi':phi,
        'env': env,
        'num_trials':100,
        # 'task':[0.7,-0.3],
        'mkdata':{},
        'use_mk_data':False,
    }

    with suppress():
        res1=trial_data(input)
    errs1=[]
    goalds1=[]
    for trialind in range(res1['num_trials']):
        err=torch.norm(torch.tensor(res1['task'][trialind])-res1['agent_states'][trialind][-1][:2])
        errs1.append(err)
        goalds1.append(torch.norm(torch.tensor(res1['task'][trialind])))
    input={
        'agent':agent,
        'theta':theta2,
        'phi':phi,
        'env': env,
        'num_trials':100,
        'task':res1['task'],
        'mkdata':{},
        'use_mk_data':False,
    }
    with suppress():
        res2=trial_data(input)
    errs2=[]
    goalds2=[]
    for trialind in range(res2['num_trials']):
        err=torch.norm(torch.tensor(res2['task'][trialind])-res2['agent_states'][trialind][-1][:2])
        errs2.append(err)
        goalds2.append(torch.norm(torch.tensor(res2['task'][trialind])))

    errs1=[e.item() for e in errs1]
    goalds1=[e.item() for e in goalds1]
    errs2=[e.item() for e in errs2]
    goalds2=[e.item() for e in goalds2]

    # error scatter plot
    with initiate_plot(3,3,300):
        plt.scatter(goalds1,errs1, alpha=0.3,edgecolors='none')
        plt.scatter(goalds2,errs2, alpha=0.3,edgecolors='none')
        plt.title('error vs goal distance', fontsize=20)
        plt.ylabel('error')
        plt.xlabel('goal distance')

    # plt.hist(errs1)
    # plt.hist(errs2)

    # overhead view plot
    ax=plotoverheadcolor(res1)
    plotoverheadcolor(res2,linewidth=1,alpha=0.3,ax=ax)
    ax.get_figure()


# verify phi task vs true task
def verify():
    index=ind
    s=states[index]
    a=actions[index]
    env.reset(phi=phi,theta=theta,
        goal_position=tasks[index],vctrl=a[0,0],wctrl=a[0,1])
    vstates=[]
    for aa in a:
        env.step(aa)
        vstates.append(env.s)
    svstates=torch.stack(vstates)[:,:,0]
    svstates=svstates-svstates[0]
    plt.plot(svstates[:,0],svstates[:,1])
    plt.plot(s[:,0],s[:,1])
    plt.show()
    svstates[-1,:2]
    s[-1,:2]


def overhead_skip_density(input):
    fontsize = 9
    target_idexes = np.arange(1500, 2000)
    with initiate_plot(1.8, 1.8, 300) as fig:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
        
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
        
        skipx=[]
        skipy=[]
        for trial_i in range(input['num_trials']):
            # calculate if rewarded
            dx=input['agent_states'][trial_i][-1,1]*400-input['task'][trial_i][1]*400
            dy=input['agent_states'][trial_i][-1,0]*400-input['task'][trial_i][0]*400
            d2goal=(dx**2+dy**2)**0.5
            d2start=(input['agent_states'][trial_i][-1,1]**2+input['agent_states'][trial_i][-1,0]**2)**0.5*400

            if d2goal>200:
                skipx.append(input['task'][trial_i][1]*400)
                skipy.append(input['task'][trial_i][0]*400)
        skipx=torch.stack(skipx).tolist()  
        skipy=torch.stack(skipy).tolist()  
        densObj = kde(np.array([skipx,skipy]))
        colours = makeColours( densObj.evaluate( np.array([skipx,skipy]) ) )
        ax.scatter( skipx, skipy, color=colours)
        # img, extent = myplot(skipx, skipy, 77)
        # ax.contourf(img, extent=extent, origin='lower', cmap=cm.Greys)
        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)

        fig.tight_layout(pad=0)


def diagnose_plot_theta(agent, env, phi, theta_init, theta_final,nplots,etask=[[0.7,-0.3]],initv=0.,initw=0.,mkactions=None):
  def sample_trials(agent, env, theta, phi, etask, num_trials=5,initv=0.,initw=0.):
    # print(theta)
    agent_actions=[]
    agent_beliefs=[]
    agent_covs=[]
    agent_states=[]
    agent_dev_costs=[]
    agent_mag_costs=[]
    with torch.no_grad():
      for trial_i in range(num_trials):
        env.reset(phi=phi,theta=theta,goal_position=etask,vctrl=initv, wctrl=initw)
        epbliefs=[]
        epbcov=[]
        epactions=[]
        epstates=[]
        done=False
        while not done:
          action = agent(env.decision_info)[0]
          noise=torch.normal(torch.zeros(2),0.1)
          _action=(action+noise).clamp(-1,1)
          _,_,done,_=env.step(_action) 
          epactions.append(action)
          epbliefs.append(env.b)
          epbcov.append(env.P)
          epstates.append(env.s)
        agent_dev_costs.append((env.trial_costs))
        # agent_mag_costs.append(torch.stack(env.trial_mag_costs))
        agent_actions.append(torch.stack(epactions))
        agent_beliefs.append(torch.stack(epbliefs))
        agent_covs.append(epbcov)
        agent_states.append(torch.stack(epstates))
      estate=torch.stack(agent_states)[0,:,:,0].t()
      return_dict={
        'agent_actions':agent_actions,
        'agent_beliefs':agent_beliefs,
        'agent_covs':agent_covs,
        'estate':estate,
        # 'eaction':eaction,
        'etask':etask,
        'theta':theta,
        'devcosts':agent_dev_costs,
        'magcosts':agent_mag_costs,
      }
    return return_dict
  delta=(theta_final-theta_init)/(nplots-1)
  fig = plt.figure(figsize=[20, 20])
  # curved trails
  for n in range(nplots):
    ax1 = fig.add_subplot(6,nplots,n+1)
    theta=(n-1)*delta+theta_init
    ax1.set_xlabel('world x [cm]')
    ax1.set_ylabel('world y [cm]')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_title('state plot')
    data=sample_trials(agent, env, theta, phi, etask, num_trials=1,initv=initv,initw=initw)
    estate=data['estate']
    ax1.plot(estate[0,:]*200,estate[1,:]*200, color='r',alpha=0.5)
    goalcircle = plt.Circle((etask[0]*200,etask[1]*200), 65, color='y', alpha=0.5)
    ax1.add_patch(goalcircle)
    ax1.set_aspect('equal')
    # ax1.set_xlim([-0.1,1.1])
    # ax1.set_ylim([-0.6,0.6])
    agent_beliefs=data['agent_beliefs']
    for t in range(len(agent_beliefs[0][:,:,0])):
      cov=data['agent_covs'][0][t][:2,:2]*200*200
      pos=  [agent_beliefs[0][:,:,0][t,0]*200,
              agent_beliefs[0][:,:,0][t,1]*200]
      plot_cov_ellipse(cov, pos, nstd=3, color=None, ax=ax1,alpha=0.2)

  # v and w
    ax2 = fig.add_subplot(6,nplots,n+nplots+1)
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('control amplitude')
    # ax2.set_title('w control')
    agent_actions=data['agent_actions']
    for i in range(len(agent_actions)):
      ax2.plot(np.arange(len(agent_actions[i]))/10,agent_actions[i],alpha=0.7)
    ax2.set_ylim([-1.1,1.1])
    ax2.set_yticks([-1.,0,1.])
    ax2.set_xlim(left=0.)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    if n==2 and mkactions is not None:
          ax2.plot(np.arange(len(mkactions))/10,mkactions,alpha=0.7)
    ''' 
        # dev and mag costs
            ax2 = fig.add_subplot(6,nplots,n+nplots*2+1)
            ax2.set_xlabel('t')
            ax2.set_ylabel('costs')
            ax2.set_title('costs')
            for i in range(len(data['devcosts'])):
            ax2.plot(data['devcosts'][0], color='green',alpha=0.7)
            # ax2.plot(data['magcosts'][0], color='violet',alpha=0.7)

        straight trails
        etask=[[0.7,0.0]]
        for n in range(nplots):
            ax1 = fig.add_subplot(6,nplots,n+nplots*3+1)
            theta=(n-1)*delta+theta_init
            ax1.set_xlabel('world x, cm')
            ax1.set_ylabel('world y, cm')
            ax1.set_title('state plot')
            data=sample_trials(agent, env, theta, phi, etask, num_trials=1)
            estate=data['estate']
            ax1.plot(estate[0,:],estate[1,:], color='r',alpha=0.5)
            goalcircle = plt.Circle((etask[0][0],etask[0][1]), theta[6], color='y', alpha=0.5)
            ax1.add_patch(goalcircle)
            ax1.set_xlim([-0.1,1.1])
            ax1.set_ylim([-0.6,0.6])
            agent_beliefs=data['agent_beliefs']
            for t in range(len(agent_beliefs[0][:,:,0])):
            cov=data['agent_covs'][0][t][:2,:2]
            pos=  [agent_beliefs[0][:,:,0][t,0],
                    agent_beliefs[0][:,:,0][t,1]]
            plot_cov_ellipse(cov, pos, nstd=2, color=None, ax=ax1,alpha=0.2)

        # v and w
            ax2 = fig.add_subplot(6,nplots,n+nplots*4+1)
            ax2.set_xlabel('t')
            ax2.set_ylabel('w')
            ax2.set_title('w control')
            data.keys()
            agent_actions=data['agent_actions']
            for i in range(len(agent_actions)):
            ax2.plot(agent_actions[i],alpha=0.7)
            ax2.set_ylim([-1.1,1.1])

            # dev and mag costs
            ax2 = fig.add_subplot(6,nplots,n+nplots*5+1)
            ax2.set_xlabel('t')
            ax2.set_ylabel('costs')
            ax2.set_title('costs')
            for i in range(len(data['devcosts'])):
            ax2.plot(data['devcosts'][0], color='green',alpha=0.7)
        # ax2.plot(data['magcosts'][0], color='violet',alpha=0.7)
    '''


def mytick(x,nticks=5, roundto=0):
    # x, input. nticks, number of ticks. round, >0 for decimal, <0 for //10 power
    if roundto!=0:
        scale=10**(-roundto)
        return np.linspace(np.floor(np.min(x)/scale)*scale, np.ceil(np.max(x)/scale)*scale,nticks)
    else:
        return np.linspace(np.floor(np.min(x)), np.ceil(np.max(x)),nticks)


def vary_theta_new(agent, env, phi, theta_init, theta_final,nplots,etask=[0.7,-0.3],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=10):

  def sample_trials(agent, env, theta, phi, thistask, initv=0.,initw=0., action_noise=0.1,pert=None,):
    agent_actions=[]
    agent_beliefs=[]
    agent_covs=[]
    agent_states=[]
    with torch.no_grad():
        env.reset(phi=phi,theta=theta,goal_position=thistask,vctrl=initv, wctrl=initw)
        if pert is None:
            env.obs_traj= np.vstack([np.array([0,0]),np.random.normal(0,1,size=(env.episode_len,2))*np.asfarray(theta[4:6]).T,np.zeros(shape=(20,2))])
        else:
            env.obs_traj= np.vstack([np.array([0,0]),np.random.normal(0,1,size=(pert))*np.asfarray(theta[4:6]).T + pert,np.zeros(shape=(20,2))])
        epbliefs=[]
        epbcov=[]
        epactions=[]
        epstates=[]
        done=False
        while not done:
            action = agent(env.decision_info)[0]
            noise=torch.normal(torch.zeros(2),action_noise)
            _action=(action+noise).clamp(-1,1)
            _,_,done,_=env.step(_action) 
            epactions.append(action)
            epbliefs.append(env.b)
            epbcov.append(env.P)
            epstates.append(env.s)
        agent_actions.append(torch.stack(epactions))
        agent_beliefs.append(torch.stack(epbliefs))
        agent_covs.append(epbcov)
        agent_states.append(torch.stack(epstates))
        estate=torch.stack(agent_states)[0,:,:,0].t()
        return_dict={
        'agent_actions':agent_actions,
        'agent_beliefs':agent_beliefs,
        'agent_covs':agent_covs,
        'estate':estate,
        # 'eaction':eaction,
        'etask':thistask,
        'theta':theta,
      }
    return return_dict

  delta=(theta_final-theta_init)/(nplots-1)

  with initiate_plot(3*nplots, 3,300) as fig:
    subplotmap={}
    for n in range(nplots):
        subplotmap[n]=fig.add_subplot(1,nplots,n+1)

    for n in range(nplots):

        theta=n*delta+theta_init
        theta=torch.clamp(theta, 1e-3)

        for i in range(ntrials):
            with suppress():
                data=sample_trials(agent, env, theta, phi, etask,initv=initv,initw=initw,pert=pert)
            estate=data['estate']
            agent_beliefs=data['agent_beliefs']
            agent_actions=data['agent_actions']
            
            # overhead
            ax = subplotmap[n]
            ax.plot(estate[0,:]*200,estate[1,:]*200, color=color_settings['s'],alpha=0.5)
            ax.scatter(estate[0,-1]*200,estate[1,-1]*200, color=color_settings['a'],alpha=0.5)
            if i==0:
                goalcircle = plt.Circle((etask[0]*200,etask[1]*200), 65, color=color_settings['goal'], edgecolor='none', alpha=0.3, linewidth=0.8)
                ax.add_patch(goalcircle)
                ax.set_aspect('equal')
                ax.set_xlabel('world x [cm]')
                ax.set_ylabel('world y [cm]')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            for t in range(len(agent_beliefs[0][:,:,0])):
                cov=data['agent_covs'][0][t][:2,:2]*200*200
                pos=  [agent_beliefs[0][:,:,0][t,0]*200,
                        agent_beliefs[0][:,:,0][t,1]*200]
                plot_cov_ellipse(cov, pos, nstd=3, color=color_settings['b'], ax=ax,alpha=0.05)
            if n!=0:
                ax.set_yticklabels([])
                # ax.set_xticklabels([])
                # ax.set_xlabel('')
                ax.set_ylabel('')
            ax.set_xticks([]); ax.set_yticks([])
      
        ax.set_xticks(mytick(ax.get_xlim(),3,-1))
        ax.set_yticks(mytick(ax.get_ylim(),3,-1))
        plt.tight_layout()


def vary_theta(agent, env, phi, theta_init, theta_final,nplots,etask=[[0.7,-0.3]],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=10):

    def _sample_trials(agent, env, theta, phi, etask, initv=0.,initw=0., action_noise=0.1,pert=None):
        agent_actions=[]
        agent_beliefs=[]
        agent_covs=[]
        agent_states=[]
        with torch.no_grad():
            env.reset(phi=phi,theta=theta,goal_position=etask,vctrl=initv, wctrl=initw)
            epbliefs=[]
            epbcov=[]
            epactions=[]
            epstates=[]
            done=False
            while not done:
                action = agent(env.decision_info)[0]
                noise=torch.normal(torch.zeros(2),action_noise)
                _action=(action+noise).clamp(-1,1)
                if pert is not None and int(env.trial_timer)<len(pert):
                    _action+=pert[int(env.trial_timer)]
                _,_,done,_=env.step(_action) 
                epactions.append(action)
                epbliefs.append(env.b)
                epbcov.append(env.P)
                epstates.append(env.s)
            agent_actions.append(torch.stack(epactions))
            agent_beliefs.append(torch.stack(epbliefs))
            agent_covs.append(epbcov)
            agent_states.append(torch.stack(epstates))
            estate=torch.stack(agent_states)[0,:,:,0].t()
            return_dict={
            'agent_actions':agent_actions,
            'agent_beliefs':agent_beliefs,
            'agent_covs':agent_covs,
            'estate':estate,
            # 'eaction':eaction,
            'etask':etask,
            'theta':theta,
        }
        return return_dict
    

    def sample_trials(agent, env, theta, phi, thistask, initv=0.,initw=0., action_noise=0.1,pert=None,):
        agent_actions=[]
        agent_beliefs=[]
        agent_covs=[]
        agent_states=[]
        with torch.no_grad():
            env.reset(phi=phi,theta=theta,goal_position=thistask,vctrl=initv, wctrl=initw)
            if pert is None:
                env.obs_traj= np.vstack([np.array([0,0]),np.random.normal(0,1,size=(env.episode_len,2))*np.asfarray(theta[4:6]).T,np.zeros(shape=(20,2))])
            else:
                env.obs_traj= np.vstack([np.array([0,0]),np.random.normal(0,1,size=(pert))*np.asfarray(theta[4:6]).T + pert,np.zeros(shape=(20,2))])
            epbliefs=[]
            epbcov=[]
            epactions=[]
            epstates=[]
            done=False
            while not done:
                action = agent(env.decision_info)[0]
                noise=torch.normal(torch.zeros(2),action_noise)
                _action=(action+noise).clamp(-1,1)
                # if pert is not None and int(env.trial_timer)<len(pert):
                #     _action+=pert[int(env.trial_timer)]
                _,_,done,_=env.step(_action) 
                epactions.append(action)
                epbliefs.append(env.b)
                epbcov.append(env.P)
                epstates.append(env.s)
            agent_actions.append(torch.stack(epactions))
            agent_beliefs.append(torch.stack(epbliefs))
            agent_covs.append(epbcov)
            agent_states.append(torch.stack(epstates))
            estate=torch.stack(agent_states)[0,:,:,0].t()
            return_dict={
            'agent_actions':agent_actions,
            'agent_beliefs':agent_beliefs,
            'agent_covs':agent_covs,
            'estate':estate,
            # 'eaction':eaction,
            'etask':thistask,
            'theta':theta,
        }
        return return_dict


    delta=(theta_final-theta_init)/(nplots-1)

    with initiate_plot(3*nplots, 3*3,300) as fig:
        subplotmap={}
        for n in range(nplots):
            for i in range(3): # overhead, v, w
                subplotmap[n,i]=fig.add_subplot(3,nplots,i*nplots+n+1)


        for n in range(nplots):
            theta=n*delta+theta_init
            theta=torch.clamp(theta, 1e-3)
            for i in range(ntrials):
                with suppress():
                    data=sample_trials(agent, env, theta, phi, etask,initv=initv,initw=initw,pert=pert)
                estate=data['estate']
                agent_beliefs=data['agent_beliefs']
                agent_actions=data['agent_actions']

                # overhead
                ax = subplotmap[n,0]
                ax.plot(estate[0,:]*200,estate[1,:]*200, color=color_settings['s'],alpha=0.5)
                ax.scatter(estate[0,-1]*200,estate[1,-1]*200, color=color_settings['a'],alpha=0.5)
                if i==0:
                    goalcircle = plt.Circle((etask[0]*200,etask[1]*200), 65, color=color_settings['goal'], edgecolor='none', alpha=0.3, linewidth=0.8)
                    ax.add_patch(goalcircle)
                    ax.set_aspect('equal')
                    ax.set_xlabel('world x [cm]')
                    ax.set_ylabel('world y [cm]')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                for t in range(len(agent_beliefs[0][:,:,0])):
                    cov=data['agent_covs'][0][t][:2,:2]*200*200
                    pos=  [agent_beliefs[0][:,:,0][t,0]*200,
                            agent_beliefs[0][:,:,0][t,1]*200]
                    plot_cov_ellipse(cov, pos, nstd=3, color=color_settings['b'], ax=ax,alpha=0.05)
                if n!=0:
                    ax.set_yticklabels([])
                    # ax.set_xticklabels([])
                    # ax.set_xlabel('')
                    ax.set_ylabel('')
                ax.set_xticks(mytick(ax.get_xticks(),3,-1))
                ax.set_yticks(mytick(ax.get_yticks(),3,-1))

                # forward control
                ax = subplotmap[n,1]
                # ax = fig.add_subplot(3,nplots,n+nplots+1)
                if pert is not None and i==0:
                    pertv=np.abs(pert.T[0])
                    pertcolor=matplotlib.colors.ListedColormap(np.linspace([1,0,0,0],[1,0,0,np.max(pertv)],200))
                    pertvbox=np.broadcast_to(pertv,(20,len(pertv)))
                    im = ax.imshow(pertvbox, interpolation='bilinear', cmap=pertcolor, aspect='auto',origin='lower',extent=[0, len(pertv)/10,-0.2, 1])

                ax.set_ylabel('forward control')
                for j in range(len(agent_actions)):
                    ax.plot(np.arange(len(agent_actions[j]))/10,agent_actions[j][:,0],alpha=0.7, color=color_settings['model'], linewidth=1)
                ax.set_ylim([-0.2,1.1])
                ax.set_yticks([0,1.])
                # ax.set_xticks([])
                ax.set_xlim(left=0.)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_xticklabels([])    
                if n!=0:
                    ax.set_yticklabels([])
                    ax.set_ylabel('')
                if n==nplots//2 and mkactions is not None:
                    ax.plot(np.arange(len(mkactions))/10,mkactions[:,0],alpha=0.7, color=color_settings['a'])
                
                # angular control
                ax = subplotmap[n,2]
                # ax = fig.add_subplot(3,nplots,n+2*nplots+1)
                if pert is not None and i==0:
                    pertw=np.abs(pert.T[1])
                    pertcolor=matplotlib.colors.ListedColormap(np.linspace([1,0,0,0],[1,0,0,np.max(pertw)],100))
                    pertwbox=np.broadcast_to(pertw,(20,len(pertw)))
                    im = ax.imshow(pertwbox, interpolation='bilinear', cmap=pertcolor, aspect='auto',origin='lower',extent=[0, len(pertw)/10,-1, 1])

                ax.set_xlabel('t [s]')
                ax.set_ylabel('angular control')
                for j in range(len(agent_actions)):
                    ax.plot(np.arange(len(agent_actions[j]))/10,agent_actions[j][:,1],alpha=0.7, color=color_settings['model'])
                ax.set_ylim([-1.1,1.1])
                ax.set_yticks([-1.,0,1.])
                ax.set_xlim(left=0.)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_position('zero')
                if n!=0:
                    ax.set_yticklabels([])
                    ax.set_ylabel('')
                if n==nplots//2 and mkactions is not None:
                    ax.plot(np.arange(len(mkactions))/10,mkactions[:,1],alpha=0.7, color=color_settings['a'])

            plt.tight_layout()


def two_theta_overhead(agent, env, phi, thetas, labels=['model1','model2'],etask=[[0.7,-0.3]],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=10):

    def sample_trials(agent, env, theta, phi, thistask, initv=0.,initw=0., action_noise=0.1,pert=None,):
        agent_actions=[]
        agent_beliefs=[]
        agent_covs=[]
        agent_states=[]
        with torch.no_grad():
            env.reset(phi=phi,theta=theta,goal_position=thistask,vctrl=initv, wctrl=initw)
            if pert is None:
                env.obs_traj= np.vstack([np.array([0,0]),np.random.normal(0,1,size=(env.episode_len,2))*np.asfarray(theta[4:6]).T,np.zeros(shape=(20,2))])
            else:
                env.obs_traj= np.vstack([np.array([0,0]),np.random.normal(0,1,size=(pert))*np.asfarray(theta[4:6]).T + pert,np.zeros(shape=(20,2))])
            epbliefs=[]
            epbcov=[]
            epactions=[]
            epstates=[]
            done=False
            while not done:
                action = agent(env.decision_info)[0]
                noise=torch.normal(torch.zeros(2),action_noise)
                _action=(action+noise).clamp(-1,1)
                # if pert is not None and int(env.trial_timer)<len(pert):
                #     _action+=pert[int(env.trial_timer)]
                _,_,done,_=env.step(_action) 
                epactions.append(action)
                epbliefs.append(env.b)
                epbcov.append(env.P)
                epstates.append(env.s)
            agent_actions.append(torch.stack(epactions))
            agent_beliefs.append(torch.stack(epbliefs))
            agent_covs.append(epbcov)
            agent_states.append(torch.stack(epstates))
            estate=torch.stack(agent_states)[0,:,:,0].t()
            return_dict={
            'agent_actions':agent_actions,
            'agent_beliefs':agent_beliefs,
            'agent_covs':agent_covs,
            'estate':estate,
            # 'eaction':eaction,
            'etask':thistask,
            'theta':theta,
        }
        return return_dict

    with initiate_plot(3, 3,300) as fig:
        ax=fig.add_subplot(111)
        for n,theta in enumerate(thetas):
            theta=torch.clamp(theta, 1e-3)
            for i in range(ntrials):
                with suppress():
                    data=sample_trials(agent, env, theta, phi, etask,initv=initv,initw=initw,pert=pert)
                estate=data['estate']
                agent_beliefs=data['agent_beliefs']

                # overhead
                ax.plot(estate[0,:]*200,estate[1,:]*200, color=color_settings['pair'][n],alpha=0.5,label=labels[n])
                ax.legend()
                # ax.scatter(estate[0,-1]*200,estate[1,-1]*200, color=color_settings['a'],alpha=0.5)
                if i==0:
                    goalcircle = plt.Circle((etask[0]*200,etask[1]*200), 65, color=color_settings['goal'], edgecolor='none', alpha=0.3, linewidth=0.8)
                    ax.add_patch(goalcircle)
                    ax.set_aspect('equal')
                    ax.set_xlabel('world x [cm]')
                    ax.set_ylabel('world y [cm]')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                ax.set_xticks(mytick(ax.get_xticks(),3,-1))
                ax.set_yticks(mytick(ax.get_yticks(),3,-1))
            
        handles, labels_ = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_, handles))
        ax.legend(by_label.values(), by_label.keys(),loc='upper right')
        plt.tight_layout()



def ll2array(v):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.zeros(mask.shape)
    out[mask] = np.concatenate(v)
    return out


def vary_theta_ctrl(agent, env, phi, theta_init, theta_final,nplots=5,etask=[[0.7,-0.3]],initv=0.,initw=0.,mkactions=None,ntrials=10,pert=None):

    def sample_trials(agent, env, theta, phi, etask, initv=0.,initw=0., action_noise=0.1,pert=None):
        agent_actions=[]
        agent_beliefs=[]
        agent_covs=[]
        agent_states=[]
        with torch.no_grad():
            env.reset(phi=phi,theta=theta,goal_position=etask,vctrl=initv, wctrl=initw)
            epbliefs=[]
            epbcov=[]
            epactions=[]
            epstates=[]
            done=False
            while not done:
                action = agent(env.decision_info)[0]
                noise=torch.normal(torch.zeros(2),action_noise)
                _action=(action+noise).clamp(-1,1)
                if pert is not None and int(env.trial_timer)<len(pert):
                    _action+=pert[int(env.trial_timer)]
                _,_,done,_=env.step(_action) 
                epactions.append(action)
                epbliefs.append(env.b)
                epbcov.append(env.P)
                epstates.append(env.s)
            agent_actions.append(torch.stack(epactions))
            agent_beliefs.append(torch.stack(epbliefs))
            agent_covs.append(epbcov)
            agent_states.append(torch.stack(epstates))
            estate=torch.stack(agent_states)[0,:,:,0].t()
            return_dict={
            'agent_actions':agent_actions,
            'agent_beliefs':agent_beliefs,
            'agent_covs':agent_covs,
            'estate':estate,
            # 'eaction':eaction,
            'etask':etask,
            'theta':theta,
        }
        return return_dict
  
    delta=(theta_final-theta_init)/(nplots-1)
    fig = plt.figure(figsize=[10, 5])
    percentile=5
    basecolor=hex2rgb(color_settings['model'])
    colors=colorshift(basecolor,basecolor,0.5,nplots)
    labels=['- eigen vector','mean','+ eigen vector'  ]
    ax = fig.add_subplot(2,1,1)
    for n in range(nplots): # this is actually number of theta
        theta=(n-1)*delta+theta_init
        theta=torch.clamp(theta, 1e-3)
        agent_v=[]
        for i in range(ntrials):
            with suppress():
                data=sample_trials(agent, env, theta, phi, etask,initv=initv,initw=initw,pert=pert)
            agent_v.append(data['agent_actions'][0][:,0].tolist())
        agent_v=ll2array(agent_v) # ntrial, ts
        low=np.percentile(agent_v,percentile, axis=0)
        high=np.percentile(agent_v,100-percentile, axis=0)
        ts=np.arange(len(agent_v.T))*0.1
        if n==2:
            ax.fill_between(ts, y1=low, y2=high, color=color_settings['a'],alpha=0.2, edgecolor='none', label=labels[n])
        else:
            ax.fill_between(ts, y1=low, y2=high, color=colors[n],alpha=0.2, edgecolor='none', label=labels[n])
    ax.set_ylim([-0.2,1.1])
    ax.set_yticks([0,1.])
    ax.set_xlim(left=0.)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('forward control')
    # ax.set_xticklabels([])    
    # ax.set_xlabel('t [s]')
    ax.plot(np.arange(len(mkactions))/10,mkactions[:,0],alpha=0.7, color=color_settings['a'])
    leg=ax.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    ax = fig.add_subplot(2,1,2,sharex=ax)
    for n in range(nplots): # this is actually number of theta
        theta=(n-1)*delta+theta_init
        agent_w=[]
        for i in range(ntrials):
            with suppress():
                data=sample_trials(agent, env, theta, phi, etask,initv=initv,initw=initw)
            agent_w.append(data['agent_actions'][0][:,1].tolist())
        agent_w=ll2array(agent_w) # ntrial, ts
        low=np.percentile(agent_w,percentile, axis=0)
        high=np.percentile(agent_w,100-percentile, axis=0)
        ts=np.arange(len(agent_w.T))*0.1
        if n==2:
            ax.fill_between(ts, y1=low, y2=high, color=color_settings['a'],alpha=0.2, edgecolor='none', label=labels[n])
        else:
            ax.fill_between(ts, y1=low, y2=high, color=colors[n],alpha=0.2, edgecolor='none', label=labels[n])

    ax.set_xlabel('t [s]')
    ax.set_ylabel('angular control')
    ax.set_ylim([-1.1,1.1])
    ax.set_yticks([-1.,0,1.])
    ax.set_xlim(left=0.)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.plot(np.arange(len(mkactions))/10,mkactions[:,1],alpha=0.7, color=color_settings['a'])
    plt.tight_layout()


def pert_disk(pert):
    # simple clock version (ego centric)
    with initiate_plot(3,3) as fig:
        ax = fig.subplots(subplot_kw={'projection': 'polar'})
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim([0,1])
        ax.plot([pi/2,pi/2],[0,1],'k')
        direction=np.nan_to_num(pert[:,0]/pert[:,1],0)
        sign=-1 if np.any(direction<0) else 1
        theta=np.min(direction) if sign==-1 else np.max(direction)
        ax.plot([theta]*2,[0,1],'k')


def likelihoodvstime(agent=None, 
            actions=None, 
            tasks=None, 
            phi=None, 
            theta=None, 
            env=None,
            states=None, 
            samples=50,
            action_var=0.1,
            ep=None,
            pert=None):

    task=tasks[ep]
    with torch.no_grad():
            logllt=[] # log ll at time t
            totalll=0.
            # get mk actions
            mkactionep = actions[ep][1:]
            # make env list
            envls=[]
            for _ in range(samples):
                env.debug=True
                envls.append(copy.deepcopy(env))
            # reset task
            for env in envls:
                    env.reset(theta=theta, phi=phi, goal_position=task, vctrl=mkactionep[0][0],wctrl=mkactionep[0][1])
            # run task
            for t,mk_action in enumerate(mkactionep[1:]):
                action = [agent(env.decision_info) for env in envls]
                nextstate=states[ep][1:][t].view(-1,1)
                for env in envls:
                    env.b, env.P=env.belief_step(env.b,env.P, env.observations(nextstate), torch.tensor(mk_action).view(1,-1))
                    previous_action=mk_action # current action is prev action for next time
                    env.trial_timer+=1
                    env.decision_info=env.wrap_decision_info(previous_action=previous_action, time=env.trial_timer)

                action_ll = [logll(torch.tensor(mk_action),a,std=np.sqrt(action_var)).view(-1).tolist() for a in action]
                obs_ll = [logll(error=env.obs_err(), std=theta[4:6].view(1,-1)).view(-1).tolist() for env in envls]
                totalll = totalll + np.sum(action_ll)+ np.sum(obs_ll)
                logllt.append(totalll/samples) # this is log liklihood

    return  np.exp(logllt) # this is likelihood


def irc_pert(agent, env, phi, theta,etask=[[0.7,-0.3]],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=10):

    def sample_trials(agent, env, theta, phi, etask, initv=0.,initw=0., action_noise=0.1,pert=None):
        agent_actions=[]
        agent_beliefs=[]
        agent_covs=[]
        agent_states=[]
        with torch.no_grad():
            env.reset(phi=phi,theta=theta,goal_position=etask,vctrl=initv, wctrl=initw)
            epbliefs=[]
            epbcov=[]
            epactions=[]
            epstates=[]
            done=False
            while not done:
                action = agent(env.decision_info)[0]
                noise=torch.normal(torch.zeros(2),action_noise)
                _action=(action+noise).clamp(-1,1)
                if pert is not None and int(env.trial_timer)<len(pert):
                    _action+=pert[int(env.trial_timer)]
                _,_,done,_=env.step(_action) 
                epactions.append(action)
                epbliefs.append(env.b)
                epbcov.append(env.P)
                epstates.append(env.s)
            agent_actions.append(torch.stack(epactions))
            agent_beliefs.append(torch.stack(epbliefs))
            agent_covs.append(epbcov)
            agent_states.append(torch.stack(epstates))
            estate=torch.stack(agent_states)[0,:,:,0].t()
            return_dict={
            'agent_actions':agent_actions,
            'agent_beliefs':agent_beliefs,
            'agent_covs':agent_covs,
            'estate':estate,
            # 'eaction':eaction,
            'etask':etask,
            'theta':theta,
        }
        return return_dict
  
    fig = plt.figure(figsize=[5, 3])
    percentile=5
    # basecolor=hex2rgb(color_settings['model'])

    # forward
    ax = fig.add_subplot(2,1,1)
    agent_v=[]
    for i in range(ntrials):
        with suppress():
            data=sample_trials(agent, env, theta, phi, etask,initv=initv,initw=initw,pert=pert)
        agent_v.append(data['agent_actions'][0][:,0].tolist())
    
        pertv=pert.T[0]

    pertv=np.abs(pert.T[0])
    pertcolor=matplotlib.colors.ListedColormap(np.linspace([1,0,0,0],[1,0,0,np.max(pertv)],200))
    pertvbox=np.broadcast_to(pertv,(20,len(pertv)))
    im = ax.imshow(pertvbox, interpolation='bilinear', cmap=pertcolor, aspect='auto',
                    origin='lower',extent=[0, len(pertv)/10,-0.2, 1])

    agent_v=ll2array(agent_v) # ntrial, ts
    low=np.percentile(agent_v,percentile, axis=0)
    high=np.percentile(agent_v,100-percentile, axis=0)
    ts=np.arange(len(agent_v.T))*0.1
    ax.fill_between(ts, y1=low, y2=high, color=color_settings['model'],alpha=0.2, edgecolor='none', label='IRC')
    ax.set_ylim([-0.2,1.1])
    ax.set_yticks([0,1.])
    ax.set_xlim(left=0.)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('forward control')
    # ax.set_xticklabels([])    
    # ax.set_xlabel('t [s]')
    ax.plot(np.arange(len(mkactions))/10,mkactions[:,0],alpha=0.7, color=color_settings['a'], label='monkey')
    leg=ax.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    # angular
    ax = fig.add_subplot(2,1,2,sharex=ax)
    agent_w=[]
    for i in range(ntrials):
        with suppress():
            data=sample_trials(agent, env, theta, phi, etask,initv=initv,initw=initw)
        agent_w.append(data['agent_actions'][0][:,1].tolist())
    agent_w=ll2array(agent_w) # ntrial, ts
    low=np.percentile(agent_w,percentile, axis=0)
    high=np.percentile(agent_w,100-percentile, axis=0)
    ts=np.arange(len(agent_w.T))*0.1

    ax.fill_between(ts, y1=low, y2=high, color=color_settings['model'],alpha=0.2, edgecolor='none', label='IRC')

    ax.set_xlabel('t [s]')
    ax.set_ylabel('angular control')
    ax.set_ylim([-1.1,1.1])
    ax.set_yticks([-1.,0,1.])
    ax.set_xlim(left=0.)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.plot(np.arange(len(mkactions))/10,mkactions[:,1],alpha=0.7, color=color_settings['a'])

    pertw=np.abs(pert.T[1])
    pertcolor=matplotlib.colors.ListedColormap(np.linspace([1,0,0,0],[1,0,0,np.max(pertw)],200))
    pertwbox=np.broadcast_to(pertw,(20,len(pertw)))
    im = ax.imshow(pertwbox, interpolation='bilinear', cmap=pertcolor, aspect='auto',
                    origin='lower',extent=[0, len(pertv)/10,-1, 1])



    plt.tight_layout()


def policygiventheta(env=None,agent=None, theta=None):

    env.reset(phi=phi,theta=theta)
    bl=torch.zeros(5)
    # br=torch.tensor([0.0,0.0,0.0,0,0])
    P=torch.eye(5) * 1e-8 
    P[0,0]=(env.theta[9]*0.05)**2 # sigma xx
    P[1,1]=(env.theta[10]*0.05)**2 # sigma yy
    reso=30

    with initiate_plot(15, 10, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # vary distance
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,0]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,0]=1. 
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,1)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.set_title('distance')

        # vary angle
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,1]=-1.5
        beliefr=beliefl.clone().detach()
        beliefr[0,1]=1.5 
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,2)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('angle')

        # vary v
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,2]=-1.
        beliefr=beliefl.clone().detach()
        beliefr[0,2]=1.     
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,3)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('forward v')

        # vary w
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,3]=-1.5
        beliefr=beliefl.clone().detach()
        beliefr[0,3]=1.5   
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,4)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('angular w')
        
        # vary time
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,4]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,4]=30.    
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,5)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('time dt')

        # vary prev v
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,5]=-1.
        beliefr=beliefl.clone().detach()
        beliefr[0,5]=1.    
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,6)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.set_title('previous forward ctrl')


        # vary prev w
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,6]=-1.5
        beliefr=beliefl.clone().detach()
        beliefr[0,6]=15   
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,7)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('previous angular ctrl')
        
        # vary xx
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,7]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,7]=0.2  
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,8)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('xx uncertainty')    
        
        # vary xy
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,8]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,8]=0.2 
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,9)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('xy uncertainty') 

        # vary yy
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,9]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,9]=0.2  
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,10)
        ax.set_xlim([0, reso]); ax.set_ylim([-1, 1])
        ax.plot(actions[:,0],color=color_settings['v'])
        ax.plot(actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_title('yy uncertainty')

        # vary x heading
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,10]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,10]=0.2 
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,11)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.set_title('x heading uncertainty')
        
        # vary y heading
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,11]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,11]=0.2  
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,12)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.yaxis.set_ticks([])
        ax.set_title('y heading uncertainty')

        # vary heading
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,12]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,12]=0.2
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,13)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.yaxis.set_ticks([])
        ax.set_title('heading uncertainty')

        # vary vv
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,13]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,13]=0.9 
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,14)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.yaxis.set_ticks([])
        ax.set_title('vv uncertainty')

        # vary ww
        beliefl=env.wrap_decision_info().clone().detach()
        beliefl[0,14]=0.
        beliefr=beliefl.clone().detach()
        beliefr[0,14]=0.9
        delta=(beliefr-beliefl)/(reso-1)
        actions=[]
        with torch.no_grad():
            for n in range(reso):
                b=beliefl+delta*n
                action=agent(b)[0]
                actions.append(action)
        actions=torch.stack(actions)
        ax = fig.add_subplot(3,5,15)
        ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
        ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
        ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axes.xaxis.set_ticks([0,0.5,1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.yaxis.set_ticks([])
        ax.set_title('ww uncertainty')

        # legend and label
        teal_patch = mpatches.Patch(color=color_settings['v'], label='forward')
        orange_patch = mpatches.Patch(color=color_settings['w'], label='angular')
        ax.legend(handles=[teal_patch,orange_patch],loc='upper right',fontsize=6)


def policygiventhetav2(n,env=None):
    env.reset(phi=phi,theta=theta)
    bl=torch.zeros(5)
    # br=torch.tensor([0.0,0.0,0.0,0,0])
    P=torch.eye(5) * 1e-8 
    P[0,0]=(env.theta[9]*0.05)**2 # sigma xx
    P[1,1]=(env.theta[10]*0.05)**2 # sigma yy
    reso=30

    with initiate_plot(10, 20, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # vary ith
        for i in range(n):
            beliefl=env.wrap_decision_info().clone().detach()
            beliefl[0,i]=0.
            beliefr=beliefl.clone().detach()
            beliefr[0,i]=1. 
            delta=(beliefr-beliefl)/(reso-1)
            actions=[]
            with torch.no_grad():
                for n in range(reso):
                    b=beliefl+delta*n
                    action=agent(b)[0]
                    actions.append(action)
            actions=torch.stack(actions)
            ax = fig.add_subplot(10,5,i+1)
            ax.set_xlim([0, 1]); ax.set_ylim([-1, 1])
            ax.plot(np.linspace(0,1,reso), actions[:,0],color=color_settings['v'])
            ax.plot(np.linspace(0,1,reso),actions[:,1],color=color_settings['w'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axes.xaxis.set_ticks([0,0.5,1])
            ax.set_title('{}th'.format(i))


# inversed theta with error bar
def inversed_theta_bar(inverse_data):
    cov=theta_cov(inverse_data['Hessian'])
    theta_stds=stderr(cov)
    x_pos = np.arange(len(theta_names))
    data=[std/mean for std,mean in zip(theta_stds, theta_mean)]
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.bar(x_pos, data, color = 'tab:blue')
        ax.set_xticks(x_pos)
        ax.set_ylabel('inferred parameter std/mean')
        ax.set_xticklabels(theta_names,rotation=45,ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


# inversed theta uncertainty by std/mean
def inversed_uncertainty_bar(inverse_data):
    cov=theta_cov(inverse_data['Hessian'])
    theta_stds=stderr(cov)
    x_pos = np.arange(len(theta_names))
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar(x_pos, torch.tensor(inverse_data['theta_estimations'][-1]).flatten(), 
                yerr=theta_stds,color = 'tab:blue')
        # title and axis names
        ax.set_ylabel('inferred parameter value')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(theta_names, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


# 2.1 overhead view, monkey's belief and state in one trial
def single_trial_overhead():
    input={
        'agent':agent,
        'theta':theta,
        'phi':phi,
        'env': env,
        'num_trials':1,
        # 'task':tasks[:20],
        'mkdata':{
                        # 'trial_index': list(range(1)),               
                        'trial_index': ind,               
                        'task': [tasks[ind]],                  
                        'actions': [actions[ind]],                 
                        'states':[states[ind]],
                        },                      
        'use_mk_data':True
    }
    # load data
    etask=input['mkdata']['task'][0]
    initv=input['mkdata']['actions'][0][0][0]  
    initw=input['mkdata']['actions'][0][0][1]    
    # agent_dev_costs=[]
    # agent_mag_costs=[]
    # create plot
    with initiate_plot(3, 3, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('world x, [m]')
        ax1.set_ylabel('world y, [m]')
        # goalcircle = matplotlib.patches.Circle((etask[0],etask[1]), 0.13,), edgecolor="none",color=color_settings['goal'],alpha=0.5)
        # ax.add_patch(circle)
        goalcircle = plt.Circle((etask[0],etask[1]), 0.13, color=color_settings['goal'], edgecolor='none',linewidth=0., alpha=0.5)
        ax1.add_patch(goalcircle)
        ax1.set_xlim([-0.1,1.1])
        ax1.set_ylim([-0.6,0.6])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.set_xticks([0,0.5,1])
        ax1.set_yticks([-0.5,0.,0.5])
        with torch.no_grad():
                agent_actions=[]
                agent_beliefs=[]
                agent_covs=[]
                agent_states=[]
                env.reset(phi=phi,theta=input['theta'],goal_position=etask,vctrl=initv,wctrl=initw)
                epbliefs=[]
                epbcov=[]
                epactions=[]
                epstates=[]
                t=0
                done=False
                while not done:
                    action = input['agent'](env.decision_info)[0]
                    _,_,done,_=env.step(torch.tensor(action).reshape(1,-1)) 
                    epactions.append(action)
                    epbliefs.append(env.b)
                    epbcov.append(env.P)
                    epstates.append(env.s)
                    t=t+1
                # print(env.get_distance(env.s))
                # agent_dev_costs.append(torch.stack(env.trial_dev_costs))
                # agent_mag_costs.append(torch.stack(env.trial_mag_costs))
                agent_actions.append(torch.stack(epactions))
                agent_beliefs.append(torch.stack(epbliefs))
                agent_covs.append(epbcov)
                agent_states.append(torch.stack(epstates))
                estate=torch.stack(agent_states)[0,:,:,0]
                estate[:,0]=estate[:,0]-estate[0,0]
                estate[:,1]=estate[:,1]-estate[0,1]
                # agent path, red
                print('ploting')
                # ax1.plot(estate[:,0],estate[:,1], color=color_settings['s'],alpha=0.5)
                # agent belief path
                for t in range(len(agent_beliefs[0][:,:,0])-1):
                    cov=agent_covs[0][t][:2,:2]
                    pos=  [agent_beliefs[0][:,:,0][t,0],
                            agent_beliefs[0][:,:,0][t,1]]
                    plot_cov_ellipse(cov, pos, nstd=2, color=color_settings['b'], ax=ax1,alpha=0.5)
        # mk state, blue
        ax1.plot(input['mkdata']['states'][0][:,0],input['mkdata']['states'][0][:,1],alpha=0.9)


# 3.3 overhead view, skipped trials boundary similar
def agentvsmk_skip(env,agent, theta,states, actions, tasks, num_trials=10):
    ind=torch.randint(low=100,high=222,size=(num_trials,))
    input={
        'agent':agent,
        'theta':theta,
        'phi':phi,
        'env': env,
        'num_trials':num_trials,
        # 'task':tasks[:222],
        'mkdata':{
                        'trial_index': ind,               
                        # 'task': [tasks[i][0] for i in ind],                  
                        # 'actions': [actions[i][0] for i in ind],                 
                        # 'states':[states[i][0] for i in ind],
                        'task': tasks,                 
                        'actions': actions,                 
                        'states':states,
                        },                      
        'use_mk_data':True
    }
    with suppress():
        res1=trial_data(input)
    # plotoverhead_skip(res1)
    plotoverhead(res1)

    input={
        'agent':agent,
        'theta':theta,
        'phi':phi,
        'env': env,
        'num_trials':num_trials,
        'task':[tasks[i] for i in ind],
        'mkdata':{},                      
        'use_mk_data':False
    }

    with suppress():
        res2=trial_data(input)
    plotoverhead(res2)
    # plotoverhead_skip(res2)


def plot_critic_polar(  dmin=0.2,
                        dmax=1.0):
    critic=agent_.critic.qf0.cpu()
    nbin=10
    with torch.no_grad():
        env.reset(phi=phi,theta=theta)
        base=env.decision_info.flatten().clone().detach()
        start=base.clone().detach();start[0]=dmin; start[1]=-0.75
        xend=start.clone().detach();yend=start.clone().detach()
        xend[0]=1;yend[1]=0.75
        xdelta=(xend-start)/nbin/2
        ydelta=(yend-start)/nbin
        values=[]
        for xi in range(2*nbin):
            yvalues=[]
            for yi in range(nbin):
                thestate=start+xi*xdelta+yi*ydelta
                value=critic(torch.cat([thestate,agent(thestate).flatten()]))
                yvalues.append(value.item())
            values.append(yvalues)
        values=np.asarray(values)
        values=normalizematrix(values)
        # plt.imshow(im,origin='lower')

    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        axrange=[-0.75 , 0.75, dmin , 1]
        c=ax.imshow(values, origin='lower',extent=axrange, aspect='auto')
        add_colorbar(c)
        ax.set_ylabel('distance')
        ax.set_xlabel('angle')

    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        c=ax.contourf(values, origin='lower',extent=axrange, aspect='auto')
        ax.set_ylabel('distance')
        ax.set_xlabel('angle')
        fig.colorbar(c, ax=ax)

    with initiate_plot(5, 3, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a,d = np.meshgrid(np.linspace(-0.75, 0.75, nbin), np.linspace(dmin, 1, 2*nbin)) 
        fig = plt.figure()
        ax = fig.add_subplot(111,polar='True')
        levels=np.linspace(0, 1, 10)
        c=ax.contourf(a, d, values,levels=levels,cmap="viridis",)
        c.collections[3].set_linewidth(4)  
        c.collections[3].set_color('red') 
        c.collections[3].set_linestyle('dotted')
        ax.set_theta_offset(pi/2)
        ax.grid(False)
        ax.set_thetamin(-43)
        ax.set_thetamax(43)
        fig.colorbar(c, ax=ax)
        maskvalue=values.copy()
        maskvalue[maskvalue>0.45] = None
        c=ax.contourf(a, d, maskvalue,levels=levels,cmap="viridis",)
        fig.tight_layout()


def plot_inverse_trend(inverse_data):
    # trend
    with initiate_plot(30, 2, 300) as fig:
        numparams=len(inverse_data['theta_estimations'][0])
        for n in range(numparams):
            ax=fig.add_subplot(1,numparams,n+1,)
            y=[t[n] for t in inverse_data['theta_estimations']]
            ax.plot(y)
            ax.set_ylim([min(parameter_range[n][0],min(y)[0]),max(parameter_range[n][1],max(y)[0])])
        plt.show()
    # grad
    with initiate_plot(30, 2, 300) as fig:
        numparams=len(inverse_data['grad'][0])
        for n in range(numparams):
            ax=fig.add_subplot(1,numparams,n+1,)
            y=[t[n] for t in inverse_data['grad']]
            ax.plot(y)
            maxvalue=max(abs(min(y)),abs(max(y)))
            # ax.set_ylim(-maxvalue,maxvalue)
            ax.set_ylim(-100,100)
            ax.plot([i for i in range(len(y))],[0]*len(y))
    plt.show()
    # loss
    with initiate_plot(10,5,300) as fig:
        ax=fig.add_subplot(111)
        ax.plot(inverse_data['loss'])
    plt.show()


# make the input hashmap
def input_formatter(**kwargs):
    result={}
    for key, value in kwargs.items():
        # print("{0} = {1}".format(key, value))
        result[key]=value
    return result

# suppress the output
@contextmanager
def suppress(out=True, err=False):
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield
            

def roughbisec(a, x, lo=0, hi=None):
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        midval = a[mid]
        if midval < x:
            lo = mid+1
        elif midval > x: 
            hi = mid
        else:
            return mid
    return mid


def cart2pol(*args):
    if type(args[0])==list:
        x=args[0][0]; y=args[0][1]
    else:
        x=args[0]; y=args[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def xy2pol(*args, rotation=True): # rotated for the task
    x=args[0][0]; y=args[0][1]
    d = np.sqrt(x**2 + y**2)
    a = np.arctan2(y, x)+pi/2 if rotation else  np.arctan2(y, x)
    return d, a


def similar_trials(ind, tasks, actions=None, ntrial=10):
    indls=[] # max heap of dist and ind
    for i in range(len(tasks)):
        dx=abs(tasks[i][0]-tasks[ind][0])
        dy=abs(tasks[i][1]-tasks[ind][1])
        if actions is not None:
            dv=abs(actions[i][0][0]-actions[ind][0][0])
            dw=abs(actions[i][0][1]-actions[ind][0][1])
            d=-1*(dx**2+dy**2+0.5*dv**2+0.5*dw**2).item()
        else:
            d=-1*(dx**2+dy**2).item()
        if len(indls)>=ntrial:
            heapq.heappushpop(indls,(d,i)) # push and pop
        else:
            heapq.heappush(indls,(d,i)) # only push
    result=([i[1] for i in heapq.nlargest(10,indls)])
    return result


def similar_trials2this(tasks, thistask, thisaction=None, actions=None, ntrial=10):
    indls=[] # max heap of dist and ind
    for i in range(len(tasks)):
        dx=abs(tasks[i][0]-thistask[0])
        dy=abs(tasks[i][1]-thistask[1])
        if actions is not None:
            dv=abs(actions[i][0][0]-thisaction[0])
            dw=abs(actions[i][0][1]-thisaction[1])
            d=-1*(dx**2+dy**2+0.5*dv**2+0.5*dw**2).item()
        else:
            d=-1*(dx**2+dy**2).item()
        if len(indls)>=ntrial:
            heapq.heappushpop(indls,(d,i)) # push and pop
        else:
            heapq.heappush(indls,(d,i)) # only push
    result=([i[1] for i in heapq.nlargest(ntrial,indls)])
    return result


def normalizematrix(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_relative_r_ang(px, py, heading_angle, target_x, target_y):
    heading_angle = np.deg2rad(heading_angle)
    distance_vector = np.vstack([px - target_x, py - target_y])
    relative_r = np.linalg.norm(distance_vector, axis=0)
    
    relative_ang = heading_angle - np.arctan2(distance_vector[1],
                                              distance_vector[0])
    # make the relative angle range [-pi, pi]
    relative_ang = np.remainder(relative_ang, 2 * np.pi)
    relative_ang[relative_ang >= np.pi] -= 2 * np.pi
    return relative_r, relative_ang

        
@contextmanager
def initiate_plot(dimx=24, dimy=9, dpi=100, fontweight='normal'):
    plt.rcParams['figure.figsize'] = (dimx, dimy)
    plt.rcParams['font.weight'] = fontweight
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams["font.family"] = 'Arial'
    fig = plt.figure(dpi=dpi)
    yield fig
    plt.show()
    
    
def set_violin_plot(bp, facecolor, edgecolor, linewidth=1, alpha=1, ls='-', hatch=r''):
    plt.setp(bp['bodies'], facecolor=facecolor, edgecolor=edgecolor, 
             linewidth=linewidth, alpha=alpha ,ls=ls, hatch=hatch)
    plt.setp(bp['cmins'], facecolor=facecolor, edgecolor=edgecolor, 
             linewidth=linewidth, alpha=alpha ,ls=ls)
    plt.setp(bp['cmaxes'], facecolor=facecolor, edgecolor=edgecolor, 
             linewidth=linewidth, alpha=alpha ,ls=ls)
    plt.setp(bp['cmedians'], facecolor='k', edgecolor='k', 
             linewidth=linewidth, alpha=alpha ,ls=ls)
    plt.setp(bp['cbars'], facecolor='None', edgecolor='None', 
             linewidth=linewidth, alpha=alpha ,ls=ls)
    
    
def set_box_plot(bp, color, linewidth=1, alpha=0.9, ls='-', unfilled=False):
    if unfilled is True:
        plt.setp(bp['boxes'], facecolor='None', edgecolor=color,
                 linewidth=linewidth, alpha=1,ls=ls)
    else:
        plt.setp(bp['boxes'], facecolor=color, edgecolor=color,
                 linewidth=linewidth, alpha=alpha,ls=ls)
    plt.setp(bp['whiskers'], color='k', linewidth=linewidth, alpha=alpha ,ls=ls)
    plt.setp(bp['caps'], color='k', linewidth=linewidth, alpha=alpha ,ls=ls)
    plt.setp(bp['medians'], color='k', linewidth=linewidth, alpha=alpha ,ls=ls)
    
    
def filter_fliers(data, whis=1.5, return_idx=False):
    filtered_data = []; fliers_ides = []
    for value in data:
        Q1, Q2, Q3 = np.percentile(value, [25, 50, 75])
        lb = Q1 - whis * (Q3 - Q1); ub = Q3 + whis * (Q3 - Q1)
        filtered_data.append(value[(value > lb) & (value < ub)])
        fliers_ides.append(np.where((value > lb) & (value < ub))[0])
    if return_idx:
        return filtered_data, fliers_ides
    else:
        return filtered_data
    
    
def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10**(-precision), precision)


def my_floor(a, precision=0):
    return np.round(a - 0.5 * 10**(-precision), precision)


def reset_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    

def match_targets(df, reference):
    df.reset_index(drop=True, inplace=True)
    reference.reset_index(drop=True, inplace=True)
    df_targets = df.loc[:, ['target_x', 'target_y']].copy()
    reference_targets = reference.loc[:, ['target_x', 'target_y']].copy()

    closest_df_indices = []
    for _, reference_target in reference_targets.iterrows():
        distance = np.linalg.norm(df_targets - reference_target, axis=1)
        closest_df_target = df_targets.iloc[distance.argmin()]
        closest_df_indices.append(closest_df_target.name)
        df_targets.drop(closest_df_target.name, inplace=True)

    matched_df = df.loc[closest_df_indices]
    matched_df.reset_index(drop=True, inplace=True)
    
    return matched_df


def config_colors():
    colors = {'LSTM_c': 'olive', 'EKF_c': 'darkorange', 'monkV_c': 'indianred', 'monkB_c': 'blue',
              'sensory_c': '#29AbE2', 'belief_c': '#C1272D', 'motor_c': '#FF00FF',
              'reward_c': 'C0', 'unreward_c': 'salmon', 
              'gain_colors': ['k', 'C2', 'C3', 'C5', 'C9']}
    return colors


# Heissian polots
def sample_batch(states=None, actions=None, tasks=None, batch_size=20,**kwargs):
    totalsamples=len(tasks)
    sampleind=torch.randint(0,totalsamples,(batch_size,)) # trial inds
    sample_states=[states[i] for i in sampleind]
    sample_actions=[actions[i] for i in sampleind]
    sample_tasks=[tasks[i] for i in sampleind]
    return sample_states, sample_actions, sample_tasks


def compute_H_monkey(env, 
                    agent, 
                    theta_estimation, 
                    phi, 
                    H_dim=11, 
                    num_episodes=1,
                    num_samples=1,
                    action_var=0.1,
                    **kwargs):
    # TODO integrate sample batch
    states, actions, tasks=kwargs['monkeydata']
    totalsamples=len(tasks)
    sampleind=torch.randint(0,totalsamples,(num_episodes,)) # trial inds
    sample_states=[states[i] for i in sampleind]
    sample_actions=[actions[i] for i in sampleind]
    sample_tasks=[tasks[i] for i in sampleind]
    thistheta=torch.nn.Parameter(torch.Tensor(theta_estimation))
    phi=torch.Tensor(phi)
    phi.requires_grad=False
    loss = monkeyloss(agent, sample_actions, sample_tasks, phi, 
                        thistheta, env, 
                        action_var=action_var,
                        num_iteration=1, 
                        states=sample_states, 
                        samples=num_samples,
                        gpu=False)
    print('bp for grad')                    
    grads = torch.autograd.grad(loss, thistheta, create_graph=True,allow_unused=True)[0]
    H = torch.zeros(H_dim,H_dim)
    for i in range(H_dim):
        print('calculate {}th row of H'.format(i))
        H[i] = torch.autograd.grad(grads[i], thistheta, retain_graph=True,allow_unused=True)[0].view(-1)
    return H


def sample_all_tasks(
    na=10, 
    nd=10,
    matrix=True):
    angles=np.linspace(-0.75, 0.75, na)#; angles=torch.tensor(angles)
    distances=np.linspace(0.2, 1, nd)#; distances=torch.tensor(distances)
    if matrix:
        d,a=np.array(np.meshgrid(distances,angles))
        return d,a
    else:
        tasks=[]
        for a in angles:
            for d in distances:
                tasks.append(pol2xy(a,d))
        return tasks


def pol2xy(a,d):
    x=d*np.cos(a)
    y=d*np.sin(a)
    return [x,y]



def convert_2d_response(x, y, z, xmin, xmax, ymin, ymax, num_bins=20, kernel_size=3, isconvolve=True):
    @njit
    def compute(*args):
        x_bins = np.linspace(xmin - 1, xmax + 1, num_bins + 1)
        y_bins = np.linspace(ymin - 1, ymax + 1, num_bins + 1)

        x_indices = np.digitize(x, x_bins) - 1
        y_indices = np.digitize(y, y_bins) - 1

        data = np.zeros((num_bins, num_bins))
        count = data.copy()
        for z_idx, z_value in enumerate(z):
            data[y_indices[z_idx], x_indices[z_idx]] += z_value
            count[y_indices[z_idx], x_indices[z_idx]] += 1

        data /= count
        return x_bins, y_bins, data
    
    x_bins, y_bins, data = compute(x, y, z, xmin, xmax, ymin, ymax, num_bins)
    xx, yy = np.meshgrid(x_bins, y_bins)
    
    if isconvolve:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kernel = np.ones((kernel_size, kernel_size))
            data = convolve(data, kernel, boundary='extend')
    return xx, yy, data


def trial_data(input):
    '''
        trial data.
        based on input, get agent data (w/wo mk data)

        input:(
                agent,                              nn 
                env, gym                            obj
                number of sampled trials,           int
                use_mk_data,                        bool
                task                                list
                mkdata:{
                    trial index                  int
                    task,                        list
                    actions,                     array/list of lists/none
                    states}                      array/list of lists/none
                    
        log (
                monkey trial index,                 int/none
                monkey control,                     
                monkey position (s),
                agent control,
                agent position (s),
                belief mu,
                cov,
                theta,
                )
    '''
    # prepare variables
    result={
    'theta':input['theta'],
    'phi':input['phi'],
    'task':[],
    'num_trials':input['num_trials'],
    'agent_actions':[],
    'agent_beliefs':[],
    'agent_covs':[],
    'agent_states':[],
    'use_mk_data':input['use_mk_data'],
    'mk_trial_index':None,
    'mk_actions':[],
    'mk_states':[],
    }
    # check
    env=input['env']
    action_std=0.1 if 'action_std' not in input else input['action_std']
    if input['use_mk_data'] and not input['mkdata']: #TODO more careful check
        warnings.warn('need to input monkey data when using monkey data')
    if 'task' not in input:
        warnings.warn('random targets')

    result['mk_trial_index']=input['mkdata']['trial_index']
    result['task']=[input['mkdata']['task'][i] for i in input['mkdata']['trial_index']]
    result['mk_actions']=[input['mkdata']['actions'][i] for i in input['mkdata']['trial_index']]
    result['mk_states']=[input['mkdata']['states'][i] for i in input['mkdata']['trial_index']]

    if 'task' not in input: # random some task
            for trial_i in range(input['num_trials']):
                distance=torch.ones(1).uniform_(0.3,1)
                angle=torch.ones(1).uniform_(-pi/5,pi/5)
                task=[(torch.cos(angle)*distance).item(),(torch.sin(angle)*distance).item()]
                result['task'].append(task)
    else:  # use the task given
            if len(input['task'])==2:
                result['task']=input['task']*input['num_trials']
            else:
                result['task']=input['task']

    with torch.no_grad():
        for trial_i in range(input['num_trials']):
            # print(result['task'][trial_i])
            if input['use_mk_data']: # reset based on mk init condition
                env.reset(phi=input['phi'],theta=input['theta'],
                goal_position=result['task'][trial_i],
                vctrl=result['mk_actions'][trial_i][0][0], 
                wctrl=result['mk_actions'][trial_i][0][1])
                # print('mk action', input['mkdata']['actions'][trial_i][0])
                # print('prev action',env.previous_action)
            else: # still reset based on mk init condition
                env.reset(phi=input['phi'],
                theta=input['theta'],
                goal_position=result['task'][trial_i],
                vctrl=result['mk_actions'][trial_i][0][0], 
                wctrl=result['mk_actions'][trial_i][0][1])

            # prepare trial var
            epbliefs=[]
            epbcov=[]
            epactions=[]
            epstates=[]
            if input['use_mk_data']: # using mk data
                t=0
                while t<len(result['mk_actions'][trial_i]):
                    noise=np.random.normal(0,action_std,size=(2))
                    action = (agent(env.decision_info)[0] +noise).float()
                    action.clamp_(-1,1)
                    if  t+1<result['mk_states'][trial_i].shape[0]:
                        env.step(torch.tensor(result['mk_actions'][trial_i][t]).reshape(1,-1),
                        next_state=result['mk_states'][trial_i][t+1].view(-1,1)) 
                    epactions.append(action)
                    epbliefs.append(env.b)
                    epbcov.append(env.P)
                    epstates.append(env.s)
                    t=t+1
                result['agent_beliefs'].append(torch.stack(epbliefs)) # the predicted belief of monkey
                result['agent_covs'].append(epbcov)   # the predicted uncertainty of monkey
                result['agent_actions'].append(torch.stack(epactions)) # the actions agent try to excute
                result['agent_states'].append(torch.stack(epstates)[:,:,0]) # will be same as mk states
            else: # not using mk data
                done=False
                t=0
                while not done:
                    noise=np.random.normal(0,action_std,size=(2))
                    action = (agent(env.decision_info)[0] +noise).float()
                    action.clamp_(-1,1)
                    if 'pert' in input['mkdata']:
                        action+=input['mkdata']['pert'][t]
                    _,_,done,_=env.step(torch.tensor(action).reshape(1,-1)) 
                    epactions.append(action)
                    epbliefs.append(env.b)
                    epbcov.append(env.P)
                    epstates.append(env.s)
                    t=+1
                result['agent_actions'].append(torch.stack(epactions))
                result['agent_beliefs'].append(torch.stack(epbliefs)[:,:,0])
                result['agent_covs'].append(epbcov)
                print(torch.stack(epstates))
                result['agent_states'].append(torch.stack(epstates))
    return result


def inverse_trajectory_monkey(theta_trajectory,
                    env=None,
                    agent=None,
                    phi=None, 
                    background_data=None, 
                    background_contour=False,
                    number_pixels=10,
                    background_look='contour',
                    ax=None, 
                    num_episodes=2,
                    loss_sample_size=2, 
                    H=None,
                    action_var=0.1,
                    **kwargs):
    '''    
        plot the inverse trajectory in 2d pc space
        -----------------------------
        input:
        theta trajectory: list of list(theta)
        method: PCA or other projections
        -----------------------------
        output:
        background contour array and figure
    '''
    with torch.no_grad():

        # plot trajectory
        fig = plt.figure(figsize=[3, 3])
        ax = fig.add_subplot()
        data_matrix=column_feature_data(theta_trajectory)
        mu=np.mean(data_matrix,0)
        try:
            score, evectors, evals = pca(data_matrix)
        except np.linalg.LinAlgError:
            score, evectors, evals = pca(data_matrix)


        row_cursor=0
        while row_cursor<score.shape[0]-1:
            row_cursor+=1
            ax.plot(score[row_cursor-1:row_cursor+1,0],
                    score[row_cursor-1:row_cursor+1,1],
                    '-',
                    linewidth=0.1,
                    color='g')
        # plot log likelihood contour
        sample_states, sample_actions, sample_tasks=sample_batch(states=states,actions=actions,tasks=tasks,batch_size=loss_sample_size)
        loss_function=monkey_loss_wrapped(env=env, 
        agent=agent, 
        phi=phi, 
        states=sample_states,
        actions=sample_actions,
        action_var=action_var,
        tasks=sample_tasks,
        num_episodes=num_episodes)
        finaltheta=score[row_cursor, 0], score[row_cursor, 1]
        current_xrange=list(ax.get_xlim())
        current_xrange[0]-=0.1
        current_xrange[1]+=0.1
        current_yrange=list(ax.get_ylim())
        current_yrange[0]-=0.1
        current_yrange[1]+=0.1
        maxaxis=max(abs(current_xrange[0]-finaltheta[0]),abs(current_xrange[1]-finaltheta[0]),abs(current_yrange[0]-finaltheta[1]),abs(current_yrange[1]-finaltheta[1]))
        xyrange=[[-maxaxis,maxaxis],[-maxaxis,maxaxis]]
        print('start background')
        if background_contour:
            background_data=plot_background(
                ax, 
                xyrange,
                mu, 
                evectors, 
                loss_function, 
                number_pixels=number_pixels,
                look=background_look,
                background_data=background_data)

        # plot theta inverse trajectory
        row_cursor=0
        while row_cursor<score.shape[0]-1:
            row_cursor+=1
            ax.plot(score[row_cursor-1:row_cursor+1,0],
                    score[row_cursor-1:row_cursor+1,1],
                    '-',
                    linewidth=0.5,
                    color='w') # line
            if row_cursor%20==0 or row_cursor==1:
                ax.quiver(score[row_cursor-1, 0], score[row_cursor-1, 1],
                        score[row_cursor, 0]-score[row_cursor-1, 0], score[row_cursor, 1]-score[row_cursor-1, 1],
                        angles='xy',color='w',scale=0.2,width=1e-2, scale_units='xy') # arrow
        ax.scatter(score[row_cursor, 0], score[row_cursor, 1], marker=(5, 1), s=200, color=[1, .5, .5])
        ax.set_xlabel('projected parameters')
        ax.set_ylabel('projected parameters')
        ax.set_xticks([-0.3,0,0.3])
        ax.set_yticks([-0.3,0,0.3])
        # plot hessian
        if H is not None:
            cov=theta_cov(H)
            cov_pc=evectors[:,:2].transpose()@np.array(cov)@evectors[:,:2]
            plot_cov_ellipse(cov_pc,pos=score[-1,:2],alpha_factor=0.5,ax=ax)

        return background_data


def plot_background(
    ax, 
    xyrange,
    mu, 
    evectors, 
    loss_function, 
    number_pixels=10, 
    look='contour', 
    alpha=1,
    background_data=None,
    **kwargs):
    with torch.no_grad():
        X,Y=np.meshgrid(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels),np.linspace(xyrange[1][0],xyrange[1][1],number_pixels))
        if background_data is None:
            background_data=np.zeros((number_pixels,number_pixels))
            for i,u in enumerate(np.linspace(xyrange[1][0],xyrange[1][1],number_pixels)):
                for j,v in enumerate(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels)):
                    score=np.array([u,v])
                    reconstructed_theta=score@evectors.transpose()[:2,:]+mu
                    reconstructed_theta=torch.tensor(reconstructed_theta).float()
                    reconstructed_theta.clamp(0.001,3)
                    background_data[i,j]=loss_function(reconstructed_theta)
                    print(background_data[i,j],reconstructed_theta)
        if look=='contour':
            c=ax.contourf(X,Y,background_data.transpose(),alpha=alpha,zorder=1)
            plt.colorbar(c,ax=ax)
        # elif look=='pixel':
        #     im=ax.imshow(X,Y,background_data,alpha=alpha,zorder=1)
        #     add_colorbar(im)
        return background_data


# TODO
def plot_background_par(
    ax, 
    xyrange,
    mu, 
    evectors, 
    loss_function, 
    number_pixels=10, 
    look='contour', 
    alpha=1,
    background_data=None,
    **kwargs):

    def _cal(u,v,i,j):
        with torch.no_grad():
            score=np.array([u,v])
            reconstructed_theta=score@evectors.transpose()[:2,:]+mu
            reconstructed_theta=torch.tensor(reconstructed_theta).float()
            reconstructed_theta.clamp(0.001,3)
            background_data[i,j]=loss_function(reconstructed_theta)

    def _cal(u,v,i,j,background_data):
        print([u,v,i,j])


    X,Y=np.meshgrid(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels),np.linspace(xyrange[1][0],xyrange[1][1],number_pixels))
    if background_data is None:
        background_data=np.zeros((number_pixels,number_pixels))
        pool = multiprocessing.Pool(4)
        jobparam=[(u,v,i,j,background_data) for (i,u),(j,v) in zip(enumerate(np.linspace(xyrange[1][0],xyrange[1][1],number_pixels)),enumerate(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels)))]
        pool.map_async(_cal, (job for job in jobparam)).get(timeout=10)

            
        if look=='contour':
            c=ax.contourf(X,Y,background_data.transpose(),alpha=alpha,zorder=1)
            fig.colorbar(c,ax=ax)
    return background_data


    pool = multiprocessing.Pool(4)
    jobs=list(range(10))
    def _cal(x):
        return x**2
    res=pool.map_async(_cal, (job for job in jobs))
    w = sum(res.get(timeout=10))
    print(w)

    import multiprocessing
    def start_process():
        print ('Starting', multiprocessing.current_process().name)

    pool_size =2
    pool = multiprocessing.Pool(processes=pool_size,
                                initializer=start_process,
                                )

    pool_outputs = pool.map(_cal,jobs)
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks


def monkey_loss_wrapped(
                env=None, 
                agent=None, 
                phi=None, 
                actions=None,
                tasks=None,
                states=None,
                num_episodes=10,
                gpu=False,
                action_var=0.001,
                **kwargs):
  new_function= lambda theta_estimation: monkeyloss(
            agent=agent, 
            actions=actions, 
            tasks=tasks, 
            phi=phi, 
            theta=theta_estimation, 
            env=env,
            num_iteration=1, 
            states=states, 
            samples=num_episodes, 
            action_var=action_var,
            gpu=gpu)
    
  return new_function


def plot_background_ev(
    ax, 
    evector,
    loss_function, 
    number_pixels=10, 
    alpha=0.5,
    scale=1.,
    background_data=None,
    **kwargs):
    ev1=evector[:,0].view(-1,1)
    ev2=evector[:,1].view(-1,1)
    if background_data is None:
        background_data=np.zeros((number_pixels,number_pixels))
        with torch.no_grad():
            X=np.linspace(theta-ev1*0.5*scale,theta+ev1*0.5*scale,number_pixels)
            Y=np.linspace(theta-ev2*0.5*scale,theta+ev2*0.5*scale,number_pixels)
            for i in range(number_pixels):
                for j in range(number_pixels):
                        reconstructed_theta=theta+X[i]+Y[j]
                        if torch.all(reconstructed_theta>0):
                            background_data[i,j]=loss_function(reconstructed_theta)
                        else:
                            background_data[i,j]=None
            plt.contourf(background_data,alpha=alpha)
        return background_data


def is_skip(task,state,dthreshold=0.26,sthreshold=None):
    if sthreshold:
        return (dthreshold<torch.norm(torch.tensor(task)-state[:2,-1]) and 
            sthreshold>torch.norm(torch.tensor(task)-state[:2,-1]))
    else:
        return dthreshold<torch.norm(torch.tensor(task)-state[:2,-1])


def plotctrl(input,**kwargs):
    labels=[' v', ' w']
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if 'ax' in kwargs:
            ax = kwargs['ax'] 
        else:
            ax = fig.add_subplot(111)
        if 'color' in kwargs:
            color=kwargs['color']
        else:
            color=[color_settings['v'],color_settings['w']]
        prefix=kwargs['prefix'] if 'prefix' in kwargs else ''
        # ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time, dt')
        ax.set_ylabel('control magnitude')
        # plot data
        for trial_i in range(input['num_trials']):
            ax.plot(input['agent_actions'][trial_i][:,0],color=color[0],alpha=1/input['num_trials']**.3, label=prefix+labels[0])
            ax.plot(input['agent_actions'][trial_i][:,1],color=color[1],alpha=1/input['num_trials']**.3, label=prefix+labels[1])
        # legend and label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),loc='upper right')
        ax.set_xlim(left=0.)
        ax.set_ylim(-1,1)
        ax.set_yticks([-1,0,1])
        return  ax


def plotctrl_mk(indls,actions=None,**kwargs):
    labels=[' v', ' w']
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if 'ax' in kwargs:
            ax = kwargs['ax'] 
        else:
            ax = fig.add_subplot(111)
        if 'color' in kwargs:
            color=kwargs['color']
        else:
            color=[color_settings['v'],color_settings['w']]
        prefix=kwargs['prefix'] if 'prefix' in kwargs else ''
        # ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time, dt')
        ax.set_ylabel('control magnitude')
        # plot data
        for trial_i in indls:
            ax.plot(actions[trial_i][1:,0],color=color[0],alpha=1/len(indls)**.3, label=prefix+labels[0])
            ax.plot(actions[trial_i][1:,1],color=color[1],alpha=1/len(indls)**.3, label=prefix+labels[1])
        # legend and label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),loc='upper right')
        ax.set_xlim(left=0.)
        ax.set_ylim(-1,1)
        ax.set_yticks([-1,0,1])
        return  ax


#----overhead----------------------------------------------------------------
def plotoverhead(input,alpha=0.8,**kwargs):
    '''
        input:
        'theta':            input['theta'],
        'phi':              input['phi'],
        'agent_actions':    [],
        'agent_beliefs':    [],
        'agent_covs':       [],
        'agent_states':     [],
        'use_mk_data':      input['use_mk_data'],
        'mk_trial_index':   None,
        'mk_actions':       [],
        'mk_states':        [],
    '''
    pathcolor='gray' if 'color' not in kwargs else kwargs['color'] 
    fontsize = 9
    
    if 'ax' in kwargs:
        ax = kwargs['ax'] 
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
        goal=plt.Circle(np.array(input['task'][0])@np.array([[0,1],[-1,0]])*400,65,color=color_settings['goal'], alpha=alpha,label='target')
        ax.add_patch(goal)
        for trial_i in range(input['num_trials']):
            input['agent_states'][trial_i][:,0]=input['agent_states'][trial_i][:,0]-input['agent_states'][trial_i][0,0]
            input['agent_states'][trial_i][:,1]=input['agent_states'][trial_i][:,1]-input['agent_states'][trial_i][0,1]
            ax.plot(-input['agent_states'][trial_i][:,1]*400,input['agent_states'][trial_i][:,0]*400, 
                c=pathcolor, lw=0.1, ls='-')
            # calculate if rewarded
            # dy=input['agent_states'][trial_i][-1,0]*400-input['task'][trial_i][0]*400
            # dx=input['agent_states'][trial_i][-1,1]*400-input['task'][trial_i][1]*400
            # d2goal=(dx**2+dy**2)**0.5
            # if d2goal<=65: # reached goal
            #     ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
            #         c='g', marker='.', s=1, lw=1)
            # elif d2goal>65 and d2goal< 130:
            #     ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
            #         c='k', marker='.', s=1, lw=1)
            #     # yellow error line
            #     ax.plot([-input['agent_states'][trial_i][-1,1]*400,-input['task'][trial_i][1]*400],
            #                 [input['agent_states'][trial_i][-1,0]*400,input['task'][trial_i][0]*400],color='yellow',alpha=0.3, linewidth=1)

            # else: # skipped
            #     ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
            #         c='r', marker='.', s=1, lw=1)

    else:
        with initiate_plot(1.8, 1.8, 300) as fig:
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
            ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
            x_temp = np.linspace(-235, 235)
            ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
            ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
            ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
            ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
            ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
            ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
            ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)
            fig.tight_layout(pad=0)
            ax.plot(np.linspace(0, 230 + 7),
                    np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
            goal=plt.Circle(np.array(input['task'][0])@np.array([[0,1],[-1,0]])*400,65,color=color_settings['goal'], alpha=alpha,label='target')
            ax.add_patch(goal)
            
            for trial_i in range(input['num_trials']):
                input['agent_states'][trial_i][:,0]=input['agent_states'][trial_i][:,0]-input['agent_states'][trial_i][0,0]
                input['agent_states'][trial_i][:,1]=input['agent_states'][trial_i][:,1]-input['agent_states'][trial_i][0,1]
                ax.plot(-input['agent_states'][trial_i][:,1]*400,input['agent_states'][trial_i][:,0]*400, 
                    c=pathcolor, lw=0.1, ls='-')
                # # calculate if rewarded
                # dy=input['agent_states'][trial_i][-1,0]*400-input['task'][trial_i][0]*400
                # dx=input['agent_states'][trial_i][-1,1]*400-input['task'][trial_i][1]*400
                # d2goal=(dx**2+dy**2)**0.5
                # if d2goal<=65: # reached goal
                #     ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                #         c='g', marker='.', s=1, lw=1)
                # elif d2goal>65 and d2goal< 130:
                #     ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                #         c='k', marker='.', s=1, lw=1)
                #     # yellow error line
                #     ax.plot([-input['agent_states'][trial_i][-1,1]*400,-input['task'][trial_i][1]*400],
                #                 [input['agent_states'][trial_i][-1,0]*400,input['task'][trial_i][0]*400],color='yellow',alpha=0.3, linewidth=1)

                # else: # skipped
                #     ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                #         c='r', marker='.', s=1, lw=1)
    return ax


def plotoverheadhuman(indls,states,tasks,alpha=0.8,fontsize=5,ax=None,color=color_settings['s'], label=None):
    if ax:
        for trial_i in indls:
            xs=states[trial_i][:,0]
            ys=states[trial_i][:,1]
            ax.plot(-ys*400,xs*400, c=color, lw=1, ls='-',label=label,alpha=alpha)
        return ax
    with initiate_plot(3, 3, 300) as fig:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        # ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
        x_temp = np.linspace(-235*3, 235*3)
        ax.plot(x_temp, np.sqrt(1200**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$6 m$', fontsize=fontsize)
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')

        ax.text(-200, 100, s=r'$1 m$', fontsize=fontsize)
        ax.text(-100, 0, s=r'$1 m$', fontsize=fontsize)
        # ax.plot(np.linspace(0, 1200 ),
        #         np.tan(np.deg2rad(55)) * np.linspace(0, 600) - 10, c='k', ls=':')
                
        goal=plt.Circle(np.array(tasks[indls[0]])@np.array([[0,1],[-1,0]])*400,65,color=color_settings['goal'], edgecolor='none',alpha=alpha,label='target')
        ax.add_patch(goal)
        for trial_i in indls:
            xs=states[trial_i][:,0]
            ys=states[trial_i][:,1]
            ax.plot(-ys*400,xs*400, c=color, lw=1, ls='-',label=label,alpha=alpha)
    return ax
            

def plotoverheadhuman_compare(data1,data2,alpha=0.8,fontsize=5):
    with initiate_plot(3, 3, 300) as fig:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        # ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
        x_temp = np.linspace(-235*3, 235*3)
        ax.plot(x_temp, np.sqrt(1200**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$6 m$', fontsize=fontsize)
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')

        ax.text(-200, 100, s=r'$1 m$', fontsize=fontsize)
        ax.text(-100, 0, s=r'$1 m$', fontsize=fontsize)
        # ax.plot(np.linspace(0, 1200 ),
        #         np.tan(np.deg2rad(55)) * np.linspace(0, 600) - 10, c='k', ls=':')
                
        goal=plt.Circle(np.array(data1['task'][0])@np.array([[0,1],[-1,0]])*400,65,color=color_settings['goal'], facecolor='none',alpha=alpha,label='target')
        ax.add_patch(goal)
        for trial_i in range(data1['num_trials']):
            data1['agent_states'][trial_i][:,0]=data1['agent_states'][trial_i][:,0]-data1['agent_states'][trial_i][0,0]
            data1['agent_states'][trial_i][:,1]=data1['agent_states'][trial_i][:,1]-data1['agent_states'][trial_i][0,1]
            ax.plot(-data1['agent_states'][trial_i][:,1]*400,data1['agent_states'][trial_i][:,0]*400, c=color_settings['s'], lw=1, ls='-')
            data2['agent_states'][trial_i][:,0]=data2['agent_states'][trial_i][:,0]-data2['agent_states'][trial_i][0,0]
            data2['agent_states'][trial_i][:,1]=data2['agent_states'][trial_i][:,1]-data2['agent_states'][trial_i][0,1]
            ax.plot(-data2['agent_states'][trial_i][:min(data2['agent_states'][trial_i][:,1].shape[0],data1['agent_states'][trial_i][:,1].shape[0])-1,1]*400,data2['agent_states'][trial_i][:min(data2['agent_states'][trial_i][:,1].shape[0],data1['agent_states'][trial_i][:,1].shape[0])-1,0]*400, c=color_settings['model'], lw=1, ls='-')
            

def plotoverhead_mk(indls, alpha=0.3,**kwargs):
    pathcolor='gray' if 'color' not in kwargs else kwargs['color'] 
    fontsize = 9
    if 'ax' in kwargs:
        ax = kwargs['ax'] 
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
        for trial_i in indls:
            ax.plot(-states[trial_i][:,1]*400,states[trial_i][:,0]*400, 
                c=pathcolor, lw=0.1, ls='-')
    
    else:
        with initiate_plot(1.8, 1.8, 300) as fig:
            ax = fig.add_subplot(111)
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
            ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
            x_temp = np.linspace(-235, 235)
            ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
            ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
            ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
            ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
            ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
            ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
            ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)
            fig.tight_layout(pad=0)
            ax.plot(np.linspace(0, 230 + 7),
                    np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
            goal=plt.Circle(np.array(tasks[indls[0]])@np.array([[0,1],[-1,0]])*400,65,color=color_settings['goal'], alpha=alpha,label='target')
            ax.add_patch(goal)
            
            for trial_i in indls:
                ax.plot(-states[trial_i][:,1]*400,states[trial_i][:,0]*400, 
                c=pathcolor, lw=0.1, ls='-')

    return ax


def plotoverheadcolor(input,**kwargs):
    linewidth=kwargs['linewidth'] if 'linewidth' in kwargs else 0.1
    alpha=kwargs['alpha']  if 'alpha' in kwargs else 1
    fontsize = 9
    target_idexes = np.arange(1500, 2000)
    if 'ax' in kwargs:
        ax = kwargs['ax'] 
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
        # setting color                
        colors = pl.cm.Set1(np.linspace(0,1,input['num_trials']))
        for trial_i in range(input['num_trials']):
            ax.plot(-input['agent_states'][trial_i][:,1]*400,input['agent_states'][trial_i][:,0]*400, 
                color=colors[trial_i], lw=0.1, ls='-',linewidth=linewidth,alpha=alpha)
            ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                color=colors[trial_i], marker='.', s=1, lw=1)

        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)

    else:
        with initiate_plot(1.8, 1.8, 300) as fig:
            ax = fig.add_subplot(111) 
            ax.set_aspect('equal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
            ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
            
            ax.plot(np.linspace(0, 230 + 7),
                    np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
            # setting color                
            colors = pl.cm.Set1(np.linspace(0,1,input['num_trials']))
            for trial_i in range(input['num_trials']):
                ax.plot(-input['agent_states'][trial_i][:,1]*400,input['agent_states'][trial_i][:,0]*400, 
                    color=colors[trial_i], lw=0.1, ls='-',linewidth=linewidth,alpha=alpha)
                ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                    color=colors[trial_i], marker='.', s=1, lw=1)

            x_temp = np.linspace(-235, 235)
            ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
            ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
            ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
            
            ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
            ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
            ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
            ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)

            fig.tight_layout(pad=0)
    return  ax


def plotoverhead_skip(input):
    '''
        input:
        'theta':            input['theta'],
        'phi':              input['phi'],
        'agent_actions':    [],
        'agent_beliefs':    [],
        'agent_covs':       [],
        'agent_states':     [],
        'use_mk_data':      input['use_mk_data'],
        'mk_trial_index':   None,
        'mk_actions':       [],
        'mk_states':        [],
    '''
    fontsize = 9
    target_idexes = np.arange(1500, 2000)
    with initiate_plot(1.8, 1.8, 300) as fig:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
        
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
        
        for trial_i in range(input['num_trials']):
            ax.plot(-input['agent_states'][trial_i][:,1]*400,input['agent_states'][trial_i][:,0]*400, 
                c='gray', lw=0.1, ls='-')
            # calculate if rewarded
            dx=input['agent_states'][trial_i][-1,1]*400-input['task'][trial_i][1]*400
            dy=input['agent_states'][trial_i][-1,0]*400-input['task'][trial_i][0]*400
            d2goal=(dx**2+dy**2)**0.5
            d2start=(input['agent_states'][trial_i][-1,1]**2+input['agent_states'][trial_i][-1,0]**2)**0.5*400
            if d2goal<65:
                ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                    c='k', marker='.', s=1, lw=1)
            elif d2goal>200: # skipped
                ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                    c='r', marker='.', s=1, lw=1)
            else:
                ax.scatter(-input['task'][trial_i][1]*400, input['task'][trial_i][0]*400, 
                    c='k', marker='.', s=1, lw=1)
  
        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)

        fig.tight_layout(pad=0)

# change theta in direction of mattered most eigen vector 
def matter_most_eigen_direction():
    theta_mean=torch.tensor([[0.5],
            [1.57],
            [0.5],
            [0.5],
            [0.5],
            [0.5],
            [0.13],
            [0.5],
            [0.5],
            [0.5],
            [0.5],
    ])
    mattermost=evector[0]
    # matterleast=evector[-1]
    diagnose_plot_theta(agent, env, phi, theta_init, theta_mean+0.1*mattermost.view(-1,1),5)
    # diagnose_plot_theta(agent, env, phi, theta_init, theta_mean+3*matterleast.view(-1,1),3)

# describes data df
def histdfcol():
    seen={}
    for each in df.floor_density:
        if each in seen:
            seen[each]+=1
        else:
            seen[each]=1

    def itkey(keys):
        for key in keys:
            yield key

    keys=['gain_v',
    'gain_w',
    'perturb_vpeakmax',
    'perturb_wpeakmax',
    'perturb_sigma',
    'perturb_dur',
    'perturb_vpeak',
    'perturb_wpeak',
    'perturb_start_time',
    'floor_density',
    'pos_r_end',
    'pos_theta_end',
    'target_x',
    'target_y',
    'target_r',
    'target_theta',
    'full_on',
    'rewarded',
    'trial_dur',
    'relative_radius_end',
    'relative_angle_end',
    'category',]

    keygen=itkey(keys)

    key=keygen.__next__()

    plt.hist(df[key])
    plt.title(key)
    plt.show()


def colorgrad_line(x,y,z,ax=None, linewidth=2):
    x,y,z=np.array(x),np.array(y),np.array(z)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors=np.array(z)
    if ax:
        lc = LineCollection(segments, cmap='viridis')
        lc.set_array(colors)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)
        plt.colorbar(line,ax=ax)
    else:
        fig, ax = plt.subplots()
        lc = LineCollection(segments, cmap='viridis')
        lc.set_array(colors)
        lc.set_linewidth(linewidth)
        line = ax.add_collection(lc)
        fig.colorbar(line,ax=ax)
    return ax


def colorgrad_inverse_traj(inverse_data):
        # plot trajectory
        fig = plt.figure(figsize=[3, 3])
        ax = fig.add_subplot()
        data_matrix=column_feature_data(inverse_data['theta_estimations'])
        mu=np.mean(data_matrix,0)
        try:
            score, evectors, evals = pca(data_matrix)
        except np.linalg.LinAlgError:
            score, evectors, evals = pca(data_matrix)
        x=score[:,0]
        y=score[:,1]
        z=inverse_data['loss']
        z=np.array(z)*-1
        z= (z-min(z))/(max(z)-min(z))
        # z=np.log(z)
        ax.set_xlim(min(x),max(x))
        ax.set_ylim(min(y),max(y))
        colorgrad_line(x,y,z,ax=ax)
        ax.set_xlabel('projected parameters')
        ax.set_ylabel('projected parameters')
        ax.set_title('inverse trajectory, color for likelihood')

# inferred obs vs density
def obsvsdensity(noiseparam):
    errbar=False
    x,y=[],[]
    err=[]
    for k,v in noiseparam.items():
        x.append(k)
        y.append(v['param'])
        if 'std' in v:
            err.append(v['std'])
            errbar=True
    y=np.array(y)[:,:,0]
    err=np.array(err)
    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    lables=['pro v','pro w','obs v','obs w']
    if errbar:
        for i in range(4):
            ax.errorbar( [i for i in range(4)],y[:,i],err[:,i],label=lables[i])
    else:
        for i in range(4):
            ax.plot(y[:,i],label=lables[i])
    ax.set_xticks([i for i in range(4)])
    ax.set_xticklabels(["0.0001", "0.0005", "0.001", "0.005"])
    ax.legend(fontsize=16)
    ax.set_ylabel('noise std',fontsize=16)
    ax.set_xlabel('density',fontsize=16)
    ax.set_title('inferred noise level vs ground density',fontsize=20)

    fig=plt.figure(figsize=(5,5))
    ax=fig.add_subplot(111)
    ax.plot(y[:,0]/y[:,2],label='forward v')
    ax.plot(y[:,1]/y[:,3],label='angular w')
    # ax.set_xscale('log')
    ax.set_xticks([i for i in range(4)])
    ax.set_xticklabels(["0.0001", "0.0005", "0.001", "0.005"])
    ax.legend(fontsize=16)
    ax.set_ylabel('noise std',fontsize=16)
    ax.set_xlabel('density',fontsize=16)
    ax.set_title('inferred observation reliable degree vs ground density',fontsize=20)


def plotpert_fill(v,pertv, w, pertw, ax=None,alpha=0.8):
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if ax is None:
            ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time, dt')
        ax.set_ylabel('perturbed control magnitude')
        # process data
        x=list(range(len(v)))
        # plot data
        ax.axhline(y=0., color='black', linestyle='--',alpha=alpha*0.5,linewidth=0.5)

        ax.fill_between(x, v, pertv,color=color_settings['v'],alpha=alpha, label='foward v')
        ax.fill_between(x, w, pertw,color=color_settings['w'],alpha=alpha, label='angular w')
        # legend and label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # ax.set_xlim(0,40)
        # ax.set_ylim(-1,1)
        return  ax


def plotpertfillwrapper():
    v=actions[ind][:,0][1:]
    w=actions[ind][:,1][1:]
    pertv=down_sampling(df.iloc[ind].perturb_v/200,0.0012,0.1)
    pertw=down_sampling(df.iloc[ind].perturb_w/180*pi,0.0012,0.1)
    v,w,pertv,pertw=torch.tensor(v),torch.tensor(w),torch.tensor(pertv).float(),torch.tensor(pertw).float()
    overlap=min(len(v),len(pertv))
    pertv=pertv[:overlap] +v[:overlap]
    pertw=pertw[:overlap] +w[:overlap]
    v,w=v[:overlap],w[:overlap]
    plotpert_fill(v,pertv, w, pertw, ax=None,alpha=0.8)


def get_stops(states, tasks, thistask, env, theta, agent,pert=None):
    indls=similar_trials2this(tasks,thistask)
    # get the stops from monkey data
    mkstops=[]
    for i in indls:
        mkstops.append(states[i][-1][:2].view(-1).tolist())
    mkstops=np.array(mkstops)

    # sample stop distribution from agent doing these tasks
    ircstops=[]
    for i in indls:
        env.reset(theta=theta, phi=phi, goal_position=tasks[i])
        if pert is not None:
            env.obs_traj= np.vstack([np.array([0,0]),np.random.normal(0,1,size=pert.shape)*np.asfarray(theta[4:6]).T + pert,np.zeros(shape=(20,2))])
        _,_,_,epstates=run_trial(agent,env,given_action=None, given_state=None, action_noise=0.1,pert=pert)
        stop=epstates[-1][:2].view(-1).tolist()
        ircstops.append(stop)
    ircstops=np.array(ircstops)
    
    return mkstops,ircstops


def run_trial(agent=None,env=None,given_action=None, given_state=None, action_noise=0.1,pert=None):
    # 10 a 10 s. 
    # when both
    # use a1 and s2
    # at t1, use a1. results in s2

    def _collect():
        epactions.append(action)
        epbliefs.append(env.b)
        epbcov.append(env.P)
        epstates.append(env.s)
    # saves
    epactions,epbliefs,epbcov,epstates=[],[],[],[]
    with torch.no_grad():
        # if at least have something
        if given_action is not None and given_state is not None: # have both
            t=0
            while t<len(given_state):
                action = agent(env.decision_info)[0]   
                _collect()           
                env.step(torch.tensor(given_action[t]).reshape(1,-1),next_state=given_state[t].view(-1,1)) 
                t+=1
        elif given_state is not None: # have states but no actions
            t=0
            while t<len(given_state):
                action = agent(env.decision_info)[0]
                _collect()
                env.step(torch.tensor(action).reshape(1,-1),next_state=given_state[t].view(-1,1)) 
                t+=1

        elif given_action is not None: # have actions but no states
            t=0
            while t<len(given_action):
                action = agent(env.decision_info)[0]
                _collect()
                env.step(torch.tensor(given_action[t]).reshape(1,-1).reshape(1,-1)) 
                t+=1

        else:  # nothing
            done=False
            t=0
            while not done:
                action = agent(env.decision_info)[0]
                _collect()
                noise=torch.normal(torch.zeros(2),action_noise)
                _action=(action+noise).clamp(-1,1)
                if pert is not None and int(env.trial_timer)<len(pert):
                    _action+=pert[int(env.trial_timer)]
                _,_,done,_=env.step(torch.tensor(_action).reshape(1,-1)) 
                t+=1
    return epactions,epbliefs,epbcov,epstates


def plotpert(v, w, ax=None,alpha=0.8):
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if ax is None:
            ax = fig.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time, dt')
        ax.set_ylabel('pert magnitude')
        # process data
        x=list(range(len(v)))
        # plot data
        ax.axhline(y=0., color='black', linestyle='--',alpha=alpha*0.5,linewidth=0.5)

        ax.plot(x, v,color=color_settings['v'],alpha=alpha, label='foward v')
        ax.plot(x, w,color=color_settings['w'],alpha=alpha, label='angular w')
        # legend and label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # ax.set_xlim(0,40)
        # ax.set_ylim(-1,1)
        return  ax


def quickoverhead(statelike,ax=None,alpha=0.5):
  with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(111) if not ax else ax
    # ax.plot(given_state[:,0],given_state[:,1])
    for key,value in statelike.items():
      ax.plot(torch.cat(value,1).t()[:,0],torch.cat(value,1).t()[:,1], label=key)
    # ax.plot(torch.cat(epstates,1).t()[:,0],torch.cat(epstates,1).t()[:,1], label='state')
    # ax.plot(torch.cat(epbliefs,1).t()[:,0],torch.cat(epbliefs,1).t()[:,1],label='belief')
    goal=plt.Circle(tasks[ind],0.13,color=color_settings['goal'], alpha=alpha,label='target')
    ax.add_patch(goal)
    ax.axis('equal')
    ax.set_xlabel('world x')
    ax.set_ylabel('world y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # legend and label
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),loc=2, prop={'size': 6})


def theta_bar(finaltheta,finalcov=None,xlabels=theta_names,err=None, ax=None, label=None,shift=0,width=0.5, color=None):
    if finalcov is None: err=None
    else: err=torch.diag(finalcov)**0.5 if err is None else err
    with initiate_plot(12, 4, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if ax is None:
            ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar([i+shift for i in range(len(finaltheta))], finaltheta.view(-1),width,yerr=err,label=label, color=color)
        # title and axis names
        ax.set_ylabel('inferred parameter value')
        ax.set_xticks([i for i in range(len(xlabels))])
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
    return ax


def firstevector_bar(finaltheta,xlabels=theta_names, ax=None, label=None,shift=0,width=0.5):
    
    with initiate_plot(6, 4, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if ax is None:
            ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar([i+shift for i in range(len(xlabels))], finaltheta,width,label=label)
        # title and axis names
        ax.set_ylabel('most constrained eigen vector parameter value')
        ax.set_xticks([i for i in range(len(xlabels))])
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
    return ax


def thetaconfhist(finalcov):
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar([i for i in range(finalcov.shape[0])], torch.diag(finalcov)**0.5*1/torch.tensor(theta_mean), color = 'tab:blue')
        # title and axis names
        ax.set_ylabel('inferred parameter uncertainty (std/mean)')
        ax.set_xticks([i for i in range(finalcov.shape[0])])
        ax.set_xticklabels(theta_names, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


import matplotlib.ticker as mticker
sf = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(sf._formatSciNotation('%1e' % x))
fmt = mticker.FuncFormatter(g)


def d_process(logs,densities):
    means=[log[-1][0]._mean if log else None for log in logs]
    stds=[np.diag(log[-1][0]._C)**0.5 if log else None for log in logs]
    xs,ys,errs=[],[],[]
    for i in range(len(logs)):
        if means[i] is not None and stds[i] is not None and densities[i] is not None:
            x,y,err=densities[i],means[i],stds[i]
            xs.append(x);ys.append(y);errs.append(err)
    xs,ys,errs=torch.tensor(xs).float(),torch.tensor(ys).float(),torch.tensor(errs).float()
    return xs,ys,errs


def d_noise(xs,ys,errs):
    lables=['pro v','pro w','obs v','obs w']
    with initiate_plot(2.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.set_xticks([i for i in range(len(xs))])
        ax.set_xticklabels([fmt(i.item()) for i in xs])
        ax.set_ylabel('noise std',fontsize=12)
        ax.set_xlabel('density',fontsize=12)
        ax.set_title('inferred noise level vs ground density',fontsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for i in range(4):
            ax.errorbar( [i for i in range(len(xs))],ys[:,i],errs[:,i],label=lables[i], alpha=0.7)
        ax.legend(fontsize=8)


def d_kalman_gain(xs,ys,errs,conf=20,alpha=0.9):
    with initiate_plot(3, 3, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)

        # angular w
        pn_sample=np.random.normal(ys[:,3],errs[:,3],size=(100,ys.shape[0])).clip(1e-3,)
        on_sample=np.random.normal(ys[:,5],errs[:,5],size=(100,ys.shape[0])).clip(1e-3,)
        k_sample=pn_sample**2/(pn_sample**2+on_sample**2)
        mu=np.array([np.mean(k_sample[:,i]) for i in range(k_sample.shape[1])])
        lb=[np.percentile(k_sample[:,i],conf) for i in range(k_sample.shape[1]) ]
        hb=[np.percentile(k_sample[:,i],100-conf) for i in range(k_sample.shape[1]) ]
        cibound=np.array([lb,hb])
        err=np.array([ np.abs(mu.T-cibound[0,:] ), np.abs(cibound[1,:]-mu.T) ] )
        ax.errorbar(np.arange(len(xs))-0.1,y=mu,yerr=err,label='angular w',alpha=alpha)

        # forward v
        pn_sample=np.random.normal(ys[:,2],errs[:,2],size=(100,ys.shape[0])).clip(1e-3,)
        on_sample=np.random.normal(ys[:,4],errs[:,4],size=(100,ys.shape[0])).clip(1e-3,)
        k_sample=pn_sample**2/(pn_sample**2+on_sample**2)
        mu=np.array([np.mean(k_sample[:,i]) for i in range(k_sample.shape[1])])
        lb=[np.percentile(k_sample[:,i],conf) for i in range(k_sample.shape[1]) ]
        hb=[np.percentile(k_sample[:,i],100-conf) for i in range(k_sample.shape[1]) ]
        cibound=np.array([lb,hb])
        err=np.array([ np.abs(mu.T-cibound[0,:] ), np.abs(cibound[1,:]-mu.T) ] )
        ax.errorbar(np.arange(len(xs))+0.1,y=mu,yerr=err,label='forward v',alpha=alpha)

        # ax.hlines(y=0.5, xmin=0, xmax=len(xs)-1,linestyle='--', linewidth=2, color='black',alpha=0.8)
        # ax.plot(ys[:,2]/ys[:,4],label='forward v')
        # ax.plot(ys[:,2]**2/(ys[:,2]**2+ys[:,4]**2),label='forward v')
        # ax.plot(ys[:,3]**2/(ys[:,3]**2+ys[:,5]**2),label='angular w')
        # ax.plot(ys[:,3]/ys[:,5],label='angular w')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.set_yscale('log')
        ax.set_yticks([0.,0.5,1.0])
        ax.set_xticks(list(range(len(xs))))
        ax.set_xticklabels([fmt(i.item()) for i in xs])
        ax.legend(fontsize=12)
        ax.set_ylabel('Kalman gain',fontsize=12)
        ax.set_xlabel('density',fontsize=12)
        # ax.set_title('observation reliable degree',fontsize=16)


def barpertacc(accs,trialtype, ax=None, label=None,shift=0,width=0.4):
    with initiate_plot(3, 3, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if ax is None:
            ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar([i+shift for i in range(len(accs))], accs,width,label=label)
        # title and axis names
        ax.set_ylabel('trial reward rate')
        ax.set_xticks([i for i in range(len(accs))])
        ax.set_xticklabels(trialtype, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
    return ax


def barpertacc(accs,trialtype, ax=None, label=None,shift=0,width=0.4):
    with initiate_plot(3, 3, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if ax is None:
            ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar([i+shift for i in range(len(accs))], accs,width,label=label)
        # title and axis names
        ax.set_ylabel('radial error')
        ax.set_xticks([i for i in range(len(accs))])
        ax.set_xticklabels(trialtype, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
    return ax


def mybar(data,xlabels, ax=None, label=None,shift=0,width=0.4):
    with initiate_plot(3, 3, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if ax is None:
            ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar([i+shift for i in range(len(data))], data,width,label=label)
        # title and axis names
        ax.set_ylabel('radial error')
        ax.set_xticks([i for i in range(len(data))])
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
    return ax


def gauss_div(m1,s1,m2,s2):
    newm=1/(1/s1-1/s2)*(m1/s1-m2/s2)
    news=1/(1/s1-1/s2)
    return newm, news


def dfoverhead_single(df,ind=None,alpha=0.5):
    ind=np.random.randint(0,len(df)) if not ind else ind
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.plot(df.iloc[ind].pos_x,df.iloc[ind].pos_y, label='path')
        goal=plt.Circle([df.iloc[ind].target_x,df.iloc[ind].target_y],65,facecolor=color_settings['goal'],edgecolor='none', alpha=alpha,label='target')
        ax.plot(0.,0., "*", color='black',label='start')
        ax.add_patch(goal)
        ax.axis('equal')
        ax.set_xlabel('world x [cm]')
        ax.set_ylabel('world y [cm]')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # legend and label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),loc=2, prop={'size': 6})


def dfctrl_single(df,ind=None,alpha=0.5):
    ind=np.random.randint(0,len(df)) if not ind else ind
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ts=np.arange(len(df.iloc[ind].action_v))/10
        ax.plot(ts,df.iloc[ind].action_v, label='forward v control')
        ax.plot(ts,df.iloc[ind].action_w, label='angular w control')
        # ax.plot(0.,0., "*", color='black',label='start')
        ax.set_xlabel('time [s]')
        ax.set_ylabel('control magnitude')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(left=0.)
        ax.set_ylim(-1.,1)
        ax.set_yticks([-1,0,1])
        # print(ax.xaxis.get_ticklabels())
        # for n, label in enumerate(ax.xaxis.get_ticklabels()):
        #     if n % 4 != 0:
        #         label.set_visible(False)
        # ax.axhline(0., dashes=[1,2], color='black', alpha=0.3)
        ax.spines['bottom'].set_position(('data', 0.))
        # legend and label
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),loc='upper right', prop={'size': 6})


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plotmetatrend(df):
    data=list(df.category)
    vocab={'skip':0,'normal':1,"crazy":2,'lazy':3,'wrong_target':4}
    data=[vocab[each] for each in data]
    x = [i for i in range(len(data))]
    # plt.plot(y)
    plt.plot(x, smooth(data,30), lw=2,label='trial type')
    plt.xlabel('trial number')
    plt.ylabel('reward, reward rate and trial type')
    plt.legend()
    data=list(df.rewarded)
    data=[1 if each else 0 for each in data]
    plt.plot(x, smooth(data,30), lw=2,label='reward')
    print(vocab)


def ovbystate():
    ind=torch.randint(low=100,high=300,size=(1,))
    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.plot(states[ind][:,0],states[ind][:,1], color='r',alpha=0.5)
        goalcircle = plt.Circle([tasks[ind][0],tasks[ind][1]], 0.13, color='y', alpha=0.5)
        ax.add_patch(goalcircle)
        ax.set_xlim(0,1)
        ax.set_ylim(-0.6,0.6)


def get_ci(log, low=5, high=95, threshold=2,ind=-1):
    res=[l[2] for l in log[:ind//threshold]]
    mean=log[ind][0]._mean
    allsamples=[]
    for r in res:
        for point in r:
            allsamples.append([point[1],point[0]])
    allsamples.sort(key=lambda x: x[0])
    aroundsolution=allsamples[:ind//threshold]
    aroundsolution.sort(key=lambda x: x[0])
    alltheta=np.vstack([x[1] for x in aroundsolution])

    lower_ci=[np.percentile(alltheta[:,i],low) for i in range(alltheta.shape[1])]
    upper_ci=[np.percentile(alltheta[:,i],high) for i in range(alltheta.shape[1])]
    asymmetric_error = np.array(list(zip(lower_ci, upper_ci))).T
    res=np.array([ np.abs(mean.T-asymmetric_error[0,:] ), np.abs(asymmetric_error[1,:]-mean.T) ] )
    # res=asymmetric_error
    return res
    

def twodatabar(data1,data2,err1=None,err2=None,labels=None,shift=0.4,width=0.5,ylabel=None,xlabel=None,color=['b','r'],xname=''):
    xs=list(range(max(len(data1),len(data2))))
    label1=labels[0] if labels else None
    label2=labels[1] if labels else None
    with initiate_plot(6, 4, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        # Create bars and choose color
        ax.bar(xs, data1,width,yerr=err1,label=label1,color=color[0])
        ax.bar([i+shift for i in range(len(xs))], data2,width,yerr=err2,label=label2,color=color[1])
        # title and axis names
        ax.set_ylabel(ylabel)
        ax.set_xticks([i for i in range(max(len(data1),len(data2)))])
        if xlabel:
            ax.set_xticklabels(xlabel, rotation=45, ha='right')
        ax.set_xlabel(xname)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend()
   

def conditional_cov(covyy,covxx, covxy):
    # return the conditional cov of y|x
    # here use covxy==covyx as no causal here
    return covyy-covxy@np.linalg.inv(covxx)@covxy.T


def conditional_cov_block(cov, paramindex):
    cov=np.asfarray(cov)
    covxy=np.array(list(cov[:,paramindex][:paramindex]) + list(cov[:,paramindex][paramindex+1:]))
    covyy=np.diag(cov)[paramindex]
    temp=np.delete(cov,(paramindex),axis=0)
    covxx=np.delete(temp,(paramindex),axis=1)
    return covyy-covxy@np.linalg.inv(covxx)@covxy.T


def scatter_hist(x,y):
    def _scatter_hist(x, y, ax, ax_histx, ax_histy):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        ax.scatter(x, y,alpha=0.3)

        # now determine nice limits by hand:
        binwidth = 0.5
        xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
        lim = (int(xymax/binwidth)*1.1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        ax_histx.hist(x, bins=22)
        ax_histy.hist(y, bins=22, orientation='horizontal')

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    # use the previously defined function
    _scatter_hist(x, y, ax, ax_histx, ax_histy)
    plt.show()


def overheaddf_tar(df, alpha=1,**kwargs):
    fontsize = 9
    with initiate_plot(1.8, 1.8, 300) as fig:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)
        fig.tight_layout(pad=0)
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')
        ax.scatter(df[df.rewarded].target_x,df[df.rewarded].target_y, c='k',alpha=alpha, edgecolors='none',marker='.', s=2, lw=1)
        ax.scatter(df[~df.rewarded].target_x,df[~df.rewarded].target_y, c='r',alpha=alpha,edgecolors='none', marker='.', s=2, lw=1)


def overheaddf_path(df,indls, alpha=0.5,**kwargs):
    pathcolor='gray' if 'color' not in kwargs else kwargs['color'] 
    fontsize = 9
    with initiate_plot(1.8, 1.8, 300) as fig:
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.axes.xaxis.set_ticks([]); ax.axes.yaxis.set_ticks([])
        ax.set_xlim([-235, 235]); ax.set_ylim([-2, 430])
        x_temp = np.linspace(-235, 235)
        ax.plot(x_temp, np.sqrt(420**2 - x_temp**2), c='k', ls=':')
        ax.text(-10, 425, s=r'$70\degree$', fontsize=fontsize)
        ax.text(130, 150, s=r'$400cm$', fontsize=fontsize)
        ax.plot(np.linspace(-230, -130), np.linspace(0, 0), c='k')
        ax.plot(np.linspace(-230, -230), np.linspace(0, 100), c='k')
        ax.text(-230, 100, s=r'$100cm$', fontsize=fontsize)
        ax.text(-130, 0, s=r'$100cm$', fontsize=fontsize)
        fig.tight_layout(pad=0)
        ax.plot(np.linspace(0, 230 + 7),
                np.tan(np.deg2rad(55)) * np.linspace(0, 230 + 7) - 10, c='k', ls=':')

        for trial_i in indls:
            ax.plot(df.iloc[trial_i].pos_x,df.iloc[trial_i].pos_y,c=pathcolor, lw=0.1, ls='-', alpha=alpha)


def conditional_cov(covyy,covxx, covxy):
    # return the conditional cov of y|x
    # here use covxy==covyx as no causal here
    return covyy-covxy@np.linalg.inv(covxx)@covxy.T


def conditional_cov_block(cov, paramindex):
    cov=np.asfarray(cov)
    covxy=np.array(list(cov[:,paramindex][:paramindex]) + list(cov[:,paramindex][paramindex+1:]))
    covyy=np.diag(cov)[paramindex]
    temp=np.delete(cov,(paramindex),axis=0)
    covxx=np.delete(temp,(paramindex),axis=1)
    return covyy-covxy@np.linalg.inv(covxx)@covxy.T


def correlation_from_covariance(covariance):
    v = np.sqrt(np.diag(covariance)) # std
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def process_inv(res, removegr=True, ci=5,ind=-1):
    # get final theta and cov 
    if type(res) == str:
        res=Path(res)
    print(res)
    with open(res, 'rb') as f:
        log = pickle.load(f)
    finalcov=torch.tensor(log[ind][0]._C).float()
    finaltheta=torch.tensor(log[ind][0]._mean).view(-1,1)
    theta=torch.cat([finaltheta[:6],finaltheta[-4:]])
    cov = finalcov[torch.arange(finalcov.size(0))!=6] 
    cov = cov[:,torch.arange(cov.size(1))!=6] 
    cirange=get_ci(log, low=ci, high=100-ci,ind=ind).astype('float32')
    if removegr:
        return theta, cov, np.delete(cirange,(6),axis=1)
    return finaltheta, finalcov, cirange


def multimonkeytheta(monkeynames, mus, covs, errs, shifts=None):
    nmonkey=len(monkeynames)
    basecolor=color_settings['a']
    colorrgb=hex2rgb(basecolor)
    colorlist=colorshift(colorrgb, [0,-1,1],0.3, nmonkey)
    if shifts is None:
        shifts=np.linspace(-1/(nmonkey+1),1/(nmonkey+1),nmonkey)
    if nmonkey==2:
        shifts=[-0.15,0.15]
    for i in range(nmonkey):
        if i==0:
            ax=theta_bar(mus[i],covs[i], label=monkeynames[i],width=1/(nmonkey+1),shift=shifts[i], err=errs[i], color=colorlist[i])
        else:
            theta_bar(mus[i],covs[i],ax=ax, label=monkeynames[i],width=1/(nmonkey+1),shift=shifts[i], err=errs[i],color=colorlist[i])
    return ax


def multimonkeyobs(monkeynames, mus, covs, errs, ):
    def _bar(finaltheta,xlabels=theta_names[4:6],err=None, ax=None, label=None,shift=0,width=0.5, color=None):
        with initiate_plot(4, 3, 300) as fig, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if ax is None:
                ax = fig.add_subplot(111)
            # Create bars and choose color
            ax.bar([i+shift for i in range(len(xlabels))], finaltheta,width,yerr=err,label=label, color=color)
            # title and axis names
            ax.set_ylabel('inferred observation noise')
            ax.set_xticks([i for i in range(len(xlabels))])
            ax.set_xticklabels(xlabels, rotation=45, ha='right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend()
        return ax

    nmonkey=len(monkeynames)
    basecolor=color_settings['o']
    colorrgb=hex2rgb(basecolor)
    colorlist=colorshift(colorrgb, colorrgb,0.8, nmonkey)
    shifts=np.linspace(-1/nmonkey,1/nmonkey,nmonkey)
    for i in range(nmonkey):
        if i==0:
            ax=_bar(mus[i][4:6], label=monkeynames[i],width=1/nmonkey,shift=shifts[i], err=errs[i][:,4:6], color=colorlist[i])
        else:
            _bar(mus[i][4:6],ax=ax, label=monkeynames[i],width=1/nmonkey,shift=shifts[i], err=errs[i][:,4:6],color=colorlist[i])
    return ax


def multimonkeyeig(monkeynames, evectors):
    nmonkey=len(monkeynames)
    basecolor=color_settings['a']
    colorrgb=hex2rgb(basecolor)
    colorlist=colorshift(colorrgb, [0,-1,1],0.3, nmonkey)
    shifts=np.linspace(-1/nmonkey,1/nmonkey,nmonkey)
    for i in range(nmonkey):
        if i==0:
            ax=theta_bar(evectors[i], label=monkeynames[i],width=1/nmonkey,shift=shifts[i],  color=colorlist[i])
        else:
            theta_bar(evectors[i],ax=ax, label=monkeynames[i],width=1/nmonkey,shift=shifts[i], color=colorlist[i])
    return ax


def plotv(indls,resirc,actions=None,**kwargs):
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)

        # plot data
        for trial_i in indls:
            ax.plot(actions[trial_i][1:,0],color=hex2rgb(color_settings['a']),alpha=1/len(indls)**.3, label='monkey')

        for trial_i in range(resirc['num_trials']):
            ax.plot(resirc['agent_actions'][trial_i][:,0],color=hex2rgb(color_settings['model']),alpha=1/resirc['num_trials']**.3, label='IRC')
        # legend and label
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time, dt')
        ax.set_ylabel('control magnitude')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),loc='upper right')
        ax.set_xlim(left=0.)
        ax.set_ylim(-0.1,1)
        ax.set_yticks([0,1])
        return  ax


def plotw(indls,resirc,actions=None,**kwargs):
    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)

        # plot data
        for trial_i in indls:
            ax.plot(actions[trial_i][1:,1],color=hex2rgb(color_settings['a']),alpha=1/len(indls)**.3, label='monkey')

        for trial_i in range(resirc['num_trials']):
            ax.plot(resirc['agent_actions'][trial_i][:,1],color=hex2rgb(color_settings['model']),alpha=1/resirc['num_trials']**.3, label='IRC')
        # legend and label
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time, dt')
        ax.set_ylabel('control magnitude')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),loc='upper right')
        ax.set_xlim(left=0.)
        ax.set_ylim(-1.,1)
        ax.set_yticks([-1,0,1])
        ax.spines['bottom'].set_position('zero')
        return  ax


def multimonkeyerr(densities, mus, monkeynames):
    def _bar(data,xlabels=densities,err=None, ax=None, label=None,shift=0,width=0.5, color=None, ylabel='radial error [cm]', xlabel='densities'):
        with initiate_plot(4, 4, 300) as fig, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if ax is None:
                ax = fig.add_subplot(111)
            # Create bars and choose color
            ax.bar([i+shift for i in range(len(densities))], data,width,yerr=err,label=label, color=color)
            # title and axis names
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.set_xticks([i for i in range(len(densities))])
            ax.set_xticklabels(xlabels)
            # ax.set_xticklabels(xlabels, rotation=45, ha='right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend()
        return ax

    nmonkey=len(mus)
    basecolor=color_settings['a']
    colorrgb=hex2rgb(basecolor)
    colorlist=colorshift(colorrgb, [0,-1,1],0.3, nmonkey)
    shifts=np.linspace(-1/nmonkey,1/nmonkey,nmonkey)
    for i in range(nmonkey):
        if i==0:
            ax=_bar(mus[i], label=monkeynames[i],width=1/(nmonkey+0.2),shift=shifts[i], color=colorlist[i])
        else:
            _bar(mus[i],ax=ax, label=monkeynames[i],width=1/(nmonkey+0.2),shift=shifts[i],color=colorlist[i])
    return ax


def plotv_fill(indls,resirc,actions=None,percentile=5,**kwargs):

    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        
        # plot data
        mk=[]
        for trial_i in indls:
            mk.append(actions[trial_i][1:,0].tolist())
        mk=ll2array(mk) # trial, ts
        low=np.percentile(mk,percentile, axis=0)
        high=np.percentile(mk,100-percentile, axis=0)
        ts=np.arange(len(mk.T))*0.1
        ax.fill_between(ts, y1=low, y2=high, color=color_settings['a'],alpha=0.5, edgecolor='none', label='monkey')

        mk=[]
        for trial_i in range(resirc['num_trials']):
            mk.append(resirc['agent_actions'][trial_i][:,0].tolist())
        mk=ll2array(mk) # trial, ts
        low=np.percentile(mk,percentile, axis=0)
        high=np.percentile(mk,100-percentile, axis=0)
        ts=np.arange(len(mk.T))*0.1
        ax.fill_between(ts, y1=low, y2=high, color=color_settings['model'],alpha=0.5, edgecolor='none', label='IRC')

        # legend and label
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time, dt')
        ax.set_ylabel('control magnitude')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),loc='upper right')
        ax.set_xlim(left=0.)
        ax.set_ylim(-0.1,1)
        ax.set_yticks([0,1])
        ax.set_xticks([0,1,2])
        return  ax


def plotw_fill(indls,resirc,actions=None,percentile=5,**kwargs):

    with initiate_plot(3.8, 1.8, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        
        # plot data
        mk=[]
        for trial_i in indls:
            mk.append(actions[trial_i][1:,1].tolist())
        mk=ll2array(mk) # trial, ts
        low=np.percentile(mk,percentile, axis=0)
        high=np.percentile(mk,100-percentile, axis=0)
        ts=np.arange(len(mk.T))*0.1
        ax.fill_between(ts, y1=low, y2=high, color=color_settings['a'],alpha=0.5, edgecolor='none', label='monkey')

        mk=[]
        for trial_i in range(resirc['num_trials']):
            mk.append(resirc['agent_actions'][trial_i][:,1].tolist())
        mk=ll2array(mk) # trial, ts
        low=np.percentile(mk,percentile, axis=0)
        high=np.percentile(mk,100-percentile, axis=0)
        ts=np.arange(len(mk.T))*0.1
        ax.fill_between(ts, y1=low, y2=high, color=color_settings['model'],alpha=0.5, edgecolor='none', label='IRC')

        # legend and label
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('time, dt')
        ax.set_ylabel('control magnitude')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),loc='upper right')
        ax.set_xlim(left=0.)
        ax.set_ylim(-1.,1)
        ax.set_yticks([-1,0,1])
        ax.spines['bottom'].set_position('zero')
        ax.set_xticks([0,1,2])

        return  ax



def getcbarnorm(min,mid,max):
    divnorm=matplotlib.colors.TwoSlopeNorm(vmin=min, vcenter=mid, vmax=max)
    return divnorm





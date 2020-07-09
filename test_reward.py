# will generate a circle of goal ( actually a gassian)
# and a cov ellipse


import numpy as np
from numpy import pi
import time
import torch
import gym
from stable_baselines.ddpg.policies import LnMlpPolicy, MlpPolicy

import matplotlib.pyplot as plt
import scipy.stats as stats
import math

from stable_baselines import DDPG
from FireflyEnv import ffenv,ffenv_new_cord,ffenv_original
from reward_functions import reward_singleff
from Config import Config
arg=Config()

env=ffenv_new_cord.FireflyAgentCenter(arg)
model=DDPG.load('DDPG_belief_reward_500000_4 5 21 45.zip')


model.set_env(env)


# action=[1,1]
# action=[1,-1]
# action=[1,0]
# action=[0,0]
# def raction():
#     return [np.random.randint(2),np.random.randint(2)]
# env.step(action)
# env.s
# env.b
# env.episode_time

# env.reset()

env.reset(
    pro_gains=torch.Tensor([1,2]),
    pro_noise_stds=torch.Tensor([0.4,0.4]),
    obs_noise_stds=torch.Tensor([0.4,0.4]))
while env.episode_time<30:
    # action=raction()
    action=[1,1]
    action=[1,-1]
    action=[1,0]
    env.step(action)
    fig=plot_belief(env)

env.reset()
while env.episode_time<30 and not env.stop:
    action,_=model.predict(env.decision_info)
    decision_info,_,_,_=env.step(action)
    fig=plot_belief(env,title=(action,env.phi),kwargs={'title':action})
    fig.savefig("{}.png".format(env.episode_time))

#     fig.savefig("{}.png".format(env.episode_time))
# env.phi
env.std_range
env.reset_task_param()
# torch.distributions.Normal(0,env.phi[2:4]).sample()        
# env.phi[2:4]
# env.P[:3,:3]
# pos=env.b[0,:2]
# cov=env.P[:2,:2]

def plot_belief(env,title='title',**kwargs):
    f1=plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.set_ylim(-1.5,1.5)
    ax.set_xlim(-1.5,1.5)
    pos=env.b[0,:2]
    cov=env.P[:2,:2]
    # plt.plot(pos[0],pos[1],'o')
    # print('test',title)
    if kwargs.get('title'):
        print('test',kwargs.get('title'))
        title=kwargs.get('title')
    plt.title(title)
    plt.plot(env.s[0,0],env.s[0,1],'ro')
    plt.plot([pos[0],pos[0]+env.decision_info[0,0]*np.cos(env.b[0,2]+env.decision_info[0,1]) ],
    [pos[1],pos[1]+env.decision_info[0,0]*np.sin(env.b[0,2]+env.decision_info[0,1])],'g')
    plt.quiver(pos[0], pos[1],np.cos(env.b[0,2].item()),np.sin(env.b[0,2].item()), color='r', scale=10)
    plot_cov_ellipse(cov, pos, nstd=2,ax=ax)
    # plot_cov_ellipse(np.diag([1,1])*0.05, [env.goalx,env.goaly], nstd=1, ax=ax)
    plot_circle(np.eye(2)*env.phi[-1].item(),[env.goalx,env.goaly],ax=ax,color='y')
    return f1

# np.sqrt(env.P[:3,:3])

# mu=0
# sigma=0.1
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, stats.norm.pdf(x, mu, sigma))


# plot_cov_ellipse(cov, [0.5,0.5], nstd=2)
# plot_cov_ellipse(np.diag([1,1])*0.05, [0.5,0.5], nstd=2)



from scipy.stats import norm, chi2

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

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

def plot_cov_ellipse(cov, pos, nstd=2, color=None, ax=None, **kwargs):
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
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    if color is not None:
        ellip.set_color(color)
    ellip.set_alpha(0.5)
    ax.add_artist(ellip)
    return ellip
    
def plot_circle(cov, pos, color=None, ax=None, **kwargs):

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
    c.set_alpha(0.5)
    ax.add_artist(c)
    return c

# points = np.random.multivariate_normal(
#         mean=(1,1), cov=[[0.4, 9],[9, 10]], size=1000
#         )
# # Plot the raw points...
# x, y = points.T
# plt.plot(x, y, 'ro')

# # Plot a transparent 3 standard deviation covariance ellipse
# plot_point_cov(points, nstd=3, alpha=0.5, color='green')

# plt.show()


## - ----------------------------
# show ellipse and circle 
# given r, cov, and offset mu, calculate overlap
def overlap_prob(r, cov, mu ):
    # normalizing_z=2*pi*r**2
    rop=cov.copy()
    rop[0,0]=rop[0,0]+r**2
    rop[1,1]=rop[0,0]+r**2

    # normalizing_z=1/2/pi/np.sqrt(np.linalg.det(rop))


    if type(mu) == list:
        vector_mu=np.asarray(mu).reshape((2,1))
    # expectation=np.exp((-1/2/pi/r**2) * vector_mu.transpose()@np.linalg.inv(rop)@vector_mu)
    expectation=np.exp(-1/2* vector_mu.transpose()@np.linalg.inv(rop)@vector_mu)

    return expectation

def overlap_intergration(r, cov, mu, bins=20):
    xrange=[mu[0]-r,mu[0]+r]
    yrange=[mu[1]-r,mu[1]+r]
    P=0

    for i in np.linspace(xrange[0],xrange[1],bins):
        for j in np.linspace(yrange[0],yrange[1],bins):
            if i**2+j**2<=r**2:
                expectation=( (1/2/pi/np.sqrt(np.linalg.det(cov)))
                    * np.exp(-1/2
                    * np.array([i,j]).transpose()@np.linalg.inv(cov)@np.array([i,j]).reshape(2,1) ))
                # print(expectation)
                P=P+expectation/(bins/2/r)**2
    return P

def plot_overlap(r,cov,mu,title=None):
    f1=plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.set_ylim(-1.5,1.5)
    ax.set_xlim(-1.5,1.5)
    if title is not None:
        ax.title.set_text(str(title))
    plot_cov_ellipse(cov,[0,0],nstd=1,ax=ax)
    plot_circle(np.eye(2)*r,mu,ax=ax,color='r')
    return f1

# mu=[0,0]
# plot_cov_ellipse(cov,[0.,0.])
# plot_cov_ellipse(np.eye(2)*0.1,mu,nstd=1)

# cov=np.array([[0.03,0.03],[0.03,0.05]])
# cov=cov*0.6
# cov=cov*10
# r=0.2
# for i in np.linspace(-0.5,0.5,19):
#     for j in np.linspace(-0.5,0.5,19):
#         plot_overlap(r,cov,[i,j],title=str(str(overlap_intergration(r,cov,[i,j]))+str(overlap_prob(r,cov,[i,j]))))
#         plt.savefig("{}_{}.png".format(str(i),str(j)))
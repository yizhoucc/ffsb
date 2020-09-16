import numpy as np
from numpy import pi
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1
from matplotlib.patches import Ellipse, Circle


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


def plot_cov_ellipse(cov, pos=[0,0], nstd=2, color=None, ax=None,alpha_factor=1, **kwargs):
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
    ellip.set_alpha(0.5*alpha_factor)
    ax.add_artist(ellip)
    return ellip


def plot_circle(cov, pos, color=None, ax=None,alpha_factor=1, **kwargs):
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
    c.set_alpha(0.5*alpha_factor)
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



import numpy as np
from numpy import pi
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1
from matplotlib.patches import Ellipse, Circle
from InverseFuncs import *

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
# from InverseFuncs import *

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


# def change_of_basis(X, W):
#   """
#   Projects data onto a new basis.

#   Args:
#     X (numpy array of floats) : Data matrix each column corresponding to a
#                                 different random variable
#     W (numpy array of floats) : new orthonormal basis columns correspond to
#                                 basis vectors

#   Returns:
#     (numpy array of floats)   : Data matrix expressed in new basis
#   """

#   Y = np.matmul(X, W)
#   return Y


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
        env.reset(phi=phi,theta=theta,goal_position=etask[0])
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



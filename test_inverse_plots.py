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


def compute_H(env, agent, theta_estimation, true_theta, phi, trajectory_data=None,H_dim=7, num_episodes=100,is1d=False):
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


def compute_loss(env, agent, theta_estimation, true_theta, phi, trajectory_data=None, num_episodes=100,is1d=False):
  if trajectory_data is None:
    states, actions, tasks=trajectory(agent, torch.Tensor(phi), torch.Tensor(true_theta), env, num_episodes, is1d=is1d)
  else:
    actions=trajectory_data['actions']
    tasks=trajectory_data['tasks']
  theta_estimation=torch.nn.Parameter(torch.Tensor(theta_estimation))
  loss = getLoss(agent, actions, tasks, torch.Tensor(phi), theta_estimation, env,states=states)
  return loss


def compute_loss_wrapped(env, agent, true_theta, phi, trajectory_data=None, num_episodes=100,is1d=False):
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


def diagnose_dict(index,inverse_data, monkeystate=True,num_trials=10):
  inverse_data=load_inverse_data(inverse_data)
  packed={}
  packed['index']=index
  packed['phi']=torch.tensor(inverse_data['phi'])
  packed['theta']=torch.tensor(inverse_data['theta_estimations'][-1])
  if monkeystate:
    packed['estate']=states[index]
  else: 
    packed['estate']=None
    packed['monkeystate']=states[index]
  packed['eaction']=actions[index]
  packed['etask']=tasks[index]
  packed['agent']=agent
  packed['env']=env
  packed['env']=env
  packed['initv']=actions[index][0][0]
  packed['initw']=actions[index][0][1]
  return packed

def diagnose_trial(index, phi, eaction, etask, theta, agent, env, num_trials=10,monkeystate=None,estate=None, guided=True, initv=0, initw=0):
  agent_actions=[]
  agent_beliefs=[]
  agent_covs=[]
  agent_states=[]
  with torch.no_grad():
    for trial_i in range(num_trials):
      env.reset(phi=phi,theta=theta,goal_position=etask[0],initv=initv, initw=initw)
      epbliefs=[]
      epbcov=[]
      epactions=[]
      epstates=[]
      t=0
      while t<len(eaction):
        action = agent(env.decision_info)[0]
        if estate is None and guided:
          # if t==1: print('no state')
          _,done=env(torch.tensor(eaction[t]).reshape(1,-1),task_param=theta) 
        elif estate is None and not guided:
          # if t==1: print('not guided')
          _,done=env(torch.tensor(action).reshape(1,-1),task_param=theta) 
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

def diagnose_plot(estate, theta,eaction, etask, agent_actions, agent_beliefs, agent_covs,astate=None,index=None):
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
  ax3.plot(estate[0,:],estate[1,:])
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
  # diaginfo['agent_beliefs'][0][:,:,0][:,0]
  # diaginfo['agent_beliefs'][0][:,:,0][:,1]
  # len(diaginfo['agent_beliefs'][0][:,:,0])
  for t in range(len(agent_beliefs[0][:,:,0])):
    cov=diaginfo['agent_covs'][0][t][:2,:2]
    pos=  [agent_beliefs[0][:,:,0][t,0],
            agent_beliefs[0][:,:,0][t,1]]
    plot_cov_ellipse(cov, pos, nstd=2, color=None, ax=ax4,alpha=0.2)

def diagnose_plot_theta(agent, env, phi, theta_init, theta_final,nplots,):
  def sample_trials(agent, env, theta, phi, etask, num_trials=5, initv=0.):
    # print(theta)
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
        done=False
        while not done:
          action = agent(env.decision_info)[0]
          _,done=env(torch.tensor(action).reshape(1,-1),task_param=theta) 
          epactions.append(action)
          epbliefs.append(env.b)
          epbcov.append(env.P)
          epstates.append(env.s)
          t=t+1
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
  fig = plt.figure(figsize=[16, 20])
  # curved trails
  etask=[[0.7,-0.3]]
  for n in range(nplots):
    ax1 = fig.add_subplot(4,nplots,n+1)
    theta=(n-1)*delta+theta_init
    ax1.set_xlabel('world x, cm')
    ax1.set_ylabel('world y, cm')
    ax1.set_title('state plot')
    data=sample_trials(agent, env, theta, phi, etask, num_trials=1, initv=0.)
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
    ax2 = fig.add_subplot(4,nplots,n+nplots+1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('w')
    ax2.set_title('w control')
    agent_actions=data['agent_actions']
    for i in range(len(agent_actions)):
      ax2.plot(agent_actions[i],alpha=0.7)
    ax2.set_ylim([-1.1,1.1])

  # straight trails
  etask=[[0.7,0.0]]
  for n in range(nplots):
    ax1 = fig.add_subplot(4,nplots,n+nplots*2+1)
    theta=(n-1)*delta+theta_init
    ax1.set_xlabel('world x, cm')
    ax1.set_ylabel('world y, cm')
    ax1.set_title('state plot')
    data=sample_trials(agent, env, theta, phi, etask, num_trials=1, initv=0.)
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
    ax2 = fig.add_subplot(4,nplots,n+nplots*3+1)
    ax2.set_xlabel('t')
    ax2.set_ylabel('w')
    ax2.set_title('w control')
    data.keys()
    agent_actions=data['agent_actions']
    for i in range(len(agent_actions)):
      ax2.plot(agent_actions[i],alpha=0.7)
    ax2.set_ylim([-1.1,1.1])

def diagnose_plot_xaqgrant(estate, theta,eaction, etask, agent_actions, agent_beliefs, agent_covs,astate=None,index=None, tasks=None, actions=None):
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

def diagnose_stopcompare(estate, theta,eaction, etask, agent_actions, agent_beliefs, agent_covs,astate=None,index=None, tasks=None, actions=None):
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

def diagnose_plot_xaqgrantoverhead(estate, theta,eaction, etask, agent_actions, agent_beliefs, agent_covs,astate=None):
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

def diagnose_plot_stop(estate, theta,eaction, etask, agent_actions, agent_beliefs, agent_covs,astate=None, index=None):
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
      if tasks[i][0][0]>tasks[ind][0][0]-0.05 \
      and tasks[i][0][0]<tasks[ind][0][0]+0.05 \
      and tasks[i][0][1]>tasks[ind][0][1]-0.03 \
      and tasks[i][0][1]<tasks[ind][0][1]+0.03 \
      and actions[i][0][0]>actions[ind][0][0]-0.1 \
      and actions[i][0][0]<actions[ind][0][0]+0.1:
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


# plot losses
a=load_inverse_data('30_8_38')
theta_trajectory=a['theta_estimations']
true_theta=a['true_theta']
theta_estimation=theta_trajectory[-1]
phi=np.array(a['phi'])
H=a['Hessian'] 
stds=a['theta_std']
losses=np.array(a['loss'])
# background_data=plot_inverse_trajectory(theta_trajectory,true_theta,env,agent, phi=phi)
plt.plot(losses)


img=plt.imshow(H[-1])
add_colorbar(img)
plt.imshow(torch.sign(H[-1]))

# plot monkey similar trials, path and controls
indls=similar_trials(ind,tasks,actions)
fig = plt.figure(figsize=[16, 8])
ax1 = fig.add_subplot(121)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('path')
ax1.set_xlim([-0.1,1.1])
ax1.set_ylim([-0.6,0.6])
ax2 = fig.add_subplot(122)
ax2.set_xlabel('t')
ax2.set_ylabel('controls')
ax2.set_title('controls')
ax2.set_ylim([-1.1,1.1])
for ind in indls:
  ax1.plot(states[ind][0,:],states[ind][1,:], alpha=0.2,color='red')
  goalcircle = plt.Circle((tasks[ind][0][0],tasks[ind][0][1]), 0.13, color='y', alpha=0.3)
  ax1.add_patch(goalcircle)

  monkeyobs=False
  pac=diagnose_dict(ind,'3_21_42', monkeystate=monkeyobs)
  pac['theta']=torch.tensor([[0.2245],
        [1.0581],
        [0.3250],
        [0.1881],
        [0.1867],
        [0.1844],
        [0.1592],
        [0.0547],
        [0.6404],
        [0.7203]])
  pac['initv']=actions[ind][0][0]
  pac['guided']=monkeyobs
  pac['num_trials']=1
  diaginfo=diagnose_trial(**pac)
  ax1.plot(diaginfo['astate'][0][:,0],diaginfo['astate'][0][:,1], alpha=0.2,color='green')

  ax2.plot(torch.tensor(actions[ind])[:,0], color='orange', alpha=0.3)
  ax2.plot(torch.tensor(actions[ind])[:,1], color='teal', alpha=0.3)

  ax2.plot(diaginfo['agent_actions'][0][:,0], color='blue', alpha=0.3)
  ax2.plot(diaginfo['agent_actions'][0][:,1], color='brown', alpha=0.3)



# diagnoise inverse results
ind=torch.randint(low=100,high=7000,size=(1,))
monkeyobs=False
pac=diagnose_dict(ind,'30_8_38', monkeystate=monkeyobs)
pac['guided']=monkeyobs
pac['num_trials']=15
pac['theta']=torch.tensor([[0.4],
        [1.57],
        [0.3],
        [0.3],
        [0.3],
        [0.3],
        [0.13],
        [0.1],
        [0.2],
        [0.2],
        [0.1],
        [0.1],
])
# pac['phi']=pac['theta']
diaginfo=diagnose_trial(**pac)
print(ind)
# diagnose_plot(**diaginfo)
# diagnose_plot_stop(**diaginfo)
# diagnose_plot_xaqgrant(tasks=tasks,actions=actions,**diaginfo)
diagnose_stopcompare(tasks=tasks,actions=actions,**diaginfo)



ind=torch.randint(low=100,high=7000,size=(1,))
plot_monkey_trial(df,int(ind))

# slice through policy by theta
phi=pac['phi']
theta_init=torch.tensor([[0.4000],
        [1.5708],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.13],
        [0.0050],
        [0.3],
        [0.3]])
theta_final=torch.tensor([[0.4000],
        [1.5708],
        [0.6],
        [0.6],
        [0.6],
        [0.6],
        [0.13],
        [0.0050],
        [0.9],
        [0.9]])
diagnose_plot_theta(agent, env, phi, theta_init, theta_final,5)


# target on and off difference
off=df[df.isFullOn==False]
on=df[df.isFullOn==True]
onstraight=on[on.FFX<10]
onstraight=onstraight[onstraight.FFX>-10]
offstraight=off[off.FFX<10]
offstraight=offstraight[offstraight.FFX>-10]

lastron=[]
for x,y in zip(onstraight.real_relative_radius,onstraight.FFY):
  if np.min(x)<x[-1]-10:
    lastron.append(-x[-1])
  else:
    lastron.append(x[-1])
  plt.scatter(lastron[-1], y)
np.mean(lastron)
  
lastroff=[]
for x in (offstraight.real_relative_radius):
  if np.min(x)<x[-1]-10:
    lastroff.append(-x[-1])
  else:
    lastroff.append(x[-1])
np.mean(lastroff)

plt.hist(lastron,bins=20)
plt.title('straight ahead with visible target')
plt.xlabel('cm stop from target')
plt.ylabel('number of trials')
plt.hist(lastroff,bins=20)
plt.title('straight ahead with nonvisible target')
plt.xlabel('cm stop from target')
plt.ylabel('number of trials')


# what is monkey's control when at edge for full on trials?
for x,y in zip(onstraight.action_v,onstraight.real_relative_radius):
  plt.plot([0,1],[65,65], 'r-', lw=2)
  plt.plot(x,y, 'teal', alpha=0.3)
plt.title('v control vs distance')
plt.xlabel('v control')
plt.ylabel('cm distance from target')

# what is mokney's position when decelerate?
vs=[]
for x,y in zip(onstraight.action_v,onstraight.real_relative_radius):
  decindex=np.argmin(np.gradient(np.gradient(x)))
  vs.append(y[decindex])
vs=[v for v in vs if v<100]
plt.hist(vs,bins=30)
plt.title('monkey distance to goal when start to decelerate')
plt.xlabel('cm distance to goal')
plt.ylabel('number of trials')

# # testing relationship of two noise parameters
# background_data_fake=np.zeros((number_pixels,number_pixels))
# X,Y=np.meshgrid(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels),np.linspace(xyrange[1][0],xyrange[1][1],number_pixels))
# for i in range(background_data_fake.shape[0]):
#   for j in range(background_data_fake.shape[1]):
#     # background_data_fake[i,j]=X[i,j]*Y[i,j]/(X[i,j]+Y[i,j])
#     background_data_fake[i,j]=(X[i,j])*(0.5-Y[i,j])
# plt.imshow(background_data_fake)
# plt.imshow(background_data)


[[min(true_theta[param_pair[0]],
theta_estimation[param_pair[0]])[0],max(true_theta[param_pair[0]],theta_estimation[param_pair[0]])[0]],[min(true_theta[param_pair[1]],
theta_estimation[param_pair[1]])[0],max(true_theta[param_pair[1]],theta_estimation[param_pair[1]])[0]]]
# testing relationship of two noise parameters
param_pair=[2,3]
param_range=[[0.4,1.4],[0.001,0.5]]
loss_function=compute_loss_wrapped(env, agent, np.array(true_theta).reshape(-1,1), 
  np.array(phi).reshape(-1,1), trajectory_data=None, num_episodes=1000,is1d=True)
resolution=5
bgdata=loss_surface_two_param(true_theta,loss_function, param_pair,
    number_pixels=resolution, param_range=param_range)
fig = plt.figure(figsize=[9, 9])
ax = fig.add_subplot()
im=ax.imshow(bgdata)
add_colorbar(im)
ax.set_title('logll loss surface in param space {} and {}'.format(param_pair[0], param_pair[1]), fontsize= 20)
ax.set_xlabel('parameter {}'.format(param_pair[0]), fontsize=15)
ax.set_xticklabels(["{:.2f}".format(x) for x in np.linspace(param_range[0][0],param_range[0][1],resolution+1).tolist()])
ax.set_yticklabels(["{:.2f}".format(x) for x in np.linspace(param_range[1][0],param_range[1][1],resolution+1).tolist()])
ax.set_ylabel('parameter {}'.format(param_pair[1]), fontsize=15)
# ax.scatter(true_theta[param_pair[0]],true_theta[param_pair[1]],color='r')
xs=np.linspace(param_range[0][0],param_range[0][1],resolution+1).tolist()
ys=np.linspace(param_range[1][0],param_range[1][1],resolution+1).tolist()
try:
  ax.scatter([i for i in list(range(resolution)) if xs[i]<true_theta[param_pair[0]][0] and xs[i+1]>true_theta[param_pair[0]][0]],
    [i for i in list(range(resolution)) if ys[i]<true_theta[param_pair[1]][0] and ys[i+1]>true_theta[param_pair[1]][0]],color='r')
  ax.scatter([i for i in list(range(resolution)) if xs[i]<theta_estimation[param_pair[0]][0] and xs[i+1]>theta_estimation[param_pair[0]][0]],
    [i for i in list(range(resolution)) if ys[i]<theta_estimation[param_pair[1]][0] and ys[i+1]>theta_estimation[param_pair[1]][0]],color='b')
except:
  print(theta_estimation[param_pair[0]],theta_estimation[param_pair[1]])
  print(true_theta[param_pair[0]],true_theta[param_pair[1]])




# check sparse------------------------------------------
# with l2
baselines_mlp_model =TD3.load('trained_agent/1d_easy1000000_9_30_5_20.zip')
agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[512,512],n_inputs=15,n_actions=1)
w1=agent.fc1.weight
plt.hist(torch.mean(w1.clone().detach(),0))
# no l2
baselines_mlp_model =TD3.load('trained_agent/1d_easy1000000_0_29_22_58.zip')
agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[512,512],n_inputs=15,n_actions=1)
plt.hist(torch.mean(w1.clone().detach(),0))

# 1d-----------------------------------------------------
baselines_mlp_model =TD3_torch.TD3.load('trained_agent/1d_1000000_9_16_22_20.zip')
agent=baselines_mlp_model.actor
# baselines_mlp_model =TD3.load('trained_agent/1d_7dim1000000_9_30_17_52.zip')
# agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[512,512],n_inputs=15,n_actions=1)
arg.goal_radius_range=[0.1,0.3]
arg.TERMINAL_VEL = 0.025
arg.goal_radius_range=[0.15,0.3]
arg.std_range = [0.02,0.1,0.02,0.1]
arg.TERMINAL_VEL = 0.025  # terminal velocity? # norm(action) that you believe as a signal to stop 0.1.
arg.DELTA_T=0.2
arg.EPISODE_LEN=35
env=ffac_1d.FireflyTrue1d(arg)
env.agent_knows_phi=False


# check dev cost
def action_cost_dev(action, previous_action):
  cost=(torch.norm(torch.tensor(1.0)*action-previous_action)*10)**2
  return cost
def new_dev(action, previous_action, param):
  vcost=(action[0]-previous_action[0])*10*param[0]
  wcost=(action[1]-previous_action[1])*10*param[1]
  print(action[0],previous_action[0])
  cost=vcost**2+wcost**2
  return cost
prev_action=torch.tensor([0.,0.])
costs=[]
for each_action in pac['eaction']:
  each_action=torch.tensor(each_action)
  cost=new_dev(each_action, prev_action,[0.05,0.5])
  costs.append(cost)
  # print(each_action)
  prev_action=each_action
plt.plot(costs)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(torch.tensor(pac['eaction'])[:,1],torch.tensor(pac['eaction'])[:,0])
ax1.set_ylim([-0.1,1.1])
ax1.set_xlim([-1.1,1.1])






for atheta in theta_trajectory:
    plt.plot(true_theta)
    plt.plot(atheta)
    plt.show()

loss_function=compute_loss_wrapped(env, agent, torch.tensor(true_theta).reshape(-1,1).cuda(), 
  torch.tensor(phi).reshape(-1,1).cuda(), trajectory_data=None, num_episodes=10,is1d=True)
print(loss_function(true_theta))
print(loss_function(theta_estimation))

background_data=plot_inverse_trajectory(theta_trajectory,true_theta,env,agent, 
phi=phi,background_contour=True,H=True,number_pixels=9,loss_function=loss_function)

env=ffac_1d.FireflyTrue1d_cpu(arg)
agent=agent.mu.cpu()

H=compute_H(env, agent, theta_estimation, true_theta, phi, H_dim=10, trajectory_data=None, num_episodes=2,is1d=False)
H=compute_H(env, agent, theta_estimation, true_theta, phi, H_dim=7, trajectory_data=None, num_episodes=20,is1d=True)
cov=theta_cov(H)
stderr(cov)
ev, evector=torch.eig(H,eigenvectors=True)
p=torch.round(torch.log(torch.sign(H)*H))*torch.sign(H)
img=plt.imshow(p)
img=plt.imshow(H)
add_colorbar(img)

[evv[0].item()*torch.sign(evv[0]) for evv in ev]
plt.plot()


theta_estimation=torch.nn.Parameter(torch.tensor(theta_estimation))
env.reset(theta=theta_estimation, phi=torch.tensor(phi))


# new hessian computation
def jacobian(y, x, create_graph=False):                                                               
    jac = []                                                                                          
    flat_y = y.reshape(-1)                                                                            
    grad_y = torch.zeros_like(flat_y)                                                                 
    for i in range(len(flat_y)):                                                                      
        grad_y[i] = 1.                                                                                
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))                                                           
        grad_y[i] = 0.                                                                                
    return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
def hessian(y, x):                                                                                    
    return jacobian(jacobian(y, x, create_graph=True), x)  
    
states, actions, tasks=trajectory(agent, torch.Tensor(phi), torch.Tensor(true_theta), env, 20,is1d=True)
phi=torch.nn.Parameter(torch.Tensor(phi))
phi.requires_grad=False
loss = getLoss(agent, actions, tasks, phi, theta_estimation, env,states=states, gpu=False)
hessian(loss, theta_estimation)



# # # new cord--------------------------------------------------
# baselines_mlp_model = TD3.load('trained_agent/TD_95gamma_mc_smallgoal_500000_9_24_1_6.zip')
# agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[128,128],n_inputs=29)
# # loading enviorment, same as training
# env=ffenv_new_cord.FireflyAgentCenter(arg)
# # ---seting the env for inverse----
# # TODO, move it to a function of env
# env.agent_knows_phi=False

# a=load_inverse_data('test_acc_EP200updates100lr0.1step215_8_51EP200updates100sample2IT2')
# theta_trajectory=a['theta_estimations']
# true_theta=a['true_theta']
# theta_estimation=theta_trajectory[-1]
# phi=np.array(a['phi'])
# background_data=plot_inverse_trajectory(theta_trajectory,true_theta,env,agent, phi=phi)

# background_data=plot_inverse_trajectory(theta_trajectory,true_theta,env,agent, phi=phi,background_contour=True)
# H=compute_H(env, agent, theta_estimation, true_theta, phi, trajectory_data=None, num_episodes=100)
# cov=theta_cov(H)
# stderr(cov)



# # action cost-----------------------------------------------
# baselines_mlp_model = TD3.load('trained_agent//TD_action_cost_700000_8_19_21_56.zip')
# agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[128,128],n_inputs=31)
# # loading enviorment, same as training
# env=firefly_action_cost.FireflyActionCost(arg)
# # ---seting the env for inverse----
# # TODO, move it to a function of env
# env.agent_knows_phi=False

# a=load_inverse_data('EP200updates200lr0.1step23_7_29EP200updates200sample2IT2')
# theta_trajectory=a['theta_estimations']
# true_theta=a['true_theta']
# theta_estimation=theta_trajectory[-1]
# phi=np.array(a['phi'])
# background_data=plot_inverse_trajectory(theta_trajectory,true_theta,env,agent, phi=phi)

# background_data=plot_inverse_trajectory(theta_trajectory,true_theta,env,agent, phi=phi,background_contour=True)





# # acc-----------------------------------------------
# baselines_mlp_model = TD3.load('trained_agent//acc_retrain_1000000_2_18_21_4.zip')
# agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[128,128],n_inputs=30)
# env=firefly_action_cost.FireflyActionCost(arg)
# env.agent_knows_phi=False

# a=load_inverse_data('test_acc_EP200updates100lr0.1step219_23_4EP200updates100sample2IT2')
# theta_trajectory=a['theta_estimations']
# true_theta=a['true_theta']
# theta_estimation=theta_trajectory[-1]
# phi=np.array(a['phi'])

# # no bg, faster
# background_data=plot_inverse_trajectory(theta_trajectory,true_theta,env,agent, phi=phi)
# # with bg as contour
# background_data=plot_inverse_trajectory(theta_trajectory,true_theta,env,agent, phi=phi,background_contour=True)
# # with bg as img
# background_data=plot_inverse_trajectory(theta_trajectory,
# true_theta,env,agent, phi=phi,background_data=None,
# background_contour=True,
# background_look='pixel')
# # with bg as img, with bg data already in hand
# background_data=plot_inverse_trajectory(theta_trajectory,
# true_theta,env,agent, phi=phi,background_data=background_data,
# background_contour=True,
# background_look='pixel')

# accac-----------------------------------------------------
# baselines_mlp_model = TD3.load('trained_agent//accac_final_1000000_9_11_20_25.zip')
# agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[512,512],n_inputs=32)
# arg.goal_radius_range=[0.1,0.3]
# arg.TERMINAL_VEL = 0.025
# arg.goal_radius_range=[0.15,0.3]
# arg.std_range = [0.02,0.3,0.02,0.3]
# arg.TERMINAL_VEL = 0.025  # terminal velocity? # norm(action) that you believe as a signal to stop 0.1.
# arg.DELTA_T=0.2
# arg.EPISODE_LEN=35
# env=firefly_accac.FireflyAccAc(arg)
# env.agent_knows_phi=False

# a=load_inverse_data('18_22_23')
# theta_trajectory=a['theta_estimations']
# true_theta=a['true_theta']
# theta_estimation=theta_trajectory[-1]
# phi=np.array(a['phi'])
# # no bg, faster
# background_data=plot_inverse_trajectory(theta_trajectory,true_theta,env,agent, phi=phi)




# background_data=plot_inverse_trajectory(theta_trajectory,true_theta,env,agent, 
# phi=phi,background_contour=True,H=False,number_pixels=15)


# background_data=plot_inverse_trajectory(theta_trajectory,true_theta,env,agent, 
# phi=phi,background_contour=True,H=False,number_pixels=5,background_look='pixel',background_data=background_data)

# plt.imshow(background_data)

# H=compute_H(env, agent, theta_estimation, np.array(true_theta).reshape(-1,1), phi, trajectory_data=None,H_dim=len(true_theta), num_episodes=100)
# H_trace=np.trace(H)
# cov=theta_cov(H)
# torch.eig(torch.Tensor(cov))



# plot trajectories-------------------------------------------
# action costs
theta_trajectorys=[]
a=load_inverse_data('EP100updates500lr0.1step226_6_13EP100updates500sample2IT2')
theta_trajectorys.append(a['theta_estimations'])
a=load_inverse_data('EP100updates500lr0.1step226_7_55EP100updates500sample2IT2')
theta_trajectorys.append(a['theta_estimations'])
a=load_inverse_data('EP100updates500lr0.1step225_21_18EP100updates500sample2IT2')
theta_trajectorys.append(a['theta_estimations'])
a=load_inverse_data('EP100updates500lr0.1step225_23_4EP100updates500sample2IT2')
theta_trajectorys.append(a['theta_estimations'])
a=load_inverse_data('EP100updates500lr0.1step226_0_53EP100updates500sample2IT2')
theta_trajectorys.append(a['theta_estimations'])
a=load_inverse_data('EP100updates500lr0.1step226_2_32EP100updates500sample2IT2')
theta_trajectorys.append(a['theta_estimations'])
a=load_inverse_data('EP100updates500lr0.1step226_4_16EP100updates500sample2IT2')
theta_trajectorys.append(a['theta_estimations'])


def true_vs_estimation(theta_trajectorys,true_theta,index):
  end_list=[]
  for a in theta_trajectorys:
      end_list.append(a[-1][index])
  true_value=(true_theta[index])
  return end_list, true_value


# if trace of hessian getting smaller
H_traces=[]
cov_traces=[]
Hs=[]
for index, each_theta in enumerate(theta_trajectory):
  H=compute_H(env, agent, each_theta, np.array(true_theta).reshape(-1,1), phi, trajectory_data=None,H_dim=len(true_theta), num_episodes=100)
  H_trace=np.trace(H)
  H_traces.append(H_trace)
  cov=theta_cov(H)
  cov_traces.append(np.trace(cov))
  Hs.append(H)
  print(index)
  # cov_pc=evectors[:,:len(true_theta)].transpose()@np.array(cov)@evectors[:,:len(true_theta)]
  # plot_cov_ellipse(cov_pc,pos=score[-1,:2],alpha_factor=0.5,ax=ax)
  # stderr=np.sqrt(np.diag(cov)).tolist()




# plot true vs estimation
for index in range(9):
  plt.figure
  estimations, true_value=true_vs_estimation(theta_trajectorys,true_theta,index)
  plt.scatter(estimations, true_value*len(estimations))
  plt.scatter(true_value, true_value, color='r')
  plt.title('index of param: {}, std: {}'.format(index, tstd(estimations)))
  # print(tstd(estimations))
  plt.show()
# sem(estimations)


# test hessian at true theta
import os
alldata=[]
filenames=os.listdir('inverse_data/')
prefix='fromtrue_1true'
filenames=[f  for f in filenames if f[:len(prefix)]==prefix]
for f in filenames:
  alldata.append(load_inverse_data(f))

fig = plt.figure(figsize=[16, 9])
ax = fig.add_subplot()

ax.plot(np.array([d['std_error'][0] for d in alldata]).transpose())
ax.set_title('std error of each parameters at true theta', fontsize= 20)
ax.set_xlabel('parameters in theta')
ax.set_ylabel('std error (if avaliable)')
ax.text(0,0,'note angular process gain, and obs gains have larger std error', fontsize= 20)

# test trace of hessian at true theta, not useful
fig = plt.figure(figsize=[16, 9])
ax = fig.add_subplot(121)
ax.plot([d['H_trace'][0] for d in alldata if d['H_trace'][0] < 1e7])
ax2 = fig.add_subplot(122)
ax2.hist([d['H_trace'][0] for d in alldata if d['H_trace'][0] < 1e7])
ax.set_title('trace of hessian at true theta', fontsize= 20)
ax.set_xlabel('different true thetas')
ax.set_ylabel('trace of hessian')


# test for the simple 1d
env=Simple1d(arg)
with torch.no_grad():
    actionls=[]
    positionls=[]
    vls=[]
    bls=[]
    rls=[]
    mcls=[]
    dcls=[]
    dls=[]
    thetals=[]
    goalls=[]
    env.reset()
    for i in range(999):
        action=agent(env.decision_info)
        env.step(action)
        actionls.append(action)
        positionls.append(env.s[0])
        vls.append(env.s[1])
        bls.append(env.b[0])
        rls.append(env.episode_reward)
        dls.append(env.get_distance()[1])
        thetals.append(env.theta)
        goalls.append(env.goalx)
i=0

previ=i
while rls[i]==0.:
    i+=1
print('reward',rls[i])
print('goal',goalls[previ:i][0]) if i!=previ else print(goalls[previ:i])
print('theta',thetals[previ:i][0]) if i!=previ else print(thetals[previ:i])
plt.plot(actionls[previ+1:i+1])
plt.plot(vls[previ:i])
plt.plot(dls[previ-1:i-1])
i+=1  

plt.plot(dls)
vls[previ:i]
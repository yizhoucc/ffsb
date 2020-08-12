import torch
import numpy as np
from matplotlib import pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn import random_projection


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


def change_of_basis(X, W):
  """
  Projects data onto a new basis.

  Args:
    X (numpy array of floats) : Data matrix each column corresponding to a
                                different random variable
    W (numpy array of floats) : new orthonormal basis columns correspond to
                                basis vectors

  Returns:
    (numpy array of floats)   : Data matrix expressed in new basis
  """

  Y = np.matmul(X, W)
  return Y


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
  score = change_of_basis(X, evectors)

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


def plot_inverse_trajectory(theta_trajectory,true_theta, phi=None, method='PCA'):
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

    # reshape the data into colume as features
    data_matrix=column_feature_data(theta_trajectory)
    if method=='PCA':
      # Perform PCA on the data matrix X
      score, evectors, evals = pca(data_matrix)


      fig = plt.figure(figsize=[8, 8])
      ax1 = fig.add_subplot()

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

      ax1.scatter(true_theta_pc[0],true_theta_pc[1],marker='o',c='',edgecolors='g')


      row_cursor=0
      while row_cursor<score.shape[0]-1:
          row_cursor+=1
          # plot point
          ax1.plot(score[row_cursor, 0], score[row_cursor, 1], '.', color=[.5, .5, .5])
          # plot arrow
          # fig = plt.figure(figsize=[8, 8])
          # ax1 = fig.add_subplot()
          # ax1.set_xlim([-1,1])
          # ax1.set_ylim([-1,1])
          ax1.quiver(score[row_cursor-1, 0], score[row_cursor-1, 1],
                  score[row_cursor, 0]-score[row_cursor-1, 0], score[row_cursor, 1]-score[row_cursor-1, 1],
                  angles='xy',color='g',scale=1, scale_units='xy')

      ax1.axis('equal')
      # plot hessian

      # plot log likelihood contour
      loss_function=compute_loss_wrapped(env, agent, true_theta.reshape(-1,1), np.array(phi).reshape(-1,1), trajectory_data=None, num_episodes=100)
      current_xrange=ax1.get_xlim()
      current_yrange=ax1.get_ylim()
      xyrange=[current_xrange,current_yrange]
      # ax1.contourf(X,Y,background_data)
      background_data=plot_background(ax1, xyrange,mu, evectors, loss_function, number_pixels=20)
      ax1.scatter(true_theta_pc[0],true_theta_pc[1],marker='o',c='',edgecolors='r')
      row_cursor=0
      while row_cursor<score.shape[0]-1:
          row_cursor+=1
          # plot point
          ax1.plot(score[row_cursor, 0], score[row_cursor, 1], '.', color=[.5, .5, .5])
          # plot arrow
          # fig = plt.figure(figsize=[8, 8])
          # ax1 = fig.add_subplot()
          # ax1.set_xlim([-1,1])
          # ax1.set_ylim([-1,1])
          ax1.quiver(score[row_cursor-1, 0], score[row_cursor-1, 1],
                  score[row_cursor, 0]-score[row_cursor-1, 0], score[row_cursor, 1]-score[row_cursor-1, 1],
                  angles='xy',color='g',scale=1, scale_units='xy')

    elif method=='random':
      transformer = random_projection.GaussianRandomProjection(n_components=9)    
      score=transformer.fit_transform(data_matrix)
      evectors=transformer.components_.transpose()
      
      fig = plt.figure(figsize=[8, 8])
      ax1 = fig.add_subplot()

      plt.xlabel('Projection basis vector 1')
      plt.ylabel('Projection basis vector 2')
      plt.title('Inverse for theta estimation in a random projection space')

      # plot true theta
      mu=np.mean(data_matrix,0)*0
      if type(true_theta)==list:
        true_theta=np.array(true_theta).reshape(-1)
        true_theta_pc=(true_theta)@evectors
      elif type(true_theta)==torch.nn.parameter.Parameter:
        true_theta_pc=(true_theta.detach().numpy().reshape(-1))@evectors
      else:
        true_theta_pc=(true_theta)@evectors

      ax1.scatter(true_theta_pc[0],true_theta_pc[1],marker='o',c='',edgecolors='g')


      row_cursor=0
      while row_cursor<score.shape[0]-1:
          row_cursor+=1
          # plot point
          ax1.plot(score[row_cursor, 0], score[row_cursor, 1], '.', color=[.5, .5, .5])
          # plot arrow
          # fig = plt.figure(figsize=[8, 8])
          # ax1 = fig.add_subplot()
          # ax1.set_xlim([-1,1])
          # ax1.set_ylim([-1,1])
          ax1.quiver(score[row_cursor-1, 0], score[row_cursor-1, 1],
                  score[row_cursor, 0]-score[row_cursor-1, 0], score[row_cursor, 1]-score[row_cursor-1, 1],
                  angles='xy',color='g',scale=1, scale_units='xy')

      ax1.axis('equal')
      # plot hessian

      # plot log likelihood contour
      loss_function=compute_loss_wrapped(env, agent, true_theta.reshape(-1,1), np.array(phi).reshape(-1,1), trajectory_data=None, num_episodes=100)
      current_xrange=ax1.get_xlim()
      current_yrange=ax1.get_ylim()
      current_yrange=(current_yrange[0]*10,current_yrange[1]*10)
      ax1.contourf(X,Y,background_data)
      xyrange=[current_xrange,current_yrange]
      background_data=plot_background(ax1, xyrange,mu, evectors, loss_function, number_pixels=10, method='random')
      ax1.scatter(true_theta_pc[0],true_theta_pc[1],marker='o',c='',edgecolors='r')
      row_cursor=0
      while row_cursor<score.shape[0]-1:
          row_cursor+=1
          # plot point
          ax1.plot(score[row_cursor, 0], score[row_cursor, 1], '.', color=[.5, .5, .5])
          # plot arrow
          # fig = plt.figure(figsize=[8, 8])
          # ax1 = fig.add_subplot()
          # ax1.set_xlim([-1,1])
          # ax1.set_ylim([-1,1])
          ax1.quiver(score[row_cursor-1, 0], score[row_cursor-1, 1],
                  score[row_cursor, 0]-score[row_cursor-1, 0], score[row_cursor, 1]-score[row_cursor-1, 1],
                  angles='xy',color='g',scale=1, scale_units='xy')


    plt.show()


def load_inverse_data(filename):
  'load the data pickle file, return the data dict'
  sys.path.insert(0, './inverse_data/')
  if filename[:-4]=='.pkl':
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

a=load_inverse_data('test new function2EP100updates100sample2IT2')
theta_trajectory=a['theta_estimations']
true_theta=a['true_theta']
theta_estimation=theta_trajectory[-1]
phi=np.array(a['phi'])
plot_inverse_trajectory(theta_trajectory,true_theta)


import torch
from stable_baselines import DDPG,TD3
from FireflyEnv import ffenv_new_cord
from Config import Config
arg = Config()
arg.NUM_SAMPLES=2
arg.NUM_EP = 10
arg.NUM_IT = 2 # number of iteration for gradient descent
arg.NUM_thetas = 1
arg.ADAM_LR = 0.07
arg.LR_STEP = 1.5
arg.LR_STOP = 5
arg.lr_gamma = 0.95
arg.PI_STD=1
arg.goal_radius_range=[0.05,0.2]
import policy_torch
baselines_mlp_model = TD3.load('trained_agent//TD_95gamma_mc_smallgoal_500000_9_24_1_6.zip')
agent = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[128,128])
env=ffenv_new_cord.FireflyAgentCenter(arg)
# true_theta=torch.nn.Parameter(env.reset_task_param())
# phi=true_theta.data.clone()

# filename='testnewfunc'
# single_theta_inverse(arg, env, agent, filename, 
#                 number_updates=50)                                                                                          


# def jacobian(y, x, create_graph=False):                                                               
#     jac = []                                                                                          
#     flat_y = y.reshape(-1)                                                                            
#     grad_y = torch.zeros_like(flat_y)                                                                 
#     for i in range(len(flat_y)):                                                                      
#         grad_y[i] = 1.                                                                                
#         grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
#         jac.append(grad_x.reshape(x.shape))                                                           
#         grad_y[i] = 0.                                                                                
#     return torch.stack(jac).reshape(y.shape + x.shape)                                                
                                                                                                      
# def hessian(y, x):                                                                                    
#     return jacobian(jacobian(y, x, create_graph=True), x)                                             
                                                                                                      
# def f(x):                                                                                             
#     return x * x * torch.arange(4, dtype=torch.float)                                                 
                                                                                                      
# x = torch.ones(4, requires_grad=True)                                                                 
# print(jacobian(f(x), x))                                                                              
# print(hessian(f(x), x)) 

from InverseFuncs import trajectory, getLoss

def compute_H(env, agent, theta_estimation, true_theta, phi, trajectory_data=None, num_episodes=100):
  _, actions, tasks=trajectory(agent, torch.Tensor(phi), torch.Tensor(true_theta), env, num_episodes)

  theta_estimation=torch.nn.Parameter(torch.Tensor(theta_estimation))
  loss = getLoss(agent, actions, tasks, torch.Tensor(phi), theta_estimation, env)
  grads = torch.autograd.grad(loss, theta_estimation, create_graph=True)[0]
  H = torch.zeros(9,9)
  for i in range(9):
      H[i] = torch.autograd.grad(grads[i], theta_estimation, retain_graph=True)[0].view(-1)
  # I = H.inverse()
  # E,V = scipy.linalg.eigh(H)
  # print(E)

  return H







def compute_loss(env, agent, theta_estimation, true_theta, phi, trajectory_data=None, num_episodes=100):
  if trajectory_data is None:
    _, actions, tasks=trajectory(agent, torch.Tensor(phi), torch.Tensor(true_theta), env, num_episodes)
  else:
    actions=trajectory_data['actions']
    tasks=trajectory_data['tasks']
  theta_estimation=torch.nn.Parameter(torch.Tensor(theta_estimation))
  loss = getLoss(agent, actions, tasks, torch.Tensor(phi), theta_estimation, env)
  return loss


def compute_loss_wrapped(env, agent, true_theta, phi, trajectory_data=None, num_episodes=100):
  new_function= lambda theta_estimation: compute_loss(env, agent, theta_estimation, true_theta, phi, num_episodes=100)
  return new_function


def plot_background(ax, xyrange,mu, evectors, loss_function, number_pixels=10, num_episodes=100, method='PCA'):
  background_data=np.zeros((number_pixels,number_pixels))
  X,Y=np.meshgrid(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels),np.linspace(xyrange[1][0],xyrange[1][1],number_pixels))
  if method =='PCA':
    for i,u in enumerate(np.linspace(xyrange[1][0],xyrange[1][1],number_pixels)):
      for j,v in enumerate(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels)):
        score=np.array([u,v])
        reconstructed_theta=score@evectors.transpose()[:2,:]+mu
        reconstructed_theta=reconstructed_theta.clip(1e-4,999)
        print(loss_function(torch.Tensor(reconstructed_theta).reshape(9,1)))
        background_data[i,j]=loss_function(torch.Tensor(reconstructed_theta).reshape(9,1))
  elif method =='random':
    for i,u in enumerate(np.linspace(xyrange[1][0],xyrange[1][1],number_pixels)):
      for j,v in enumerate(np.linspace(xyrange[0][0],xyrange[0][1],number_pixels)):
        score=np.array([u,v])
        distance=score-mu[:2]
        reconstructed_theta=mu+distance@np.linalg.inv(evectors)

        reconstructed_theta=reconstructed_theta.clip(1e-4,999)
        print(loss_function(torch.Tensor(reconstructed_theta).reshape(9,1)))
        background_data[i,j]=loss_function(torch.Tensor(reconstructed_theta).reshape(9,1))

  ax.contourf(X,Y,background_data)
  return background_data


# # # reshape the data into colume as features
# # data_matrix=column_feature_data(theta_trajectory)
# # # Perform PCA on the data matrix X
# # score, evectors, evals = pca(data_matrix)
# # # Plot the data projected into the new basis
# fig = plt.figure(figsize=[8, 8])
# ax1 = fig.add_subplot()
# # loss_function=compute_loss_wrapped(env, agent, true_theta, phi, trajectory_data=None, num_episodes=100)

# plot_background(ax1, [[true_theta_pc[0]-0.3,true_theta_pc[1]+0.3],[true_theta_pc[0]-0.3,true_theta_pc[1]+0.3]],true_theta, evectors, loss_function, number_pixels=15)

# # plt.imshow(background_data)
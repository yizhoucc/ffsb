import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import grad
from InverseFuncs import *
from collections import deque
import torch
import numpy as np
import time


def single_inverse(true_theta, arg, env, agent, x_traj,obs_traj, a_traj, filename, n,theta=None):
    tic = time.time()
    if theta is None:
        rndsgn = torch.sign(torch.randn(1,len(true_theta))).view(-1)
        purt= torch.Tensor([0.5,0.5,0.1,0.1,0.5,0.5,0.1,0.1,0.1]) # perturbation
        purt= torch.Tensor([2,2,2,2,2,2,2,2,2]) # log perturbation
        purt=torch.ones(9)
        purt=purt.uniform_()*1+purt*1.5
        # purt= torch.ones(9)*3 # log large perturbation

        theta = nn.Parameter(true_theta.data.clone()+rndsgn*purt)
        theta = theta_range(theta, arg.gains_range, arg.std_range, arg.goal_radius_range)  # keep inside of trained range

    else:
        theta = nn.Parameter(theta)
    #theta = nn.Parameter(reset_theta(arg.gains_range, arg.std_range, arg.goal_radius_range))
    ini_theta = theta.data.clone()
    print('initial theta: ',ini_theta)

    loss_log = deque(maxlen=arg.NUM_IT)
    loss_act_log = deque(maxlen=arg.NUM_IT)
    loss_obs_log = deque(maxlen=arg.NUM_IT)
    theta_log = deque(maxlen=arg.NUM_IT)
    gradient=deque(maxlen=arg.NUM_IT)
    optT = torch.optim.Adam([theta], lr=arg.ADAM_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optT, step_size=arg.LR_STEP, gamma=arg.lr_gamma) # decreasing learning rate x0.5 every 100steps
    prev_loss = 100000
    loss_diff = deque(maxlen=5)



    for it in tqdm(range(arg.NUM_IT)):
        loss, loss_act, loss_obs = getLoss(agent, x_traj, obs_traj,a_traj, theta, env, arg.gains_range, arg.std_range, arg.PI_STD, arg.NUM_SAMPLES)
        loss_log.append(loss.data)
        loss_act_log.append(loss_act.data)
        loss_obs_log.append(loss_obs.data)
        optT.zero_grad() #clears old gradients from the last step
        loss.backward(retain_graph=True) #computes the derivative of the loss w.r.t. the parameters using backpropagation
        gradient.append(theta.grad.data.clone())
        optT.step() # performing single optimize step: this changes theta
        # theta = theta_range(theta, arg.gains_range, arg.std_range, arg.goal_radius_range) # keep inside of trained range
        theta=theta_range_sigmoid(theta)
        # theta[6:8].data.copy_(true_theta[6:8].data) # fix obs  noise
        # theta[2:4].data.copy_(true_theta[2:4].data) # fix pro  noise
        # theta[4:8].data.copy_(true_theta[4:8].data) # fix obs gain and noise

        theta_log.append(theta.data.clone())
        if it < arg.LR_STOP:
            scheduler.step()


        loss_diff.append(torch.abs(prev_loss - loss))

        #if it > 5 and np.sum(loss_diff) < 100:
            #break
        prev_loss = loss.data


        if it%5 == 0:
            #print("num_theta:{}, num:{}, loss:{}".format(n, it, np.round(loss.data.item(), 6)))
            #print("num:{},theta diff sum:{}".format(it, 1e6 * (true_theta - theta.data.clone()).sum().data))
            print("num_theta:{}, num:{}, lr:{} loss:{}\n converged_theta:{}\n".format(n, it, scheduler.get_lr(),np.round(loss.data.item(), 6),theta.data.clone()))
            print("true theta:     {}".format(true_theta.data.clone()))
            print('\ngrad     ', theta.grad.data.clone())


    #
    loss, _, _ = getLoss(agent, x_traj,obs_traj, a_traj, theta, env, arg.gains_range, arg.std_range, arg.PI_STD, arg.NUM_SAMPLES)
    #print("loss:{}".format(loss))

    toc = time.time()
    #print((toc - tic)/60/60, "hours")


    grads = grad(loss, theta, create_graph=True)[0]
    H = torch.zeros(9,9)
    for i in range(9):
        H[i] = grad(grads[i], theta, retain_graph=True)[0]
    I = H.inverse()
    # sqrt(diagnal(H^-1))
    stderr = torch.sqrt(I.diag())

    # 1/sqrt(diag(H))
    stderr_ii = 1/torch.sqrt(torch.abs(H.diag()))




    result = {'true_theta': true_theta,
              'initial_theta': ini_theta,
              'theta': theta,
              'theta_log': theta_log,
              'loss_log': loss_log,
              'loss_act_log': loss_act_log,
              'loss_obs_log': loss_obs_log,
              'filename': filename,
              'num_theta': n,
              'converging_it': it,
              'duration': toc-tic,
              'arguments': arg,
              'stderr': stderr,
              'stderr_ii': stderr_ii,
              'grad':gradient
              }
    return result

# collections of functions for inverse control

import pickle
import warnings
import torch
from torch import nn
import numpy as np
from numpy import pi
import sys
# ----- single inverse dependencies.
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import grad
from InverseFuncs import *
from collections import deque
import torch
import numpy as np
import time
from torch import nn
from collections import deque
import tqdm
import torch 
import numpy as np
import itertools
from concurrent import futures
from monkey_functions import data_iter
from tqdm import tqdm

def reset_theta_log(gains_range, std_range, goal_radius_range, Pro_Noise = None, Obs_Noise = None):
    '''
    generate a random theta within arg range.
    '''
    pro_gains = torch.zeros(2)
    obs_gains = torch.zeros(2)

    pro_gains[0] = torch.zeros(1).uniform_(np.exp(gains_range[0]), np.exp(gains_range[1]))  # [proc_gain_vel]
    pro_gains[1] = torch.zeros(1).uniform_(np.exp(gains_range[2]), np.exp(gains_range[3]))  # [proc_gain_ang]
    obs_gains[0] = torch.zeros(1).uniform_(np.exp(gains_range[0]), np.exp(gains_range[1]))  # [obs_gain_vel]
    obs_gains[1] = torch.zeros(1).uniform_(np.exp(gains_range[2]), np.exp(gains_range[3]))  # [obs_gain_ang]
    goal_radius = torch.zeros(1).uniform_(np.exp(goal_radius_range[0]), np.exp(goal_radius_range[1]))

    if Pro_Noise is None:
       pro_noise_stds = torch.zeros(2)
       pro_noise_stds[0] = torch.zeros(1).uniform_(np.exp(std_range[0]), np.exp(std_range[1]))  # [proc_vel_noise]
       pro_noise_stds[1] = torch.zeros(1).uniform_(np.exp(std_range[2]), np.exp(std_range[3]))  # [proc_ang_noise]
    else:
        pro_noise_stds = Pro_Noise


    if Obs_Noise is None:
        obs_noise_stds = torch.zeros(2)
        obs_noise_stds[0] = torch.zeros(1).uniform_(np.exp(std_range[0]), np.exp(std_range[1]))  # [obs_vel_noise]
        obs_noise_stds[1] = torch.zeros(1).uniform_(np.exp(std_range[2]), np.exp(std_range[3]))  # [obs_ang_noise]
    else:
        obs_noise_stds = Obs_Noise

    theta = torch.cat([pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius])
    theta=torch.log(theta)
    return theta


def reset_theta_sig(gains_range= None, std_range= None, goal_radius_range= None, Pro_Noise = None, Obs_Noise = None):
    '''
    generate a random theta within arg range.
    '''
    theta=[]
    for i in range(9):
        theta.append(torch.zeros(1).uniform_(inverse_sigmoid(torch.Tensor([0.001])).item(),inverse_sigmoid(torch.Tensor([0.999])).item()))  # [proc_gain_vel]
    theta=torch.cat(theta)
    return theta


def reset_theta(gains_range, std_range, goal_radius_range, Pro_Noise = None, Obs_Noise = None):
    '''
    generate a random theta within arg range.
    '''
    pro_gains = torch.zeros(2)
    obs_gains = torch.zeros(2)

    pro_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  # [proc_gain_vel]
    pro_gains[1] = torch.zeros(1).uniform_(gains_range[2], gains_range[3])  # [proc_gain_ang]
    obs_gains[0] = torch.zeros(1).uniform_(gains_range[0], gains_range[1])  # [obs_gain_vel]
    obs_gains[1] = torch.zeros(1).uniform_(gains_range[2], gains_range[3])  # [obs_gain_ang]
    goal_radius = torch.zeros(1).uniform_(goal_radius_range[0], goal_radius_range[1])

    if Pro_Noise is None:
       pro_noise_stds = torch.zeros(2)
       pro_noise_stds[0] = torch.zeros(1).uniform_(std_range[0], std_range[1])  # [proc_vel_noise]
       pro_noise_stds[1] = torch.zeros(1).uniform_(std_range[2], std_range[3])  # [proc_ang_noise]
    else:
        pro_noise_stds = Pro_Noise


    if Obs_Noise is None:
        obs_noise_stds = torch.zeros(2)
        obs_noise_stds[0] = torch.zeros(1).uniform_(std_range[0], std_range[1])  # [obs_vel_noise]
        obs_noise_stds[1] = torch.zeros(1).uniform_(std_range[2], std_range[3])  # [obs_ang_noise]
    else:
        obs_noise_stds = Obs_Noise

    theta = torch.cat([pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius])
    return theta


def theta_range_sigmoid(theta):
    theta.data.clamp_(inverse_sigmoid(torch.Tensor([0.001])).item(),inverse_sigmoid(torch.Tensor([0.999])).item())
    return theta
    

def theta_range(theta, gains_range, std_range, goal_radius_range, Pro_Noise = None, Obs_Noise = None):

    '''
    clamp the given theta within the range
    '''

    if type(theta)==tuple:
        print("should use tensor theta not tuple! \n this method will be decraped")
        theta[0][0].data.clamp_(gains_range[0], gains_range[1])
        theta[0][1].data.clamp_(gains_range[2], gains_range[3])  # [proc_gain_ang]

        if Pro_Noise is None:
            theta[1][0].data.clamp_(std_range[0], std_range[1])  # [proc_vel_noise]
            theta[1][1].data.clamp_(std_range[2], std_range[3])  # [proc_ang_noise]
        else:
            theta[2:4].data.copy_(Pro_Noise.data)

        theta[2][0].data.clamp_(gains_range[0], gains_range[1])  # [obs_gain_vel]
        theta[2][1].data.clamp_(gains_range[2], gains_range[3])  # [obs_gain_ang]

        if Obs_Noise is None:
            theta[3][0].data.clamp_(std_range[0], std_range[1])  # [obs_vel_noise]
            theta[3][1].data.clamp_(std_range[2], std_range[3])  # [obs_ang_noise]
        else:
            theta[6:8].data.copy_(Obs_Noise.data)

        theta[4].data.clamp_(goal_radius_range[0], goal_radius_range[1])
        
        return theta
    
    else:
            
        theta[0].data.clamp_(gains_range[0], gains_range[1])
        theta[1].data.clamp_(gains_range[2], gains_range[3])  # [proc_gain_ang]

        if Pro_Noise is None:
            theta[2].data.clamp_(std_range[0], std_range[1])  # [proc_vel_noise]
            theta[3].data.clamp_(std_range[2], std_range[3])  # [proc_ang_noise]
        else:
            theta[2:4].data.copy_(Pro_Noise.data)

        theta[4].data.clamp_(gains_range[0], gains_range[1])  # [obs_gain_vel]
        theta[5].data.clamp_(gains_range[2], gains_range[3])  # [obs_gain_ang]

        if Obs_Noise is None:
            theta[6].data.clamp_(std_range[0], std_range[1])  # [obs_vel_noise]
            theta[7].data.clamp_(std_range[2], std_range[3])  # [obs_ang_noise]
        else:
            theta[6:8].data.copy_(Obs_Noise.data)

        theta[8].data.clamp_(goal_radius_range[0], goal_radius_range[1])


        return theta


def single_theta_inverse(arg, env, agent, filename, 
                number_updates=10,
                true_theta=None, phi=None,init_theta=None,
                trajectory_data=None, batchsize=50,
                use_H=False, is1d=False, fixed_param_ind=None, assign_true_param=None,
                task=None, gpu=True,
                action_var=0.1):

    env.agent_knows_phi=False
    env.presist_phi=True
    true_theta=torch.nn.Parameter(true_theta.clone().detach()) if true_theta is not None else torch.nn.Parameter(env.reset_task_param())
    true_theta.requires_grad=False
    phi=phi.clone().detach() if phi is not None else env.reset_task_param().clone().detach()
    phi=torch.nn.Parameter(phi)
    if init_theta is not None:
            init_theta = init_theta 
    else:
        init_theta=env.reset_task_param()
        while abs(torch.mean(init_theta-true_theta.clone().detach().data))<1e-3:
            init_theta=env.reset_task_param()
    if assign_true_param is not None:
        for i in range(len(init_theta)):
            if i in assign_true_param:
                init_theta[i]=true_theta[i]
    theta=torch.nn.Parameter(init_theta.clone().detach()) 
    if gpu:
        theta.cuda()
        phi.cuda()
        true_theta.cuda()

    print('initial theta: \n',theta)
    print('true theta: \n',true_theta)
    save_dict={'theta_estimations':[theta.data.clone().tolist()]}
    save_dict['true_theta']=true_theta.data.clone().tolist()
    save_dict['phi']=phi.data.clone().tolist()
    save_dict['inital_theta']=theta.data.clone().tolist()
    save_dict['Hessian']=[]
    save_dict['theta_cov']=[]
    save_dict['H_trace']=[]
    save_dict['theta_std']=[]
    save_dict['initial_lr']=arg.ADAM_LR
    save_dict['lr_stepsize']=arg.LR_STEP
    save_dict['sample_size']=arg.sample
    save_dict['batch_size']=arg.batch
    save_dict['updates_persample']=arg.NUM_IT
    save_dict['loss']=[]
    # prepare the logs
    loss_log = deque(maxlen=arg.NUM_IT)
    loss_act_log = deque(maxlen=arg.NUM_IT)
    loss_obs_log = deque(maxlen=arg.NUM_IT)
    theta_log = deque(maxlen=arg.NUM_IT)
    gradient=deque(maxlen=arg.NUM_IT)
    prev_loss = 100000
    loss_diff = deque(maxlen=5)
    env.reset(phi=phi)
    # tic = time.time()
    optT = torch.optim.Adam([theta], lr=arg.ADAM_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optT, step_size=arg.LR_STEP, gamma=arg.lr_gamma) 
    if trajectory_data is not None:
        allstates, allactions, alltasks=trajectory_data
        totalsamples=len(allstates)
    for num_update in range(number_updates):
        if trajectory_data is None:
            states, actions, tasks = trajectory(
                agent, phi, true_theta, env, NUM_EP=arg.batch,is1d=is1d,etask=task)
        else:
            sampleind=torch.randint(0,totalsamples,(batchsize,))
            states=[allstates[i] for i in sampleind]
            actions=[allactions[i] for i in sampleind]
            tasks=[alltasks[i] for i in sampleind]
        if arg.LR_STOP < scheduler.get_lr()[0]:
            scheduler.step()

        for it in range(arg.NUM_IT):
            loss = getLoss(agent, actions, tasks, phi, theta, env, action_var=action_var,
                    num_iteration=1, states=states, samples=arg.sample,gpu=gpu)
            loss_log.append(loss.clone().detach().cpu().item())
            optT.zero_grad() 
            loss.backward(retain_graph=True) 
            gradient.append(theta.grad.clone().detach())
            print('loss :', loss.clone().detach().cpu().item())
            save_dict['loss'].append(loss.clone().detach().cpu().item())
            print('gradient :', theta.grad.clone().detach())
            if fixed_param_ind is not None:
                for i in range(len(theta)):
                    if i in fixed_param_ind:
                        theta.grad[i]=0.
            optT.step() 
            theta.data[:,0]=theta.data[:,0].clamp(1e-3,999)
            theta_log.append(theta.clone().detach())

            # compute H
            if use_H:
                grads = torch.autograd.grad(loss, theta, create_graph=True)[0]
                H = torch.zeros(len(true_theta),len(true_theta))
                for i in range(len(true_theta)):
                    print('param', i)
                    H[i] = torch.autograd.grad(grads[i], theta, retain_graph=True)[0].view(-1)
                save_dict['Hessian'].append(H)
                save_dict['H_trace'].append(np.trace(H))
                cov=np.linalg.inv(H)
                save_dict['theta_cov'].append(cov)
                theta_std=np.sqrt(np.diag(cov)).tolist()
                save_dict['theta_std'].append(theta_std)
                print('H_trace: \n',np.trace(H))
                print('std err: \n',theta_std)
            # loss_diff.append(torch.abs(prev_loss - loss))
            # prev_loss = loss.data
            #     #print("num_theta:{}, num:{}, loss:{}".format(n, it, np.round(loss.data.item(), 6)))
            #     #print("num:{},theta diff sum:{}".format(it, 1e6 * (true_theta - theta.data.clone()).sum().data))
            print('number update: \n', num_update)
            print('converged_theta:\n{}'.format( theta.data.clone() ) )
            print('true theta:     \n{}'.format(true_theta.data.clone()))
            print('loss: \n',loss)
            print('learning rate: \n', scheduler.get_lr()[0])
                # print('\ngrad     ', theta.grad.data.clone())
            print('\n')
        save_dict['theta_estimations'].append(theta.data.clone().tolist())
        savename=('inverse_data/' + filename + '.pkl')
        torch.save(save_dict, savename)


def monkey_inverse(arg, env, agent, filename, 
                number_updates=10,
                phi=None,init_theta=None,
                trajectory_data=None, 
                batchsize=50,
                use_H=False, 
                is1d=False, 
                fixed_param_ind=None, 
                assign_true_param=None,
                task=None, 
                gpu=True,
                action_var=0.1,
                **kwargs):
    env.agent_knows_phi=False
    env.presist_phi=True
    if phi is None:
        raise ValueError('need to provide phi!')   
    if trajectory_data is None:
        raise ValueError('need to provide monkey trajectory!')   
    phi=phi.clone().detach()
    # phi=torch.nn.Parameter(phi)
    if init_theta is not None:
            init_theta = init_theta 
    else:
        init_theta=env.reset_task_param().view(-1,1)
    theta=torch.nn.Parameter(init_theta.clone().detach())

    if gpu:
        theta.cuda()
        phi.cuda()

    print('initial theta: \n',theta)
    save_dict={'theta_estimations':[theta.data.clone().tolist()]}
    save_dict['true_theta']=theta.data.clone().tolist()
    save_dict['phi']=phi.data.clone().tolist()
    save_dict['inital_theta']=theta.data.clone().tolist()
    save_dict['Hessian']=[]
    save_dict['theta_cov']=[]
    save_dict['H_trace']=[]
    save_dict['theta_std']=[]
    save_dict['initial_lr']=arg.ADAM_LR
    save_dict['lr_stepsize']=arg.LR_STEP
    save_dict['sample_size']=arg.sample
    save_dict['batch_size']=arg.batch
    save_dict['updates_persample']=arg.NUM_IT
    save_dict['loss']=[]
    save_dict['grad']=[]
    # prepare the logs
    loss_log = []
    theta_log = []
    gradient=[]
    env.reset(phi=phi)
    optT = torch.optim.Adam([theta], lr=arg.ADAM_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optT, step_size=arg.LR_STEP, gamma=arg.lr_gamma) 
    mkstates, mkactions, mktasks=trajectory_data
    for epoch in range(arg.NUM_IT):
        batchsize=int(epoch/arg.NUM_IT*(len(mktasks)-arg.batch))+arg.batch
        for states, actions, tasks in data_iter(batchsize,mkstates,mkactions,mktasks):
            if len(tasks)<=2:
                break
            if arg.LR_STOP < scheduler.get_lr()[0]*arg.LR_STEP:
                scheduler.step()
            for it in range(number_updates):
                loss = monkeyloss(agent, actions, tasks, phi, theta, env, action_var=action_var,
                        num_iteration=1, states=states, samples=arg.sample,gpu=gpu)
                loss_log.append(loss.clone().detach().cpu().item())
                optT.zero_grad() 
                tik=time.time()
                loss.backward(retain_graph=True) 
                print('backward time {:.0f}'.format(time.time()-tik))
                gradient.append(theta.grad.clone().detach())
                print('loss :', loss.clone().detach().cpu().item())
                save_dict['loss'].append(loss.clone().detach().cpu().item())
                print('gradient :', theta.grad.clone().detach())
                save_dict['grad'].append(theta.grad.clone().detach().cpu())
                if fixed_param_ind is not None:
                    for i in range(len(theta)):
                        if i in fixed_param_ind:
                            theta.grad[i]=0.
                if torch.all(torch.isnan(gradient[-1])==False):
                    optT.step()         
                theta.data[:,0]=theta.data[:,0].clamp(1e-3,999)
                theta_log.append(theta.clone().detach())

                # compute H
                if use_H:
                    grads = torch.autograd.grad(loss, theta, create_graph=True)[0]
                    H = torch.zeros(len(theta),len(theta))
                    for i in range(len(theta)):
                        print('param', i)
                        H[i] = torch.autograd.grad(grads[i], theta, retain_graph=True)[0].view(-1)
                    save_dict['Hessian'].append(H)
                    save_dict['H_trace'].append(np.trace(H))
                    cov=np.linalg.inv(H)
                    save_dict['theta_cov'].append(cov)
                    theta_std=np.sqrt(np.diag(cov)).tolist()
                    save_dict['theta_std'].append(theta_std)
                    print('H_trace: \n',np.trace(H))
                    print('std err: \n',theta_std)

                print('epoch: ', epoch, 'batchsize',batchsize,'\n')
                print('converged_theta:\n{}'.format( theta.clone().detach()) )
                print('learning rate: \n', scheduler.get_lr()[0])
                print('\n')
                del loss
            save_dict['theta_estimations'].append(theta.clone().detach().tolist())
            savename=('inverse_data/' + filename + '.pkl')
            torch.save(save_dict, savename)


def monkeyloss_(agent=None, 
            actions=None, 
            tasks=None, 
            phi=None, 
            theta=None, 
            env=None,
            num_iteration=1, 
            states=None, 
            samples=1, 
            gpu=False,
            action_var=0.1,
            debug=False):
    if gpu:
        logPr = torch.zeros(1).cuda()[0] #torch.FloatTensor([])
    else:
        logPr = torch.zeros(1)[0] #torch.FloatTensor([])
    
    def _wrapped_call(ep, task):     
        logPr_ep = torch.zeros(1).cuda()[0] if gpu else torch.zeros(1)[0]   
        for sample_index in range(samples): 
            mkactionep = actions[ep]
            if mkactionep==[] or mkactionep.shape[0]==0:
                continue
            env.reset(theta=theta, phi=phi, goal_position=task, vctrl=mkactionep[0][0],wctrl=mkactionep[0][1])
            numtime=len(mkactionep[1:])

            # compare mk data and agent actions
            for t,mk_action in enumerate(mkactionep[1:]): # use a t and s t (treat st as st+1)
                # agent's action
                action = agent(env.decision_info)
                # agent's obs, last step obs doesnt matter.
                if t<len(states[ep])-1:
                    if type(states[ep])==list:
                        nextstate=states[ep][1:][t]
                    elif type(states[ep])==torch.Tensor:
                        nextstate=states[ep][1:][t].view(-1,1)
                    obs=env.observations(nextstate)
                    # agent's belief
                    env.b, env.P=env.belief_step(env.b,env.P, obs, torch.tensor(mk_action).view(1,-1))
                    previous_action=mk_action # current action is prev action for next time
                    env.trial_timer+=1
                    env.decision_info=env.wrap_decision_info(
                                                previous_action=torch.tensor(previous_action), 
                                                time=env.trial_timer)
                # loss
                action_loss = -1*logll(torch.tensor(mk_action),action,std=np.sqrt(action_var))
                obs_loss = -1*logll(error=env.obs_err(), std=theta[4:6].view(1,-1))
                logPr_ep = logPr_ep + action_loss.sum() + obs_loss.sum()
                del action_loss
                del obs_loss
            # if agent has not stop, compare agent action vs 0,0
            agentstop=torch.norm(action)<env.terminal_vel
            while not agentstop and env.trial_timer<40:
                action = agent(env.decision_info)
                agentstop=torch.norm(action)<env.terminal_vel
                obs=(torch.tensor([0.5,pi/2])*action+env.obs_err()).t()
                env.b, env.P=env.belief_step(env.b,env.P, obs, torch.tensor(action).view(1,-1))
                # previous_action=torch.tensor([0.,0.]) # current action is prev action for next time
                previous_action=action
                env.trial_timer+=1
                env.decision_info=env.wrap_decision_info(
                previous_action=torch.tensor(previous_action), 
                                            time=env.trial_timer)
                # loss
                action_loss = -1*logll(torch.tensor(torch.zeros(2)),action,std=np.sqrt(action_var))
                obs_loss = -1*logll(error=env.obs_err(), std=theta[4:6].view(1,-1))
                logPr_ep = logPr_ep + action_loss.sum() + obs_loss.sum()
                del action_loss
                del obs_loss

        return logPr_ep/samples/env.trial_timer.item()
    
    tik=time.time()
    loglls=[]
    for ep, task in enumerate(tasks):
        logPr_ep=_wrapped_call(ep, task)
        logPr += logPr_ep
        loglls.append(logPr_ep)
        del logPr_ep
    regularization=torch.sum(1/(theta+1e-4))
    print('calculate loss time {:.0f}'.format(time.time()-tik))
    if debug:
        return loglls
    return logPr/len(tasks)+0.01*regularization


def monkeyloss_sim(agent=None, 
            actions=None, 
            tasks=None, 
            phi=None, 
            theta=None, 
            env=None,
            num_iteration=1, 
            states=None, 
            samples=1, 
            gpu=False,
            action_var=0.1,):
    if gpu:
        logPr = torch.zeros(1).cuda()[0] #torch.FloatTensor([])
    else:
        logPr = torch.zeros(1)[0] #torch.FloatTensor([])
    
    def _wrapped_call(ep, task):     
        logPr_ep = torch.zeros(1).cuda()[0] if gpu else torch.zeros(1)[0]   
        for sample_index in range(samples): 
            mkactionep = actions[ep] 
            if mkactionep==[] or mkactionep.shape[0]<=3:
                continue
            env.reset(theta=theta, phi=phi, goal_position=task, vctrl=states[ep][0][3]/env.phi[0],wctrl=states[ep][0][4]/env.phi[1])
            numtime=len(mkactionep)

            # compare mk data and agent actions
            for t,mk_action in enumerate(mkactionep): # use a t and s t (treat st as st+1)
                # agent's action
                action = agent(env.decision_info)
                # agent's obs, last step obs doesnt matter.
                if t<len(states[ep])-1:
                    if type(states[ep])==list:
                        nextstate=states[ep][1:][t]
                    elif type(states[ep])==torch.Tensor:
                        nextstate=states[ep][1:][t].view(-1,1)
                    obs=env.observations(nextstate)
                    # agent's belief
                    env.b, env.P=env.belief_step(env.b,env.P, obs, torch.tensor(mk_action).view(1,-1))
                    previous_action=mk_action # current action is prev action for next time
                    env.trial_timer+=1
                    env.decision_info=env.wrap_decision_info(
                                                previous_action=torch.tensor(previous_action), 
                                                time=env.trial_timer)
                # loss
                action_loss = -1*logll(torch.tensor(mk_action),action,std=np.sqrt(action_var))
                obs_loss = -1*logll(error=env.obs_err(), std=theta[4:6].view(1,-1))
                logPr_ep = logPr_ep + action_loss.sum() + obs_loss.sum()
                del action_loss
                del obs_loss
            # if agent has not stop, compare agent action vs 0,0
            agentstop=torch.norm(action)<env.terminal_vel
            while not agentstop and env.trial_timer<40:
                action = agent(env.decision_info)
                agentstop=torch.norm(action)<env.terminal_vel
                obs=(torch.tensor([0.5,pi/2])*action+env.obs_err()).t()
                env.b, env.P=env.belief_step(env.b,env.P, obs, torch.tensor(action).view(1,-1))
                # previous_action=torch.tensor([0.,0.]) # current action is prev action for next time
                previous_action=action
                env.trial_timer+=1
                env.decision_info=env.wrap_decision_info(
                previous_action=torch.tensor(previous_action), 
                                            time=env.trial_timer)
                # loss
                action_loss = -1*logll(torch.tensor(torch.zeros(2)),action,std=np.sqrt(action_var))
                obs_loss = -1*logll(error=env.obs_err(), std=theta[4:6].view(1,-1))
                logPr_ep = logPr_ep + action_loss.sum() + obs_loss.sum()
                del action_loss
                del obs_loss

        return logPr_ep/samples/env.trial_timer.item()

    tik=time.time()
    for ep, task in enumerate((tasks)):
        logPr_ep=_wrapped_call(ep, task)
        logPr += logPr_ep
        del logPr_ep
    regularization=torch.sum(1/(theta+1e-4))
    print('calculate loss time {:.0f}'.format(time.time()-tik))
    return logPr/len(tasks)+0.01*regularization


def monkeyloss(agent=None, 
            actions=None, 
            tasks=None, 
            phi=None, 
            theta=None, 
            env=None,
            num_iteration=1, 
            states=None, 
            samples=1, 
            gpu=False,
            action_var=0.1):
    if gpu:
        logPr = torch.zeros(1).cuda()[0] #torch.FloatTensor([])
    else:
        logPr = torch.zeros(1)[0] #torch.FloatTensor([])
    
    def _wrapped_call(ep, task):     
        logllep = torch.zeros(1).cuda()[0] if gpu else torch.zeros(1)[0]   
        for sample_index in range(samples): 
            mkactionep = actions[ep][1:] # a0 is inital condition
            if mkactionep==[] or mkactionep.shape[0]==0:
                continue
            env.reset(theta=theta, phi=phi, goal_position=task, vctrl=mkactionep[0][0],wctrl=mkactionep[0][1])
            logllsample=torch.zeros(1).cuda()[0] if gpu else torch.zeros(1)[0]   
            agentstop=False
            while True:
                action = torch.zeros(2) if agentstop else agent(env.decision_info)
                agentstop=torch.norm(action)<env.terminal_vel
                mkstop=env.trial_timer>=len(mkactionep[1:])
                # both not stop, a* vs a, obs mk
                # mk stop, a* vs 0, obs self
                # agent stop a* vs a, let agent corrects
                if not mkstop:
                    if type(states[ep])==list:
                        nextstate=states[ep][1:][int(env.trial_timer)]
                    elif type(states[ep])==torch.Tensor:
                        nextstate=states[ep][1:][int(env.trial_timer)].view(-1,1)
                    mk_action=mkactionep[int(env.trial_timer)]
                    obs=env.observations(nextstate)
                    env.b, env.P=env.belief_step(env.b,env.P, obs, torch.tensor(mk_action).view(1,-1))
                    previous_action=mk_action
                    # current action is prev action for next time
                    env.trial_timer+=1
                    env.decision_info=env.wrap_decision_info(
                                            previous_action=torch.tensor(previous_action), 
                                            time=env.trial_timer)
                    # loss
                    action_loss = -1*logll(torch.tensor(mk_action),action,std=np.sqrt(action_var))
                    obs_loss = -1*logll(error=env.obs_err(), std=theta[4:6].view(1,-1))
                    logllsample = logllsample + action_loss.sum() + obs_loss.sum()
                    del action_loss
                    del obs_loss

                elif mkstop and not agentstop and env.trial_timer<40:
                    # obs self
                    obs=(torch.tensor([0.5,pi/2])*action+env.obs_err()).t()
                    env.b, env.P=env.belief_step(env.b,env.P, obs, torch.tensor(action).view(1,-1))
                    previous_action=action
                    env.trial_timer+=1
                    env.decision_info=env.wrap_decision_info(
                    previous_action=torch.tensor(previous_action), 
                                                time=env.trial_timer)
                    # loss, 0 vs a*
                    action_loss = -1*logll(torch.tensor(torch.zeros(2)),action,std=np.sqrt(action_var))
                    obs_loss = -1*logll(error=env.obs_err(), std=theta[4:6].view(1,-1))
                    logllsample = logllsample + action_loss.sum() + obs_loss.sum()
                    del action_loss
                    del obs_loss
                else:
                    break
            logllsample=logllsample/env.trial_timer.item()
            logllep+=logllsample
        return logllep/samples


    tik=time.time()
    for ep, task in enumerate(tasks):
        logPr_ep=_wrapped_call(ep, task)
        logPr += logPr_ep
        del logPr_ep
    regularization=torch.sum(1/(theta+1e-4))
    print('calculate loss time {:.0f}'.format(time.time()-tik))
    return logPr/len(tasks)+0.01*regularization


def monkeyloss_fixobs(agent=None, 
            actions=None, 
            tasks=None, 
            phi=None, 
            theta=None, 
            env=None,
            num_iteration=1, 
            states=None, 
            gpu=False,
            action_var=0.1,
            obs_trajectory=None,
            usestop=True):

    def _wrapped_call(ep, task,trial_obs):     
        samples=len(trial_obs)
        logPr_ep = torch.zeros(1).cuda()[0] if gpu else torch.zeros(1)[0]  

        for obs_sample in trial_obs: 
            env.debug=True
            mkactionep = actions[ep][1:] # a0 is inital condition
            if mkactionep==[] or mkactionep.shape[0]==0:
                continue
            env.reset(theta=theta, phi=phi, goal_position=task, vctrl=mkactionep[0][0],wctrl=mkactionep[0][1])
            env.obs_traj=obs_sample

            # compare mk data and agent actions
            for t,mk_action in enumerate(mkactionep[1:]): # use a t and s t (treat st as st+1)
                # agent's action
                action = agent(env.decision_info)
                # agent's obs, last step obs doesnt matter.
                if type(states[ep])==list:
                    nextstate=states[ep][1:][t]
                elif type(states[ep])==torch.Tensor:
                    nextstate=states[ep][1:][t].view(-1,1)
                obs=env.observations(nextstate)
                # agent's belief
                env.b, env.P=env.belief_step(env.b,env.P, obs, torch.tensor(mk_action).view(1,-1))
                previous_action=mk_action # current action is prev action for next time
                env.trial_timer+=1
                env.decision_info=env.wrap_decision_info(previous_action=previous_action, time=env.trial_timer)
                # loss
                action_loss = -1*logll(torch.tensor(mk_action),action,std=np.sqrt(action_var))
                obs_loss = -1*logll(error=env.obs_err(), std=theta[4:6].view(1,-1))
                logPr_ep = logPr_ep + action_loss.sum() + obs_loss.sum()
                del action_loss
                del obs_loss
            # if mk stops but agent has not stop, treat mk action as 0,0
            agentstop=torch.norm(action)<env.terminal_vel
            while not agentstop and env.trial_timer<40 and usestop:
                action = agent(env.decision_info)
                agentstop=torch.norm(action)<env.terminal_vel
                obs=(torch.tensor([0.5,pi/2])*action+env.obs_err()).t()
                env.b, env.P=env.belief_step(env.b,env.P, obs, torch.tensor(action).view(1,-1))
                # previous_action=torch.tensor([0.,0.]) # current action is prev action for next time
                previous_action=action
                env.trial_timer+=1
                env.decision_info=env.wrap_decision_info(
                previous_action=torch.tensor(previous_action), 
                                            time=env.trial_timer)
                # loss
                action_loss = -1*logll(torch.tensor(torch.zeros(2)),action,std=np.sqrt(action_var))
                obs_loss = -1*logll(error=env.obs_err(), std=theta[4:6].view(1,-1))
                logPr_ep = logPr_ep + action_loss.sum() + obs_loss.sum()
                del action_loss
                del obs_loss

        return logPr_ep/samples/env.trial_timer.item()

    if gpu:
        logPr = torch.zeros(1).cuda()[0] #torch.FloatTensor([])
    else:
        logPr = torch.zeros(1)[0] #torch.FloatTensor([])
    
    tik=time.time()
    for ep, task in enumerate((tasks)):
        trial_obs=obs_trajectory[ep]
        logPr_ep=_wrapped_call(ep, task,trial_obs)
        logPr += logPr_ep
        del logPr_ep
    regularization=torch.sum(1/(theta+1e-4))
    print('calculate loss time {:.0f}'.format(time.time()-tik))
    return logPr/len(tasks)+0.01*regularization


def trajectory(agent, phi, theta, env, NUM_EP=100, 
            is1d=False, if_obs=False,etask=None, estate=None, 
            eaction=None,test_theta=None, return_belief=False):
    if etask is not None:
        if len(etask)==1:
            etask=etask*NUM_EP

        #------prepare saving vars-----
    states = [] # true location
    actions = [] # action
    tasks=[] # vars that define an episode
    obs=[]
    belief=[]
    phi=torch.nn.Parameter(phi)
    theta=torch.nn.Parameter(theta)
    episode = 0

    if eaction is None: # just get expert trajectory
        with torch.no_grad():
            while episode < NUM_EP:
                if etask is not None:
                    env.reset(phi=phi,theta=theta,goal_position=etask[episode][0])
                else:
                    env.reset(phi=phi,theta=theta)            
                states_ep = []
                actions_ep = []
                obs_ep=[]
                belief_ep=[]
                task_ep=[[env.goalx, env.goaly], env.phi] if not is1d else [env.goalx, env.phi]
                done=False
                while not done:
                    if hasattr(agent,'actor'):
                        if test_theta is not None:
                                env.decision_info[-len(test_theta):]=test_theta
                        action = agent.actor(env.decision_info)[0]
                    else:
                        action = agent(env.decision_info)[0]
                    actions_ep.append(action)
                    states_ep.append(env.s) 
                    _,done=env(action,task_param=theta) 
                    if obs:
                        obs_ep.append(env.o)
                    if return_belief:
                        belief_ep.append(env.b)
                belief.append(belief_ep)
                states.append(states_ep)
                obs.append(obs_ep)
                actions.append(actions_ep)
                tasks.append(task_ep)
                episode +=1
    else: # expert action, agent inverse
        while episode < NUM_EP:
            env.reset(phi=phi,theta=theta,goal_position=etask[episode][0])
            states_ep = []
            actions_ep = []
            belief_ep=[]
            task_ep=[[env.goalx, env.goaly], env.phi] if not is1d else [env.goalx, env.phi]
            t=0
            states_ep.append(env.s)
            while t<len(eaction[episode]):
                if hasattr(agent,'actor'):
                    action = agent.actor(env.decision_info)[0]
                else:
                    action = agent(env.decision_info)[0]
                if estate is not None:
                    if type(estate[episode])==list:
                        if  t+1<len(estate[episode]):
                            _,done=env(eaction[episode][t],task_param=theta,state=estate[episode][t+1]) 
                        else:
                            pass
                            # print(t+1, len(eaction[episode]))
                        pass                
                    else:
                        if  t+1<(estate[episode].shape[1]):
                            _,done=env(eaction[episode][t],task_param=theta,state=estate[episode][:,t+1].view(-1,1)) 
                else:
                    env(eaction[episode][t],task_param=theta) 
                states_ep.append(env.s) 
                actions_ep.append(action)
                belief_ep.append(env.b)
                t=t+1
            states.append(states_ep)
            actions.append(actions_ep)
            belief.append(belief_ep)
            tasks.append(task_ep)
            episode +=1 
    if if_obs:
        return states, actions, tasks, obs
    if return_belief:
        return states, actions, tasks,belief
    else:
        return states, actions, tasks


def trajectory_inverse(agent, phi, theta, env, NUM_EP=100, 
            etask=None, estate=None, 
            eaction=None,**kwargs):
    # new version, optimzed for inverse
    # change, only calculate belief (not running whole dynamic)
    if etask is not None:
        warnings.warn('need to provide inital condition!')
        #------prepare saving vars-----
    actions = [] # action
    obs=[]
    phi=torch.nn.Parameter(phi)
    theta=torch.nn.Parameter(theta)
    episode = 0
    if eaction is None: 
        warnings.warn('need to provide monkey actions!')
    else: 
        while episode < NUM_EP:
            # get initial condition
            env.reset(phi=phi,theta=theta,goal_position=etask[episode][0])
            actions_ep = []
            t=0
            while t<len(eaction[episode]):
                # choose action
                if hasattr(agent,'actor'):
                    action = agent.actor(env.decision_info)[0]
                else:
                    action = agent(env.decision_info)[0]
                if type(estate[episode])==list:
                    nextstate=estate[episode][t+1]
                    # if  t+1<len(estate[episode]):
                    #         _,done=env(eaction[episode][t],task_param=theta,state=estate[episode][t+1])                 
                elif type(estate[episode])==torch.tensor:
                    nextstate=estate[episode][:,t+1].view(-1,1)

                    # if  t+1<(estate[episode].shape[1]):
                    #     _,done=env(eaction[episode][t],task_param=theta,state=estate[episode][:,t+1].view(-1,1)) 
                else:
                    warnings.warn('need to provide monkey states!')
                obs=env.observations(nextstate)
                env.b=env.belief_step(env.b,env.P, obs, eaction[episode][t])
                actions_ep.append(action)
                t=t+1
            actions.append(actions_ep)
            episode +=1 
    return None, actions, None


def trajectory_(agent, theta, env, arg, gains_range, std_range, goal_radius_range, NUM_EP):
    '''
    get NUM_EP=500 trials of (x,o,b,a|phi,theta=phi)
    '''
    pro_gains, pro_noise_stds, obs_gains, obs_noise_stds,  goal_radius = torch.split(theta.view(-1), 2)

    x_traj = [] # true location
    obs_traj =[] # observation
    a_traj = [] # action
    b_traj = []

    env.pro_gains = pro_gains
    env.pro_noise_stds = pro_noise_stds
    env.goal_radius = goal_radius
    env.obs_gains=obs_gains
    env.obs_noise_stds=obs_noise_stds

    env.reset()     # apply the true theta to env

    env.belief = env.Breshape(theta=theta)
    episode = 0
    tot_t = 0

    #while tot_t <= TOT_T:
    while episode <= NUM_EP: #defalt 500
        episode +=1
        t = torch.zeros(1)
        x_traj_ep = []
        obs_traj_ep = []
        a_traj_ep = []
        b_traj_ep = []

        while t < arg.EPISODE_LEN: # for a single FF

            action = agent(env.belief)[0] # TODO action rounded, not precise
            # action=agent.predict(env.belief_state)[0]
            a_traj_ep.append(action)
            x_traj_ep.append(env.x)
            # env(action)
            env(action.view(-1)) # so env.x env.belief updated by action and next o
            obs_traj_ep.append(env.o) # the same next obs used to update belief

            # check time limit
            TimeEnd = (t+1 == arg.EPISODE_LEN) # if the monkey can't catch the firefly in EPISODE_LEN, reset the game.
            mask = torch.tensor([1 - float(TimeEnd)]) # mask = 0: episode is over

            tot_t += 1.
            t += 1
            if env.stop:
                pass
            if env.stop or TimeEnd:  # if the monkey stops or pass the time limit, start the new firefly
                
                env.pro_gains = pro_gains
                env.pro_noise_stds = pro_noise_stds
                env.goal_radius = goal_radius
                env.obs_gains=obs_gains
                env.obs_noise_stds=obs_noise_stds

                env.reset()

                break
        x_traj.append(x_traj_ep)
        obs_traj.append(obs_traj_ep)
        a_traj.append(a_traj_ep)
        b_traj.append(b_traj_ep)
    return x_traj, obs_traj, a_traj, b_traj


def get_loss_sorted(agent, actions, tasks, phi, theta, env, num_iteration=1, states=None, samples=100):
    
    total_loss = torch.zeros(1).cuda()[0]
    is1d=True if len([tasks[0][0]])==1 else False
    astates, aactions, atasks = trajectory(
                agent, phi, theta, env, NUM_EP=len(actions),is1d=is1d,etask=tasks, eaction=actions,estate=states)
    assert atasks==tasks

    ep_lengths = [len(a) for a in actions]
    for t in range(min(ep_lengths)):
        agent_action_t=torch.cat([a[t] for a in aactions]).sort()[0]
        expert_action_t=torch.cat([a[t] for a in actions]).sort()[0]
        loss_t=mse_loss(agent_action_t,expert_action_t).sum()
    total_loss=total_loss+loss_t
    return total_loss
    # agent_action_ep=torch.cat(aactions[0])
    # expert_action_ep=torch.cat(actions[0])


    # for episode in len(aactions):
    #     agent_action_ep=torch.cat(aactions[episode])
    #     expert_action_ep=torch.cat(actions[episode])
    # for t in len(expert_action_ep):
    #     a1=torch.cat([a[t] for a in aactions])
    #     e1=torch.cat([a[t] for a in actions])

    # a1=torch.cat([a[15] for a in aactions])
    # e1=torch.cat([a[15] for a in actions])
    # torch.nn.KLDivLoss()(a1.log(),e1)

    return logPr


def get_loss(agent, eactions, etasks, phi, theta, env, num_iteration=1, estates=None, samples=100,gpu=True):
    total_loss = torch.zeros(1).cuda()[0] if gpu else torch.zeros(1)[0]
    is1d=True if len([etasks[0][0]])==1 else False
    astates, aactions, atasks = trajectory(
                agent, phi, theta, env, NUM_EP=len(eactions),
                is1d=is1d,etask=etasks, eaction=eactions,estate=estates)
    for episode, episode_actions in enumerate(eactions):
        episode_loss=torch.zeros(1).cuda()[0] if gpu else torch.zeros(1)[0]
        for t, action in  enumerate(episode_actions):
            action_loss = mse_loss(action, aactions[episode][t])
            episode_loss = episode_loss + action_loss.sum()
        total_loss = total_loss  + episode_loss
    return total_loss


def get_loss_(agent, actions, tasks, phi, theta, env, num_iteration=1, states=None, samples=100):
    
    '''
    from the trajectories, compute the loss for current estimation of theta
    the theta input is the estimation of theta.
    we dont need true theta, as we are not supposed to know the true theta.
    phi, if given phi, make sure the env.phi is phi
    '''
    logPr = torch.zeros(1).cuda()[0] #torch.FloatTensor([])

    for num_it in range(samples): 
        # repeat for number of episodes in the trajectories
        for ep, task in enumerate(tasks): 
            # initialize the loss var
            logPr_ep = torch.zeros(1).cuda()[0]
            pos=task[0]
            # take out the teacher trajectories for one episode
            a_traj_ep = actions[ep]  
            # reset monkey's internal model, using the task parameters, estimation of theta
            env.reset(theta=theta, phi=phi, goal_position=pos)  
            # repeat for steps in episode
            for t,teacher_action in enumerate(a_traj_ep): 
                if hasattr(agent,'actor'):
                    action = agent.actor(env.decision_info)[0] 
                else:
                    action = agent(env.decision_info)[0] 
                if states is None:
                    decision_info,_=env(teacher_action,task_param=theta) 
                else:
                    if t+1<len(states[ep]):
                        decision_info,_=env(teacher_action,task_param=theta, state=states[ep][t+1])            
                action_loss = mse_loss(action, teacher_action)
                obs_loss = mse_loss(env.observations(states[ep][t]) , env.observations_mean(states[ep][t]))
                logPr_ep = logPr_ep + action_loss.sum() + obs_loss.sum()
                # return logPr_ep
            logPr = logPr  + logPr_ep
    return logPr


def getLoss(agent, actions, tasks, phi, theta, env, num_iteration=1, states=None, samples=100, gpu=True,action_var=0.1):
    
    '''
    from the trajectories, compute the loss for current estimation of theta
    the theta input is the estimation of theta.
    we dont need true theta, as we are not supposed to know the true theta.
    phi, if given phi, make sure the env.phi is phi
    '''
    if gpu:
        logPr = torch.zeros(1).cuda()[0] #torch.FloatTensor([])
    else:
        logPr = torch.zeros(1)[0] #torch.FloatTensor([])

    for ep, task in enumerate(tasks): 
        # repeat for number of episodes in the trajectories
        for sample_index in range(samples): 
            # initialize the loss var
            logPr_ep = torch.zeros(1).cuda()[0] if gpu else torch.zeros(1)[0]
            pos=task[0]
            # take out the teacher trajectories for one episode
            a_traj_ep = actions[ep]  
            # reset monkey's internal model, using the task parameters, estimation of theta
            env.reset(theta=theta, phi=phi, goal_position=pos)#, initv=a_traj_ep[0][0])  
            # repeat for steps in episode
            for t,teacher_action in enumerate(a_traj_ep): 
                if hasattr(agent,'actor'):
                    action = agent.actor(env.decision_info)[0]
                else:
                    action = agent(env.decision_info)[0]
                if states is None:
                    decision_info,_=env(teacher_action,task_param=theta)
                else:
                    if type(states[ep])==list:
                        if t+1<len(states[ep]):
                            decision_info,_=env(teacher_action,task_param=theta, state=states[ep][t+1])            
                    elif type(states[ep])==torch.Tensor:
                        if t+1<(states[ep]).shape[1]:
                            decision_info,_=env(teacher_action,task_param=theta, state=states[ep][:,t+1].view(-1,1))            

                action_loss = logll(torch.tensor(teacher_action),action,std=np.sqrt(action_var))
                obs_loss = logll(env.o, env.observations_mean(env.s), std=theta.clone().detach()[4:6])
                logPr_ep = logPr_ep + action_loss.sum() + obs_loss.sum()
                action_loss.detach()
                obs_loss.detach()

            logPr = logPr + logPr_ep
            logPr_ep.detach()
    return logPr


def logll(true=None, estimate=None, std=0.3, error=None, prob=False):
    # print(error)
    var=std**2
    if error is not None: # use for point eval, obs
        g=lambda x: 1/torch.sqrt(2*pi*torch.ones(1))*torch.exp(-0.5*x**2/var)
        z=1/g(torch.zeros(1)+1e-8)
        loss=torch.log(g(error)*z+1e-8)
    else: # use for distribution eval, aciton
        c=torch.abs(true-estimate)
        gi=lambda x: -(torch.erf(x/torch.sqrt(torch.tensor([2]))/std)-1)/2
        loss=torch.log(gi(c)*2+1e-16)
    if prob:
        return torch.exp(loss)
    return loss

def logllv1(true=None, estimate=None, std=0.3, error=None):
    # loss=1/std/torch.sqrt(torch.tensor([6.28]))*(torch.exp(-0.5*((estimate-true)/std)**2))
    # -torch.log(loss+0.001)
    if error is not None:
        loss=torch.log(torch.sqrt(torch.tensor([2*pi]))*std)+1/2*(error/std)**2
    else:
        loss=torch.log(torch.sqrt(torch.tensor([2*pi]))*std)+1/2*((estimate-true)/std)**2
    return loss


def mse_loss(true,estimate):
    loss=(true - estimate) ** 2
    return loss


def getLoss_(agent, x_traj,obs_traj, a_traj, theta, env, gains_range, std_range, PI_STD, NUM_SAMPLES):
    ''
    logPr = torch.zeros(1) #torch.FloatTensor([])
    logPr_act = torch.zeros(1)
    logPr_obs = torch.zeros(1)

    pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = torch.split(theta.view(-1), 2)

    for num_it in range(NUM_SAMPLES): # what is the numsample/particle here? default is 50

        for ep, x_traj_ep in enumerate(x_traj): # repeat for 501 episodes
            a_traj_ep = a_traj[ep]
            obs_traj_ep=obs_traj[ep]
            logPr_ep = torch.zeros(1)
            logPr_act_ep = torch.zeros(1)
            logPr_obs_ep = torch.zeros(1)

            t = torch.zeros(1)
            x = x_traj_ep[0].view(-1)

            env.pro_gains = pro_gains
            env.pro_noise_stds = pro_noise_stds
            env.goal_radius = goal_radius
            env.obs_gains=obs_gains
            env.obs_noise_stds=obs_noise_stds
            env.reset()  # reset monkey's internal model

            env.x=x # assign the same x
            env.b=x,env.P # assign same b
            env.belief = env.Breshape(b=env.b, time=t, theta=theta)

            for it, next_x in enumerate(x_traj_ep[1:]): # repeat for steps in episode
                action = agent(env.belief)[0] # simulated acton

                next_ox = env.observations_mean(next_x) # multiplied by observation gain, no noise
                # next_ox_ = env.observations(next_x)  # simulated observation (with noise)

                action_loss =5*torch.ones(2)+np.log(np.sqrt(2* pi)*PI_STD) + (action - a_traj_ep[it] ) ** 2 / 2 /(PI_STD**2)
                # obs_loss = 5*torch.ones(2)+torch.log(np.sqrt(2* pi)*obs_noise_stds) +(next_ox_ - next_ox).view(-1) ** 2/2/(obs_noise_stds**2)

                logPr_act_ep = logPr_act_ep + action_loss.sum()
                logPr_obs_ep = logPr_obs_ep #+ obs_loss.sum()
                logPr_ep = logPr_ep + logPr_act_ep #+ logPr_obs_ep

                next_b, info = env.belief_step(env.b, next_ox, a_traj_ep[it], env.box)  # no change to internal var
                env.b=next_b
                next_state = env.Breshape(b=next_b, time=t, theta=theta)
                env.belief=next_state
                t += 1

            logPr_act += logPr_act_ep
            logPr_obs += logPr_obs_ep

            logPr += logPr_ep
            #logPr = torch.cat([logPr, logPr_ep])

    return logPr, logPr_act, logPr_obs #logPr.sum()


def init_theta(phi,arg,purt=None):
    '''
    create a initial theta by adding random to phi
    '''
    rndsgn = torch.sign(torch.randn(1,len(phi))).view(-1)
    if purt is None:
        purt= torch.Tensor([0.5,0.5,0.1,0.1,0.5,0.5,0.1,0.1,0.1])
    theta = phi.data.clone()+rndsgn*purt
    theta = theta_range(theta, arg.gains_range, arg.std_range, arg.goal_radius_range)  # keep inside of trained range
    return theta


def unpack_theta(theta):
    'unpack the 1x9 tensor theta into p gain/noise, obs gain/noise, r'
    pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius = torch.split(theta.view(-1), 2)
    return pro_gains, pro_noise_stds, obs_gains, obs_noise_stds, goal_radius


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


def load_inverse_data(filename):
  'load the data pickle file, return the data dict'
  sys.path.insert(0, './inverse_data/')
  if filename[-4:]=='.pkl':
      data=torch.load('inverse_data/{}'.format(filename))
  else:
      data=torch.load('inverse_data/{}.pkl'.format(filename))
  data['filename']='inverse_data/{}.pkl'.format(filename)
  return data

def save_inverse_data(inverse_data):
    'save the loaded data dict as pkl'
    filename=inverse_data['filename']
    torch.save(inverse_data, filename)



# def theta_init(agent, env, arg):
#     '''
#     return (true theta, true loss, x trajectory, a trajectory)
#     '''

#     # true theta
#     true_theta = reset_theta(arg.gains_range, arg.std_range, arg.goal_radius_range)
#     #true_theta_log.append(true_theta.data.clone())
#     x_traj, _, a_traj, _ = trajectory(agent, true_theta, env, arg, arg.gains_range, arg.std_range,
#                                       arg.goal_radius_range, arg.NUM_EP)  # generate true trajectory
#     true_loss, _, _ = getLoss(agent, x_traj, a_traj, true_theta, env, arg.gains_range, arg.std_range, arg.PI_STD,
#                         arg.NUM_SAMPLES)  # this is the lower bound of loss?

#     init_result = {'true_theta_log': true_theta,
#                    'true_loss_log': true_loss,
#                    'x_traj_log': x_traj,
#                    'a_traj_log': a_traj
#                    }
#     return init_result



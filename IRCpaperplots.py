
from plot_ult import *
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

arg.presist_phi=True
arg.agent_knows_phi=False
arg.goal_distance_range=[0.1,1]
arg.gains_range =[0.05,1.5,pi/4,pi/1]
arg.goal_radius_range=[0.001,0.3]
arg.std_range = [0.08,1,0.08,1]
arg.mag_action_cost_range= [0.0001,0.001]
arg.dev_action_cost_range= [0.0001,0.005]
arg.dev_v_cost_range= [0.1,0.5]
arg.dev_w_cost_range= [0.1,0.5]
arg.TERMINAL_VEL = 0.1
arg.DELTA_T=0.1
arg.EPISODE_LEN=100
arg.agent_knows_phi=False
DISCOUNT_FACTOR = 0.99
arg.sample=100
arg.batch = 70
# arg.NUM_SAMPLES=1
# arg.NUM_EP=1
arg.NUM_IT = 1 
arg.NUM_thetas = 1
arg.ADAM_LR = 0.0002
arg.LR_STEP = 20
arg.LR_STOP = 0.5
arg.lr_gamma = 0.95
arg.PI_STD=1
arg.presist_phi=False
arg.cost_scale=1

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



if __name__=='__main__':
    env=ffacc_real.FireFlyPaper(arg)
    env.debug=True


    agent_=TD3_torch.TD3.load('trained_agent/re1re1repaper_3_199_198.zip')
    agent=agent_.actor.mu.cpu()
    diagnose_plot_theta(agent, env, phi, theta_init, theta_final,5)



    # load monkey data
    print('loading data')
    note='bnorm4rand'
    with open("C:/Users/24455/Desktop/bruno_normal_downsample",'rb') as f:
            df = pickle.load(f)
    df=datawash(df)
    df=df[df.category=='normal']
    # df=df[df.target_r>250]
    df=df[df.floor_density==0.005]
    # floor density are in [0.0001, 0.0005, 0.001, 0.005]
    df=df[:-100]
    print('process data')
    states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
    print('done process data')
    sample_states, sample_actions, sample_tasks=sample_batch(states=states,actions=actions,tasks=tasks,batch_size=20)
    loss_function=monkey_loss_wrapped(env=env, 
            agent=agent, 
            phi=phi, 
            num_episodes=5,
            action_var=0.001,
            states=sample_states,
            actions=sample_actions,
            tasks=sample_tasks,)



    # load inverse data
    inverse_data=load_inverse_data("testdcont2_16_45")
    theta_trajectory=inverse_data['theta_estimations']
    theta_estimation=theta_trajectory[-1]
    theta=torch.tensor(theta_estimation)
    phi=torch.tensor(inverse_data['phi'])
    H=inverse_data['Hessian'] 
    print(H)
    plot_inverse_trend(inverse_data)
    colorgrad_inverse_traj(inverse_data)


    # vis the inverses of differnt density
    inverse_data=load_inverse_data("testdefstart2_12_43")
    plot_inverse_trend(inverse_data)
    inverse_data=load_inverse_data("test2_12_40")
    plot_inverse_trend(inverse_data)

    inversedensityfiles=[
        'varfixnorm12_0_51',
        'varfixnorm22_0_51',
        'varfixnorm32_0_52',
        'varfixnorm42_0_52',
    ]

    noiseparam={}
    for eachdensity,inversefile in zip([0.0001, 0.0005, 0.001, 0.005],inversedensityfiles):
        noiseparam[eachdensity]={}
        noiseparam[eachdensity]['param']=load_inverse_data(inversefile)['theta_estimations'][-2][2:6]
        H=load_inverse_data(inversefile)['Hessian']
        if H!=[]:
            noiseparam[eachdensity]['std']=stderr(torch.inverse(H))[2:6]
    obsvsdensity(noiseparam)


    with open("Z:\\bruno_normal/3dens_packed_bruno_normal",'rb') as f:
        df = pickle.load(f)
        df=datawash(df)
        df=df[df.category=='normal']
        for eachdensity,inversefile in zip([0.0001, 0.0005, 0.001, 0.005],inversedensityfiles):
            tempdf=df[df.floor_density==eachdensity]
            states, actions, tasks=monkey_data_downsampled(tempdf,factor=0.0025)
            sample_states, sample_actions, sample_tasks=sample_batch(states=states,actions=actions,tasks=tasks,batch_size=20)
            theta=torch.tensor(load_inverse_data(inversefile)['theta_estimations'][-2])
            _loss_function=monkey_loss_wrapped(env=env, 
                agent=agent, 
                phi=phi, 
                num_episodes=5,
                action_var=0.001,
                states=sample_states,
                actions=sample_actions,
                tasks=sample_tasks,)
            H=torch.autograd.functional.hessian(_loss_function,theta,strict=True)
            H=H[:,0,:,0]   
            noiseparam[eachdensity]={}  
            noiseparam[eachdensity]['std']=stderr(torch.inverse(H))[2:6]   
            inverse_data=load_inverse_data(inversefile)
            inverse_data['Hessian']=H    
            save_inverse_data(inverse_data)

    for inversefile in  inversedensityfiles:
        theta=torch.tensor(load_inverse_data(inversefile)['theta_estimations'][-2])
        H=inverse_data['Hessian']
        print(stderr(torch.inverse(H)))


    # count reward % by density
    for eachdensity in [0.0001, 0.0005, 0.001, 0.005]:  
        print('density {}, reward% '.format(eachdensity),len(df[df.floor_density==eachdensity][df.rewarded])/len(df[df.floor_density==eachdensity]))
    # count reward by pert
    print('all trial {}, reward% '.format(len(df[df.rewarded])/len(df)))
    print('pert trial {}, reward% '.format(len(df[~df.perturb_start_time.isnull() & df.rewarded])/len(df[~df.perturb_start_time.isnull()])))
    print('non pert trial {}, reward% '.format(len(df[df.perturb_start_time.isnull() & df.rewarded])/len(df[df.perturb_start_time.isnull()])))


    # calculate H

    H=torch.autograd.functional.hessian(loss_function,theta,strict=True)
    H=H[:,0,:,0]
    stderr(torch.inverse(H))          
    inverse_data['Hessian']=H
    save_inverse_data(inverse_data)
    ev, evector=torch.eig(H,eigenvectors=True)
    env=ffacc_real.FireFlyPaper(arg)

    # background in eigen axis
    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        sample_states, sample_actions, sample_tasks=sample_batch(states=states,actions=actions,tasks=tasks,batch_size=20)
        bk=plot_background_ev(
        ax, 
        evector,
        loss_function=monkey_loss_wrapped(env=env, 
            agent=agent, 
            phi=phi, 
            num_episodes=50,
            action_var=1e-2,
            states=sample_states,
            actions=sample_actions,
            tasks=sample_tasks,), 
        number_pixels=5, 
        alpha=0.5,
        scale=0.01,
        background_data=None)
        
    # loss vs first eigen
    scale=0.1
    ev1=evector[0].view(-1,1)
    X=np.linspace(theta-ev1*0.5*scale,theta+ev1*0.5*scale,number_pixels)
    logll1d=[]
    for onetheta in X:
        onetheta=torch.tensor(onetheta)
        with torch.no_grad():
            logll1d.append(loss_function(onetheta))
    plt.plot(logll1d)

    # inverse_data['grad']

    # load agent
    agent_=TD3_torch.TD3.load('trained_agent/paper.zip')
    agent=agent_.actor.mu.cpu()
    policygiventheta()
    policygiventhetav2(env.decision_info.shape[-1])
    agentvsmk_skip(55)
    plot_critic_polar()
    ind=torch.randint(low=100,high=5000,size=(1,))
    input={
        'agent':agent,
        'theta':theta,
        'phi':phi,
        'env': env,
        'num_trials':10,
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
    single_trial_overhead()

    ind=torch.randint(low=100,high=111,size=(1,))
    indls=similar_trials(ind, tasks, actions)
    indls=indls[:20]
    input={
        'agent':agent,
        'theta':theta,
        'phi':phi,
        'env': env,
        'num_trials':len(indls),
        # 'task':tasks[:20],
        'mkdata':{
                        'trial_index': indls,               
                        'task': tasks,                  
                        'actions': actions,                 
                        'states':states,
                        },                      
        'use_mk_data':True
    }
    with suppress():
        res=trial_data(input)
    plotoverhead(res)
    plotctrl(res)


    number_pixels=5
    background_data=inverse_trajectory_monkey(
                        theta_trajectory,
                        env=env,
                        agent=agent,
                        phi=phi,
                        background_data=None,
                        H=None,
                        background_contour=True,
                        number_pixels=number_pixels,
                        background_look='contour',
                        ax=None,
                        action_var=0.2,
                        loss_sample_size=5,
                        num_episodes=13)
    inverse_data['background_data']={number_pixels:background_data}
    save_inverse_data(inverse_data)


    c=plt.imshow(background_data)
    plt.colorbar(c)

    # 1. warpped up of agents with diff theta in a row
    phi=torch.tensor([[0.5000],
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
    theta_init=torch.tensor([[0.5000],
            [1.57],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
            [0.13],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
    ])
    theta_final=torch.tensor([[0.5000],
            [1.57],
            [0.9],
            [0.9],
            [0.1],
            [0.1],
            [0.13],
            [0.1],
            [0.1],
            [0.1],
            [0.1],
    ])
    env.debug=True
    diagnose_plot_theta(agent, env, phi, theta_init, theta_final,5,etask=tasks[ind],initv=actions[ind][0][0],initw=actions[ind][0][1],mkactions=actions[ind])






    # 1. showing converge
    # 1.1 theta uncertainty from heissian shows convergence
    # 1.2 log likelihood surface in pc space, with our path and final uncertainty

    #---------------------------------------------------------------------
    # inversed theta with error bar
    # inversed_theta_bar(inverse_data)

    #---------------------------------------------------------------------
    # inversed theta uncertainty by std/mean
    # inversed_uncertainty_bar(inverse_data)
    #---------------------------------------------------------------------


    # 1.4 differnt inverse rans and final thetas in pc space, each with uncertainty

    # 2. reconstruction belief from inferred params



    # 2.1 overhead view, monkey's belief and state in one trial
    ind=torch.randint(low=100,high=5000,size=(1,))
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
    single_trial_overhead()


    #---------------------------------------------------------------------
    # plot one monkey trial using index
    ind=torch.randint(low=100,high=300,size=(1,))
    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.plot(states[ind][:,0],states[ind][:,1], color='r',alpha=0.5)
        goalcircle = plt.Circle([tasks[ind][0],tasks[ind][1]], 0.13, color='y', alpha=0.5)
        ax.add_patch(goalcircle)
        ax.set_xlim(0,1)
        ax.set_ylim(-0.6,0.6)



    #---------------------------------------------------------------------
    # 3. monkey and inferred agent act similar on similar trials

    # IRC on its own
    ind=torch.randint(low=0,high=len(tasks),size=(1,))
    indls=similar_trials(ind, tasks, actions)
    indls=indls[:10]
    with torch.no_grad():
        inputirc={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':10,
            'task':[tasks[i] for i in indls],
            'mkdata':{
                            'trial_index': indls,               
                            'task': tasks,                 
                            'actions': actions,                 
                            'states':states,},
            'use_mk_data':False
        }
        with suppress():
            resirc=trial_data(inputirc)

        inputmk={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':10,
            'mkdata':{
                            'trial_index': indls,               
                            'task': tasks,                 
                            'actions': actions,                 
                            'states':states,
                            },                      
            'use_mk_data':True
        }
        with suppress():
            resmk=trial_data(inputmk)
    # ax=plotoverhead(resirc, color='orange')
    # ax=plotoverhead(resmk, color='tab:blue')
    # plotoverhead_mk(indls,ax=ax)
    # ax.get_figure()
    # ax=plotctrl(resmk, prefix='monkey')
    ax=plotctrl_mk(indls,prefix='monkey')
    plotctrl(resirc,ax=ax, color=['blue','orangered'],prefix='IRC')
    # plotctrl(resmk,ax=ax, color=['blue','orangered'],prefix='IRCmk')
    ax.get_figure()

    plotoverhead(resirc, color='orange')
    plotoverhead(resmk, color='tab:blue')
    plotctrl(resirc, prefix='IRC')
    plotctrl(resmk, color=['blue','orangered'],prefix='monkey')


    # given mk states
    ind=torch.randint(low=0,high=len(tasks),size=(1,))
    indls=similar_trials(ind, tasks, actions)
    with torch.no_grad():
        inputirc={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':10,
            'task':[tasks[i] for i in indls],
            'mkdata':{
                            'trial_index': indls,               
                            'task': tasks,                 
                            'actions': actions,                 
                            'states':states,
                            },                      
            'use_mk_data':False}
        with suppress():
            resirc=trial_data(inputirc)
        # plotoverhead(resirc)
        inputmk={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':10,
            'mkdata':{
                            'trial_index': indls,               
                            'task': tasks,                 
                            'actions': actions,                 
                            'states':states,
                            },                      
            'use_mk_data':True
        }
        with suppress():
            resmk=trial_data(inputmk)
        # plotoverhead(resmk)
        # ax=plotctrl(resmk, color=['blue','orangered'],prefix='monkey')
    ax=plotctrl_mk(indls,prefix='monkey')
    plotctrl(resirc,ax=ax, color=['blue','orangered'],prefix='IRC')
    ax.get_figure()


    # IRC on its own in pert task
    ind=np.random.randint(low=0,high=len(df))
    print(ind)
    pert=np.array([df.iloc[ind].perturb_v/400,df.iloc[ind].perturb_w/400])
    pert=np.array(down_sampling_(pert.T))
    pert=pert.astype('float32')
    with torch.no_grad():
        inputirc={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':1,
            'task':[tasks[ind]],
            'mkdata':{
                            'trial_index': [ind],               
                            'task': tasks,                 
                            'actions': actions,                 
                            'states':states,
                            'pert':pert
                            },
            'use_mk_data':False
        }
        with suppress():
            resirc=trial_data(inputirc)

        inputmk={
            'agent':agent,
            'theta':theta,
            'phi':phi,
            'env': env,
            'num_trials':1,
            'task':[tasks[ind]],
            'mkdata':{
                            'trial_index': [ind],               
                            'task': tasks,                 
                            'actions': actions,                 
                            'states':states,
                            'pert':pert
                            },
            'use_mk_data':True
        }
        with suppress():
            resmk=trial_data(inputmk)
    
    plt.plot(pert)

    ax=plotctrl_mk([ind],prefix='monkey')
    plotctrl(resirc,ax=ax, color=['blue','orangered'],prefix='IRC')
    # plotctrl(resmk,ax=ax, color=['blue','orangered'],prefix='IRCmk')
    ax.get_figure()


pert
np.array(actions[ind])






    #---------------------------------------------------------------------
    alllogll=[]
    for true, est in zip(resmk['mk_actions'],resmk['agent_actions']):
        eplogll=[0.]
        for t,e in zip(true,est):
            eplogll.append(torch.sum(logll(t,e,std=0.01)))
        alllogll.append(sum(eplogll))
    print(np.mean(alllogll))

    for a in resmk['mk_actions']:
        plt.plot(a[:,0],c='tab:blue',alpha=0.5)
    for a in resmk['agent_actions']:
        plt.plot(a[:,0],c='tab:orange',alpha=0.5)

    for a in resmk['mk_actions']:
        plt.plot(a[:,1],c='royalblue',alpha=0.5)
    for a in resmk['agent_actions']:
        plt.plot(a[:,1],c='orangered',alpha=0.5)
    plt.xlabel('time [dt]')
    plt.ylabel('control v and w')


    plt.plot(actions[indls[-1]])
    plt.plot(resmk['agent_actions'][-1])

    true=actions[indls[-1]]
    est=resmk['agent_actions'][-1]

    sumlogll=0.
    for t,e in zip(true,est):
        sumlogll+=logll(t,e,std=0.001)


    # use irc action
    ind=torch.randint(low=100,high=1111,size=(1,))
    indls=similar_trials(ind, tasks, actions)
    indls=indls[:20]
    maxlen=max([len(actions[i]) for i in indls])
    for ind in indls:
        env.reset(goal_position=tasks[ind],theta=theta,phi=phi)
        vw=[]
        done=False
        with torch.no_grad():
            while not done and env.trial_timer<=maxlen:
                action=agent(env.decision_info)
                vw.append(action)
                _,_,done,_=env.step(action)
        vw=torch.stack(vw)[:,0,:]

        plt.plot(vw,c='orange',alpha=0.5)
        plt.plot(actions[ind],c='tab:blue',alpha=0.5)
    plt.plot(vw,c='orange',alpha=1,label='agent')
    plt.plot(actions[ind],c='tab:blue',alpha=1,label='monkey')
    plt.xlabel('time [dt]')
    plt.ylabel('control v and w')
    plt.legend()


    # use mkaction, same as in inverse
    ind=torch.randint(low=100,high=1111,size=(1,))
    indls=similar_trials(ind, tasks, actions)
    indls=indls[:20]
    for ind in indls:
        env.reset(goal_position=tasks[ind],theta=theta,phi=phi)
        vw=[]
        done=False
        mkactionep=actions[ind][1:]
        with torch.no_grad():
            while not done and env.trial_timer<len(mkactionep):
                action=agent(env.decision_info)
                vw.append(action)
                _,_,done,_=env.step(mkactionep[int(env.trial_timer)])
        vw=torch.stack(vw)[:,0,:]

        plt.plot(vw,c='orange',alpha=0.5)
        plt.plot(actions[ind],c='tab:blue',alpha=0.5)
    plt.plot(vw,c='orange',alpha=1,label='agent')
    plt.plot(actions[ind],c='tab:blue',alpha=1,label='monkey')
    plt.xlabel('time [dt]')
    plt.ylabel('control v and w')
    plt.legend()







    # 3.2 overhead view, similar trials, stopping positions distribution similar

    # 3.3 overhead view, skipped trials boundary similar

    # 3.4 control curves, similar trials, similar randomness

    # 3.5 group similar trials, uncertainty grow along path, vs another theta

    # 4. validation in other (perturbation) trials
    # 4.1 overhead path of agent vs monkey in a perturbation trial, path similar


    #---------------------------------------------------------------
    # value analysis, bar plot, avg trial, skip trial

    critic=agent_.critic.qf0.cpu()
    skipped_task=[0.2,0.9]
    avg_task=[0.,0.7]
    atgoal_task=[0,0]
    with torch.no_grad():
        base=env.decision_info.flatten().clone().detach()
        skipedge=base.clone().detach();skipedge[0:2]=torch.tensor(xy2pol(skipped_task))
        avg=base.clone().detach();avg[0:2]=torch.tensor(xy2pol(avg_task))
        atgoal=base.clone().detach();atgoal[0:2]=torch.tensor(xy2pol(atgoal_task))
        avgvalue=critic(torch.cat([avg,agent(avg).flatten()]))
        skipedgevalues=critic(torch.cat([skipedge,agent(skipedge).flatten()]))
        goalv=critic(torch.cat([atgoal,agent(atgoal).flatten()]))

    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        X=['skipped','avg','goal']
        Y=[skipedgevalues,avgvalue,goalv]
        ax.bar(X,Y)


    #  value analysis, critic heatmap
    plot_critic_polar()


    #---------------------------------------------------------------
    # intial uncertainty x y
    with initiate_plot(2, 2, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot()
        cov=theta_cov(inverse_data['Hessian'])
        theta_stds=stderr(cov)[-2:]
        x_pos = [0,1]
        data=inverse_data['theta_estimations'][-1][-2:]
        data=[d[0] for d in data]
        ax.bar(x_pos, data, yerr=theta_stds, width=0.5, color = 'tab:blue')
        ax.set_xticks(x_pos)
        ax.set_ylabel('inferred parameter')
        ax.set_xticklabels(theta_names[-2:],rotation=45,ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)





    # Heissian polots-----------------------------------------------------------
    H=compute_H_monkey(env, 
                        agent, 
                        theta, 
                        phi, 
                        H_dim=11, 
                        num_episodes=9,
                        num_samples=9,
                        monkeydata=(states, actions, tasks))           
    inverse_data['Hessian']=H
    save_inverse_data(inverse_data)

    # matter_most_eigen_direction()

    stderr(torch.inverse(H))

    p=torch.round(torch.log(torch.sign(H)*H))*torch.sign(H)

    inds=[1,3,5,8,0,2,4,7,6,9,10]
    # covariance
    with initiate_plot(5,5,300) as fig:
        ax=fig.add_subplot(1,1,1)
        cov=theta_cov(H)
        cov=torch.tensor(cov)
        im=plt.imshow(cov[inds].t()[inds].t(),cmap=plt.get_cmap('bwr'),
            vmin=-torch.max(cov),vmax=torch.max(cov))
        ax.set_title('covariance matrix', fontsize=20)
        x_pos = np.arange(len(theta_names))
        plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
        plt.xticks(x_pos, [theta_names[i] for i in inds],rotation=45,ha='right')
        add_colorbar(im)


    # correlation matrix
    b=torch.diag(torch.tensor(cov),0)
    S=torch.diag(torch.sqrt(b))
    Sinv=torch.inverse(S)
    correlation=Sinv@cov@Sinv
    with initiate_plot(5,5,300) as fig:
        ax=fig.add_subplot(1,1,1)
        im=ax.imshow(correlation[inds].t()[inds].t(),cmap=plt.get_cmap('bwr'),
            vmin=-torch.max(correlation),vmax=torch.max(correlation))
        ax.set_title('correlation matrix', fontsize=20)
        x_pos = np.arange(len(theta_names))
        plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')
        plt.xticks(x_pos, [theta_names[i] for i in inds],rotation=45,ha='right')
        add_colorbar(im)



    # eigen vector
    ev, evector=torch.eig(H,eigenvectors=True)
    ev=ev[:,0]
    ev,esortinds=ev.sort(descending=False)
    evector=evector[esortinds]
    with initiate_plot(5,5,300) as fig:
        ax=fig.add_subplot(1,1,1)
        img=ax.imshow(evector[:,inds].t(),cmap=plt.get_cmap('bwr'),
                vmin=-torch.max(evector),vmax=torch.max(evector))
        add_colorbar(img)
        ax.set_title('eigen vectors of Hessian')
        x_pos = np.arange(len(theta_names))
        plt.yticks(x_pos, [theta_names[i] for i in inds],ha='right')


    with initiate_plot(5,1,300) as fig:
        ax=fig.add_subplot(1,1,1)
        x_pos = np.arange(len(theta_names))
        # Create bars and choose color
        ax.bar(x_pos, ev, color = 'blue')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        ax.set_xticks([])
        ax.set_yscale('log')
        # ax.set_ylim(min(plotdata),max(plotdata))
        ax.set_yticks([0.1,100,1e4])
        ax.set_xlabel('eigen values, log scale')
        plt.tick_params(axis='y', which='minor')
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
        plt.gca().invert_yaxis()



    # pca background

    env=ffacc_real.FireFlyPaper(arg)
    number_pixels=5
    background_data=inverse_trajectory_monkey(
                        theta_trajectory,
                        env=env,
                        agent=agent,
                        phi=phi,
                        background_data=None,
                        H=None,
                        background_contour=True,
                        number_pixels=number_pixels,
                        background_look='contour',
                        ax=None,
                        action_var=0.001,
                        loss_sample_size=20)
    inverse_data['background_data']={number_pixels:background_data}
    save_inverse_data(inverse_data)


    # testing gaussian int
    g=lambda x,var: 1/torch.sqrt(2*pi*torch.ones(1))*torch.exp(-0.5*x**2/torch.ones(1)*var)
    gi=lambda x,var: -(torch.erf(x/torch.sqrt(2*torch.ones(1))/var)-1)/2
    X=np.linspace(-5,5,100)
    Y=[g(x,1) for x in X]
    Z=[gi(x,1) for x in X]
    # Z=[torch.log(torch.ones(1)*x) for x in X]
    # z=1/g(0,1)
    # Z=[z*y for y in Y]
    plt.figure(figsize=(6, 6), dpi=80)
    plt.plot(X,Y)
    plt.plot(X,Z)
    Z[49]


    # background of eigen axis
    env=ffacc_real.FireFlyPaper(arg)
    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        sample_states, sample_actions, sample_tasks=sample_batch(states=states,actions=actions,tasks=tasks,batch_size=5)
        bk=plot_background_ev(
        ax, 
        evector,
        loss_function=monkey_loss_wrapped(env=env, 
            agent=agent, 
            phi=phi, 
            num_episodes=5,
            states=sample_states,
            actions=sample_actions,
            tasks=sample_tasks,), 
        number_pixels=5, 
        alpha=0.5,
        scale=0.2,
        background_data=None)
    plt.imshow(bk)


    #----------------------------------------------------------------
    # function to get value of a given belief
    def getvalues(
        na=11,
        nd=9,
        env=None,
        agent=None,
        belief=None,
        theta=None,
        phi=None,
        iti=5,
        task=None,
        **kwargs
        ):
        # samples tasks
        distances,angles=sample_all_tasks(na=na,nd=nd)
        values=distances*0
        for ai in range(values.shape[0]):
            for di in range(values.shape[1]):
                # collect rollout
                input={
                'agent':agent,
                'theta':phi,
                'phi':phi,
                'env': env,
                'task':torch.tensor(pol2xy(angles[ai,di],distances[ai,di],)).view(1,-1).float(),
                'num_trials':1,
                'mkdata':{},                      
                'use_mk_data':False
                }
                with suppress():
                    res=trial_data(input)
                # calculate anticipated reward probobility
                finalcov=res['agent_covs'][0][-1][:2,:2]
                # finalmu=res['agent_beliefs'][0][-1][:2]
                # goal=torch.tensor(res['task'][0])
                # delta=finalmu-goal
                delta=mu_zero
                R = torch.eye(2)*env.goal_r_hat**2 
                S = R+finalcov
                alpha = -0.5 * delta @ S.inverse() @ delta.t()
                rewardprob = torch.exp(alpha) /2 / pi /torch.sqrt(S.det())
                # normalization
                mu_zero = torch.zeros(1,2)
                alpha_zero = -0.5 * mu_zero @ R.inverse() @ mu_zero.t()
                reward_zero = torch.exp(alpha_zero) /2 / pi /torch.sqrt(R.det())
                # rewardprob = rewardprob/reward_zero
                # calculate value
                # env.trial_timer if env.rewarded() else 60
                value=rewardprob*env.reward/(env.trial_timer+iti)
                values[ai,di]=value #if env.rewarded() else None
                plt.imshow(values)
        return values


    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        theta,rad = np.meshgrid(np.linspace(-0.75,0.75,nd), np.linspace(0.2,1,na)) 
        ax = fig.add_subplot(111,polar='True')
        ax.pcolormesh(theta, rad, values) #X,Y & data2D must all be same dimensions
        ax.set_theta_offset(pi/2)
        ax.grid(False)
        ax.set_thetamin(-43)
        ax.set_thetamax(43)




    norm = np.linalg.norm(values)

    normal_array = values/norm

    np.max(normal_array)


    # density heatmap testing-------------------------------
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy.ndimage.filters import gaussian_filter
    def myplot(x, y, s, bins=1000):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        heatmap = gaussian_filter(heatmap, sigma=s)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        return heatmap.T, extent
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    fig, ax = plt.subplots()
    # img, extent = myplot(x, y, 32)
    # ax.plot(x, y, 'k.', markersize=5)
    # ax.set_title("Scatter plot")
    img, extent = myplot(skipx, skipy, 22)
    # ax.imshow(img, extent=extent, origin='lower', cmap=cm.Greys)
    ax.contourf(img, extent=extent, origin='lower', cmap=cm.Greys)
    ax.set_title("Smoothing with  $\sigma$ = %d" % 32)
    plt.show()




    from scipy.stats import gaussian_kde as kde
    from matplotlib.colors import Normalize
    def makeColours( vals ):
        colours = np.zeros( (len(vals),3) )
        norm = Normalize( vmin=vals.min(), vmax=vals.max() )
        #Can put any colormap you like here.
        colours = [cm.ScalarMappable( norm=norm, cmap='jet').to_rgba( val ) for val in vals]
        return colours


    densObj = kde(np.array([skipx,skipy]))
    colours = makeColours( densObj.evaluate( np.array([skipx,skipy]) ) )
    len(colours)
    samples = np.random.multivariate_normal([0,0],env.P[:2,:2],200).T
    samples.shape
    np.array([skipx,skipy])
    plt.scatter( skipx, skipy, color=colours )




    overhead_skip_density(res1)



    # skipping analysis
    '''
        we have a value map and skipping data.
        we want to fit the data to a value curve.
        define,
            v(a,r)=value
            t=threshold of skip
        Loss=L(t(v(a,r)), true skip or not), minimize loss to solve for t.
    '''
    # first, give xy, return value.
    critic=agent_.critic.qf0.cpu()
    nbin=100
    dmin=0.2
    dmax=0.8
    maxangle=0.75
    with torch.no_grad():
        env.reset(phi=phi,theta=theta)
        base=env.decision_info.flatten().clone().detach()
        start=base.clone().detach();start[0]=dmin; start[1]=-maxangle
        xend=start.clone().detach();yend=start.clone().detach()
        xend[0]=dmax;yend[1]=maxangle
        xdelta=(xend-start)/nbin
        ydelta=(yend-start)/nbin
        values=[]
        for xi in range(nbin):
            yvalues=[]
            for yi in range(nbin):
                thestate=start+xi*xdelta+yi*ydelta
                value=critic(torch.cat([thestate,agent(thestate).flatten()]))
                yvalues.append(value.item())
            values.append(yvalues)
        values=np.asarray(values)
        values=normalizematrix(values)
        # values is a distance by angle matrix

    tasksgridd=np.linspace(dmin,dmax,nbin)
    tasksgrida=np.linspace(-maxangle,maxangle,nbin)





    newtasks=[]
    for task in tasks:
        newtasks.append(task[0])
    newtasks=np.array(newtasks)

    plt.scatter(newtasks[:,1],newtasks[:,0])




    taskvalues=[]
    taskskips=[]
    taskds=[]
    for task,state in zip(newtasks,states):
        d,a=xy2pol(task.tolist(),rotation=False)
        di=roughbisec(tasksgridd,d)
        ai=roughbisec(tasksgrida,a)
        taskvalues.append(values[di,ai])
        taskskips.append(is_skip(task,state,dthreshold=0.4))
        taskds.append(d)

    taskvalues=np.array(taskvalues).reshape(-1, 1)
    taskskips=[int(a) for a in taskskips]
    taskds=np.array(taskds).reshape(-1, 1)

    skipvalues=[]
    nonskipvalues=[]
    for v,s in zip(taskvalues,taskskips):
        if s:
            skipvalues.append(v)
        else:
            nonskipvalues.append(v)
    skipvalues=np.array(skipvalues)
    nonskipvalues=np.array(nonskipvalues)

    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.hist(nonskipvalues,alpha=0.5,bins=20,label='non-skipped')
        ax.hist(skipvalues,alpha=0.5,bins=20,label='skipped')
        ax.set_xlabel('value')
        ax.set_ylabel('number of trials')
        ax.legend()

    skipds=[]
    nonskipds=[]
    for v,s in zip(taskds,taskskips):
        if s:
            skipds.append(v)
        else:
            nonskipds.append(v)
    skipds=np.array(skipds)
    nonskipds=np.array(nonskipds)

    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        ax.hist(nonskipds,alpha=0.5,bins=20,label='non-skipped')
        ax.hist(skipds,alpha=0.5,bins=20,label='skipped')
        ax.set_xlabel('distance')
        ax.set_ylabel('number of trials')
        ax.legend()

    from sklearn.linear_model import LogisticRegression
    logisticRegr = LogisticRegression()
    logisticRegr.fit(taskds, taskskips)

    # predictedskip=logisticRegr.predict(newtasks)
    # count=0
    # for ypred,y in zip(predictedskip,taskskips):
    #     if not (ypred^y):
    #         count+=1
    # print(count/len(taskskips))

    from scipy.special import expit
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.scatter(taskds, taskskips, color="black", zorder=20)
    X_test = np.linspace(-5, 10, 300)
    loss = expit(taskds * logisticRegr.coef_ + logisticRegr.intercept_).ravel()
    plt.plot(taskds, loss, color="red", linewidth=3)


    xmin, xmax = -5, 5
    n_samples = 100
    np.random.seed(0)
    X = np.random.normal(size=n_samples)
    y = (X > 0).astype(float)
    X[X > 0] *= 4
    X += 0.3 * np.random.normal(size=n_samples)

    X = X[:, np.newaxis]
    X.shape
    y.shape
    # Fit the classifier
    clf = linear_model.LogisticRegression(C=1e5)
    clf.fit(X, y)

    # and plot the result
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.scatter(X.ravel(), y, color="black", zorder=20)
    X_test = np.linspace(-5, 10, 300)

    loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
    plt.plot(X_test, loss, color="red", linewidth=3)

    plt.ylabel("y")
    plt.xlabel("X")
    plt.xticks(range(-5, 10))
    plt.yticks([0, 0.5, 1])
    plt.ylim(-0.25, 1.25)
    plt.xlim(-4, 10)
    plt.legend(
        ("Logistic Regression Model", "Linear Regression Model"),
        loc="lower right",
        fontsize="small",
    )
    plt.tight_layout()
    plt.show()




    newtaskspolar=[]
    for task in tasks:
        newtaskspolar.append(xy2pol(task[0],rotation=False))
    newtaskspolar=np.array(newtaskspolar)
    plt.scatter(newtaskspolar[:,1],newtaskspolar[:,0])


    with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = fig.add_subplot(111)
        axrange=[-maxangle, maxangle, dmin , dmax]
        c=ax.imshow(values, origin='lower',extent=axrange, aspect='auto')
        add_colorbar(c)
        ax.set_ylabel('distance')
        ax.set_xlabel('angle')



# perturbation example



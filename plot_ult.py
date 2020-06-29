import numpy as np
from numpy import pi
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1




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



from stable_baselines import DDPG
from stable_baselines import TD3
import policy_torch
import numpy as np
from numpy import pi
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits import axes_grid1
# load baseline model
selu_name='DDPG_LQC_selu_finalreward1000000_0 27 21 14'
baselines_mlp_model = DDPG.load(selu_name)
torch_model_selu = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[256,256,64,32],act_fn=nn.functional.selu)
torch_model_selu.name='LQC'

# baselines_mlp_model = DDPG.load("DDPG_LQC_relu1000000_7 27 13 15")
# torch_model_selu = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[32,64])
# torch_model_selu.name='LQC'


# testing, get sample beliefs
from Config import Config
from FireflyEnv import ffenv
arg=Config()
arg.std_range=[0.0001,0.001,0.0001,0.001]
arg.gains_range=[0.99,1.,0.99,1.]
env=ffenv.FireflyEnv(arg)

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def get_beliefs(number_trials):
    beliefs=[]
    for i in range(number_trials):
        belief=env.reset()[0]
        while not env.stop:
            action=torch_model_selu(belief)
            belief=env(action)[0][0]
            # print(i,env.time)
            # print(belief)
            # print(action.tolist(), torch_model_selu(belief).tolist())
            beliefs.append(belief)
    return beliefs

def get_init_beliefs(number_trials):
    beliefs=[]
    for i in range(number_trials):
        belief=env.reset()[0]
        beliefs.append(belief)
    return beliefs

# testings, alter r(distance) and relative angle.

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

def plot_policy_surface(belief,torch_model,name_index):
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

def plot_policy_surfaces(belief,torch_model1,name_index):
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
    policy1_data_v=np.zeros((len(r_labels),len(a_labels)))
    policy1_data_w=np.zeros((len(r_labels),len(a_labels)))
    policy2_data_v=np.zeros((len(r_labels),len(a_labels)))
    policy2_data_w=np.zeros((len(r_labels),len(a_labels)))
    for ri in range(len(r_labels)):
        for ai in range(len(a_labels)):
            belief[0]=r_labels[ri]
            belief[1]=a_labels[ai]
            policy1_data_v[ri,ai]=max(torch_model1(belief)[0],0)
            policy1_data_w[ri,ai]=torch_model1(belief)[1]


    fig, ax = plt.subplots(1,2,
                        gridspec_kw={'hspace': 0.2, 'wspace': 0.4},figsize=(20,10))
    fig.suptitle('{} policy surface'.format(torch_model1.name),fontsize=40)
    ax[0].set_title('velocity',fontsize=24)
    ax[0].imshow(policy1_data_v,origin='lower',vmin=-1.,vmax=1.,extent=[a_labels[0],a_labels[-1],r_labels[0],r_labels[-1]])
    ax[0].set_ylabel('relative distance',fontsize=15)
    ax[0].set_xlabel('relative angle',fontsize=15)

    ax[1].set_title('angular velocity',fontsize=24)
    ax[1].set_ylabel('relative distance',fontsize=15)
    ax[1].set_xlabel('relative angle',fontsize=15)
    im=ax[1].imshow(policy1_data_w,origin='lower',vmin=-1.,vmax=1.,extent=[a_labels[0],a_labels[-1],r_labels[0],r_labels[-1]])
    add_colorbar(im)



    plt.savefig('./policy plots/{} policy surface {}.png'.format(torch_model1.name,name_index))


name_index=0
beliefs=get_init_beliefs(1)
for belief in beliefs:
    plot_policy_surfaces(belief,torch_model_selu,name_index=name_index)
    name_index+=1
    print(name_index)

# name_index=0
# for belief in beliefs:
#     plot_policy_surface(belief,baselines_mlp_model,name_index=name_index)
#     name_index+=1
#     print(name_index)
from stable_baselines import DDPG
from stable_baselines import TD3
import policy_torch
import numpy as np
from numpy import pi
from torch import nn
import matplotlib.pyplot as plt
# load baseline model
selu_name='DDPG_selu_4-5vgain1000000_2 28 13 5'
baselines_mlp_model = DDPG.load(selu_name)
torch_model_selu = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[256,256,64,32],act_fn=nn.functional.selu)
torch_model_selu.name='selu'

baselines_mlp_model = DDPG.load("DDPG_theta")
torch_model_relu = policy_torch.copy_mlp_weights(baselines_mlp_model,layers=[32,64])
torch_model_relu.name='relu'

# testing, get sample beliefs
from Config import Config
from FireflyEnv import ffenv
arg=Config()
# arg.std_range=[0.0001,0.001,0.0001,0.001]
# arg.gains_range=[4.,5.,0.99,1.]
env=ffenv.FireflyEnv(arg)

def get_beliefs(number_trials):
    beliefs=[]
    for i in range(number_trials):
        belief=env.reset()[0]
        while not env.stop:
            action=torch_model_relu(belief)
            belief=env(action)[0][0]
            # print(i,env.time)
            # print(belief)
            # print(action.tolist(), torch_model_selu(belief).tolist())
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

def plot_policy_surfaces(belief,torch_model1,torch_model2,name_index):
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
            policy1_data_v[ri,ai]=max(0,torch_model1(belief)[0])
            policy1_data_w[ri,ai]=torch_model1(belief)[1]
            policy2_data_v[ri,ai]=max(0,torch_model2(belief)[0])
            policy2_data_w[ri,ai]=torch_model2(belief)[1]

    fig, ax = plt.subplots(3, 2,
                        gridspec_kw={'hspace': 0.2, 'wspace': 0.4},figsize=(18,20))
    fig.suptitle('{} and {} policy surface'.format(torch_model1.name,torch_model2.name),fontsize=40)
    ax[0,0].set_title('selu velocity',fontsize=24)
    ax[0,0].imshow(policy1_data_v,origin='lower',vmin=-1.,vmax=1.,extent=[a_labels[0],a_labels[-1],r_labels[0],r_labels[-1]])
    ax[0,1].set_title('relu velocity',fontsize=24)
    ax[0,1].imshow(policy2_data_v,origin='lower',vmin=-1.,vmax=1.,extent=[a_labels[0],a_labels[-1],r_labels[0],r_labels[-1]])


    ax[1,0].set_title('selu angular velocity',fontsize=24)
    ax[1,0].imshow(policy1_data_w,origin='lower',vmin=-1.,vmax=1.,extent=[a_labels[0],a_labels[-1],r_labels[0],r_labels[-1]])
    ax[1,1].set_title('relu angular velocity',fontsize=24)
    ax[1,1].imshow(policy2_data_w,origin='lower',vmin=-1.,vmax=1.,extent=[a_labels[0],a_labels[-1],r_labels[0],r_labels[-1]])

    for a in ax.flat:
        a.set(xlabel='relative angle', ylabel='relative distance')
    # for a in ax.flat:
    #     a.label_outer()

    ax[2,0].text(-1.3,+5.5,'time {}'.format(str(belief.tolist()[4])),fontsize=24)
    ax[2,0].text(-3.3,-2.5,'theta {}'.format(str(['{:.2f}'.format(x) for x in (belief.tolist()[20:])])),fontsize=20)
    ax[2,0].text(-2.3,1.5,'scale bar,  -1                0                +1',fontsize=24)

    ax[2,0].imshow((np.asarray(list(range(10)))/10).reshape(1,-1))
    ax[2,0].axis('off')

    ax[2,1].set_title('P matrix',fontsize=24)
    ax[2,1].imshow(inverseCholesky(belief.tolist()[5:20]))
    ax[2,1].axis('off')

    plt.savefig('./policy plots/{} and {} policy surface {}.png'.format(torch_model1.name,torch_model2.name,name_index))


name_index=0
beliefs=get_beliefs(3)
for belief in beliefs:
    # plot_policy_surface(belief,torch_model_selu,name_index)
    # plot_policy_surface(belief,torch_model_relu,name_index)
    plot_policy_surfaces(belief,torch_model_selu, torch_model_relu,name_index=name_index)
    name_index+=1
    print(name_index)

# name_index=0
# for belief in beliefs:
#     # plot_policy_surface(belief,torch_model_selu,name_index)
#     # plot_policy_surface(belief,torch_model_relu,name_index)
#     plot_policy_surface(belief,baselines_mlp_model,name_index=name_index)
#     name_index+=1
#     print(name_index)
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
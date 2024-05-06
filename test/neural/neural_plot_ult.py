
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from IPython.display import clear_output
import matplotlib.colors as mcolors

# cebra embedding plots -------------------------------

def plot_embedding_contrast(ax, embedding, label, gray=False, beh_idx=(0, 1), idx_order=(0, 1, 2)):
    '''plot the embeeding and color by the difference between the beh_idx task varaibles'''
    if not gray:
        r_cmap = 'cool'
        r_c = label[:, beh_idx[0]]-label[:, beh_idx[1]]
    else:
        r_cmap = None
        r_c = 'gray'
    idx1, idx2, idx3 = idx_order
    r = ax.scatter(embedding[:, idx1],
                   embedding[:, idx2],
                   embedding[:, idx3],
                   c=r_c,
                   #    vmin=0,
                   #    vmax=1,
                   cmap=r_cmap, s=0.5)
    plt.colorbar(r)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.set_ticks([-1, 0, 1])
    ax.yaxis.set_ticks([-1, 0, 1])
    ax.zaxis.set_ticks([-1, 0, 1])
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.set_zlabel('dim 3')
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.axis('equal')
    return ax


def plot_embedding(ax, embedding, label, gray=False, beh_idx=0, idx_order=(0, 1, 2)):
    '''cebra 3d embedding'''
    if not gray:
        r_cmap = 'cool'
        r_c = label[:, beh_idx]
    else:
        r_cmap = None
        r_c = 'gray'
    idx1, idx2, idx3 = idx_order
    r = ax.scatter(embedding[:, idx1],
                   embedding[:, idx2],
                   embedding[:, idx3],
                   c=r_c,
                   vmin=0,
                   vmax=1,
                   cmap=r_cmap, s=0.5)
    plt.colorbar(r)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.set_ticks([-1, 0, 1])
    ax.yaxis.set_ticks([-1, 0, 1])
    ax.zaxis.set_ticks([-1, 0, 1])
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.set_zlabel('dim 3')
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.axis('equal')
    return ax


def plot_embedding2d(ax, embedding, label, gray=False, beh_idx=0, idx_order=(0, 1)):
    '''cebra 2d embedding'''
    if not gray:
        r_cmap = 'cool'
        r_c = label[:, beh_idx]
    else:
        r_cmap = None
        r_c = 'gray'
    idx1, idx2 = idx_order
    r = ax.scatter(embedding[:, idx1],
                   embedding[:, idx2],
                   c=r_c,
                   #    vmin=0,
                   #    vmax=1,
                   cmap=r_cmap, s=0.5)
    plt.colorbar(r)
    ax.grid(False)
    ax.xaxis.set_ticks([-1, 0, 1])
    ax.yaxis.set_ticks([-1, 0, 1])
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.axis('equal')
    return ax


def project_and_unfold(x, y):
    # Step 1: Calculate distance of each point from the origin
    distance = np.sqrt(x**2 + y**2)

    # Step 2: Find nearest point on the circle
    radius = 1
    x_projected = x / distance
    y_projected = y / distance

    # Step 3: Unfold the circle onto a line
    angle = np.arctan2(y_projected, x_projected)

    x_unfolded = angle

    y_unfolded = distance - radius

    return x_unfolded, y_unfolded


def plot_embedding2d_unflold_line(ax, embedding, label, gray=False, beh_idx=0, idx_order=(0, 1)):
    '''convert 3d to 2d by mapping dots to ring'''
    if not gray:
        r_cmap = 'cool'
        r_c = label[:, beh_idx]
    else:
        r_cmap = None
        r_c = 'gray'
    idx1, idx2 = idx_order
    x, y = embedding[:, idx1], embedding[:, idx2]
    x_unfolded, y_unfolded = project_and_unfold(x, y)

    r = ax.scatter(x_unfolded,
                   y_unfolded,
                   c=r_c,
                   #    vmin=0,
                   #    vmax=1,
                   cmap=r_cmap, s=0.5)
    plt.colorbar(r)
    ax.grid(False)
    ax.xaxis.set_ticks([-1, 0, 1])
    ax.yaxis.set_ticks([-1, 0, 1])
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.axis('equal')
    return ax


def plot_embedding2d_unflold(ax, embedding, label, gray=False, beh_idx=0, idx_order=(0, 1)):
    '''convert 3d to 2d by mapping dots to ring, ignoring the mapping'''
    if not gray:
        r_cmap = 'cool'
        r_c = label[:, beh_idx]
    else:
        r_cmap = None
        r_c = 'gray'
    idx1, idx2 = idx_order
    x, y = embedding[:, idx1], embedding[:, idx2]
    x_unfolded, y_unfolded = project_and_unfold(x, y)

    r = ax.scatter(x_unfolded,
                   r_c,
                   c=r_c,
                   #    vmin=0,
                   #    vmax=1,
                   cmap=r_cmap, s=0.5)

    ax.grid(False)
    ax.xaxis.set_ticks([-1, 0, 1])
    ax.yaxis.set_ticks([-1, 0, 1])
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.axis('equal')
    corr_coef = np.corrcoef(x_unfolded.squeeze(), r_c.squeeze())[0, 1].item()
    ax.set_title(f'corr = {corr_coef:.2f}')
    return ax


def plot_embedding2d_contrast(ax, embedding, label, gray=False, beh_idx=(0, 1), idx_order=(0, 1), contrast=lambda x, y: x - y, vmin=None, vmax=None):
    '''plot the embeeding and color by the difference between the beh_idx task varaibles'''
    if not gray:
        r_cmap = 'bwr'
        # r_c = label[:, beh_idx[0]] - label[:, beh_idx[1]]
        r_c = contrast(label[:, beh_idx[0]], label[:, beh_idx[1]])
    else:
        r_cmap = None
        r_c = 'gray'
    idx1, idx2 = idx_order
    if not vmin and not vmax:
        norm = mcolors.CenteredNorm(0)
        r = ax.scatter(embedding[:, idx1],
                       embedding[:, idx2],
                       c=r_c,
                       cmap=r_cmap, s=0.5,
                       norm=norm)
    else:
        vmin = -1*max(-vmin, vmax)
        vmax = max(-vmin, vmax)
        r = ax.scatter(embedding[:, idx1],
                       embedding[:, idx2],
                       c=r_c,
                       vmin=vmin,
                       vmax=vmax,
                       cmap=r_cmap, s=0.5)

    plt.colorbar(r)
    ax.grid(False)
    ax.xaxis.set_ticks([-1, 0, 1])
    ax.yaxis.set_ticks([-1, 0, 1])
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')
    ax.axis('equal')
    return ax


# ruiyi neural eye analysis plot -------------------------------
monkey_height = 10
DT = 0.006 # DT for raw data

def distance(x, y):
    return (x**2+y**2)**0.5


def set_violin_plot(vp, facecolor, edgecolor, linewidth=1, alpha=1, ls='-', hatch=r''):
    plt.setp(vp['bodies'], facecolor=facecolor, edgecolor=edgecolor,
             linewidth=linewidth, alpha=alpha, ls=ls, hatch=hatch)
    plt.setp(vp['cmins'], facecolor=facecolor, edgecolor=edgecolor,
             linewidth=linewidth, alpha=alpha)
    plt.setp(vp['cmaxes'], facecolor=facecolor, edgecolor=edgecolor,
             linewidth=linewidth, alpha=alpha)
    plt.setp(vp['cbars'], facecolor=facecolor, edgecolor=edgecolor,
             linewidth=linewidth, alpha=alpha)

    linecolor = 'k' if facecolor == 'None' else 'snow'
    if 'cmedians' in vp:
        plt.setp(vp['cmedians'], facecolor=linecolor, edgecolor=linecolor,
                 linewidth=linewidth, alpha=alpha)
    if 'cmeans' in vp:
        plt.setp(vp['cmeans'], facecolor=linecolor, edgecolor=linecolor,
                 linewidth=linewidth, alpha=alpha)


def downsample(data, bin_size=20):
    num_bin = data.shape[0] // bin_size
    data_ = data[:bin_size * num_bin]
    data_ = data_.reshape(num_bin, bin_size, data.shape[-1])
    data_ = np.nanmean(data_, axis=1)
    return data_


def convert_location_to_angle(gaze_r, gaze_x, gaze_y, body_theta, body_x, body_y, hor_theta_eye, ver_theta_eye, monkey_height=monkey_height, DT=DT, remove_pre=True, remove_post=True):
    '''
        convert the world overhead view location of the 'gaze' location to eye coord. 

        gaze location, the target
        gaze_r, relative distance
        gaze_x, gaze location x
        gaze_y,

        body_theta, heading direction
        body_x, monkey location x
        body_y, 

        hor_theta_eye, actual eye location in eye coord. used here to remove pre saccade (when monkey hasnt seen the target yet)
        ver_theta_eye
    '''

    # hor_theta = -np.rad2deg(np.arctan2(-(gaze_x - body_x), gaze_y - body_y) - (body_theta-np.deg2rad(90))).reshape(-1, 1)
    hor_theta = -np.rad2deg(np.arctan2(-(gaze_x - body_x), np.sqrt((gaze_y - body_y)**2 + monkey_height**2))
                            - (body_theta-np.deg2rad(90))).reshape(-1, 1)

    k = -1 / np.tan(body_theta)
    b = body_y - k * body_x
    gaze_r_sign = (k * gaze_x + b < gaze_y).astype(int)
    gaze_r_sign[gaze_r_sign == 0] = -1
    ver_theta = -np.rad2deg(np.arctan2(monkey_height,
                            gaze_r_sign * gaze_r)).reshape(-1, 1)

    # remove overshooting
    if remove_post:
        overshoot_idx = np.where(((gaze_x - body_x) * gaze_x < 0) | (gaze_y < body_y)
                                 # | (abs(hor_theta.flatten()) > 60)
                                 )[0]
        if overshoot_idx.size > 0:
            hor_theta[overshoot_idx[0]:] = np.nan

        overshoot_idx = np.where((gaze_r_sign < 0)
                                 # | (abs(ver_theta.flatten()) > 60)
                                 )[0]
        if overshoot_idx.size > 0:
            ver_theta[overshoot_idx[0]:] = np.nan

    # detect saccade
    if remove_pre:
        if hor_theta_eye.size > 2:
            saccade = np.sqrt((np.gradient(hor_theta_eye) / DT)**2 +
                              (np.gradient(ver_theta_eye) / DT)**2)
            saccade_start_idx = np.where(saccade > 100)[0]
            saccade_start_idx = saccade_start_idx[0] + \
                16 if saccade_start_idx.size > 0 else None

            hor_theta[:saccade_start_idx] = np.nan
            ver_theta[:saccade_start_idx] = np.nan

    return hor_theta, ver_theta


def compute_error(data1, data2, mask):
    # data1 = data1[~mask]; data2 = data2[~mask]
    # corr = np.corrcoef(data1, data2)
    error = abs(data1 - data2)

    rng = np.random.default_rng(seed=0)
    data1_ = data1.copy()
    data2_ = data2.copy()
    rng.shuffle(data1_)
    rng.shuffle(data2_)
    error_shuffle = abs(data1_ - data2_)
    return error


# ---------------------
from datetime import datetime

def mytime():
    '''get date as str'''
    current_date_time = datetime.now()
    current_date = current_date_time.date()
    formatted_date = current_date.strftime("%m%d")

    return  formatted_date


def normalize_01(data,low=5,high=95):
    '''normalize the data vector or matrix to 0-1 range
    use percentile to avoid outliers.'''
    themin=np.percentile(data[~np.isnan(data)],low)
    themax=np.percentile(data[~np.isnan(data)],high)
    res= (data - themin) / (themax- themin)
    res[np.isnan(data)]=np.nan
    res=np.clip(res, 0,1)
    return res

def normalize_z(data):
    '''normalize the data vector or matrix to have mean of 0 std of 1'''
    nanmask=~np.isnan(data)
    validdata=data[nanmask]
    mean = sum(data[nanmask]) / len(data[nanmask])
    variance = sum((x - mean) ** 2 for x in data[nanmask]) / len(data[nanmask])
    std_deviation = variance ** 0.5
    normalized_data = [(x - mean) / std_deviation  if x else np.nan for x in data]
    return normalized_data


# will generate a circle of goal ( actually a gassian)
# and a cov ellipse
from scipy.stats import norm, chi2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import numpy as np
from numpy import pi
import time
import torch
import gym
from stable_baselines.ddpg.policies import LnMlpPolicy, MlpPolicy
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
from stable_baselines import DDPG, TD3
import TD3_test
from FireflyEnv import ffenv_new_cord, firefly_action_cost, firefly_acc, firefly_accac, firefly_mdp, ffac_1d
from reward_functions import reward_singleff
from Config import Config
arg=Config()
arg.goal_radius_range=[0.1,0.3]
from FireflyEnv.env_utils import vectorLowerCholesky
from plot_ult import add_colorbar

# # no action cost model
# env=ffenv_new_cord.FireflyAgentCenter(arg)
# model=TD3.load('trained_agent/TD_95gamma_mc_500000_0_23_22_8.zip')

# action cost model
env=firefly_action_cost.FireflyActionCost(arg)
# modelname='trained_agent/TD_action_cost_sg_700000_9_11_6_17'# non ranged working model
modelname='trained_agent/TD_action_cost_700000_8_19_21_56'# ranged model last
model=TD3.load(modelname)

# acc model
env=firefly_acc.FireflyAcc(arg)
model=TD3.load('trained_agent/.zip')
model.set_env(env)

# acc ac model
env=firefly_accac.FireflyAccAc(arg)
model=TD3.load('trained_agent/accac_final_1000000_9_11_20_25.zip')
model.set_env(env)

# acc mdp model
env=firefly_mdp.FireflyMDP(arg)
model=TD3.load('trained_agent/mdp_noise_1000000_2_9_18_8')
model.set_env(env)


# 1d acc model
env=ffac_1d.FireflyTrue1d(arg)
model=TD3.load('trained_agent/1d_easy1000000_9_25_5_14')
model.set_env(env)




# series of plots describing the trail
#-------------------------------------------------------
# max_distance=1
# uppertau=8.08
# param_range_dict={  'tau_range':[0.05, uppertau],
#                     }
env.terminal_vel= 0.02 
env.dt=0.2
env.episode_len=35
env.std_range = [0.02,0.1,0.02,0.1]
# env.setup(arg, max_distance=max_distance,param_range_dict=param_range_dict)

decision_info=env.reset(
    pro_noise_stds=torch.Tensor([0.2,0.3]),
    obs_noise_stds=torch.Tensor([0.2,0.3]),
        # pro_gains=torch.Tensor([0.5,1]),
        # obs_gains=torch.Tensor([0.51,1])
)
trial_index=1
decision_info=env.reset(
        goal_position=task_info[trial_index]['pos'],
        phi=task_info[trial_index]['phi'],
        theta=task_info[trial_index]['theta'])

env.reset()
phi=env.phi.detach().clone()
theta=env.theta.detach().clone()

phi[3,0]=9
theta[3,0]=9
phi[1,0]=9
theta[1,0]=9
decision_info=env.reset(
        # goal_position=task_info[trial_index]['pos'],
        phi=phi,
        theta=theta)
# decision_info=env.reset()
done=False
while not done:
    action,_=model.predict(env.decision_info)
    decision_info,_,done,_=env.step(action)
    # fig=plot_belief(env,title=('action',action,"velocity", env.s[-2:],'tau:',env.phi[9]),kwargs={'title':action})
    # fig.savefig("{}.png".format(env.episode_time))
    # print(env.decision_info[0,:2],action)        
    # print(env.s[:,0],en v.phi[:-2,0])
    # print(env.episode_time)
    # print(env.s[3,0],env.b[3,0])
    print(action)


# Bangbang control
env.std_range=[1e-4,1e-3,1e-4,1e-3]
decision_info=env.reset()
action,s=model.predict(env.decision_info)
print(s)


while env.episode_time*env.dt<s:
    decision_info,_,done,_=env.step(action)
    fig=plot_belief(env,title=('CTRL',action,"velocity", env.s[-2:],'tau:',env.phi[9]))
action=[-a for a in action]
while env.s[3,0]>=0:
    decision_info,_,done,_=env.step(action)
    fig=plot_belief(env,title=('-CTRL',action,"velocity", env.s[-2:],'tau:',env.phi[9]))

# bangbang control discrete (more like rl agent)


plot_vw_distribution(env, model, number_trials=10,is1d=True)
correct_rate(env, model, number_trials=100, correct_threshold=0.5)

task_info=get_wrong_trial(env,model)
correct_rate(env, model, number_trials=100, task_info=task_info)


# change 2 param of 1d agent plt
paramnames=['distance to goal',
'current estimation of v',
'time',
'cov xx',
'cov xv',
'cov xv',
'cov vv',
'time',
'control gain',
'control noise std',
'obs gain',
'obs noise std',
'goal radius',
'tau',
'mag cost',
'dev cost'
]
param_pair=[0,10]
param_range=[[0.4,1],[0.001,99]]
number_pixels=10
env.reset()
theta=env.theta
phi=env.phi
decision_info=env.decision_info.clone().detach()
background_data=np.zeros((number_pixels,number_pixels))
X,Y=np.meshgrid(np.linspace(param_range[0][0],param_range[0][1],number_pixels),np.linspace(param_range[1][0],param_range[1][1],number_pixels))
for i in range(background_data.shape[0]):
    for j in range(background_data.shape[1]):
        d=decision_info.clone()[0]
        d[param_pair[0]]=X[i,j]
        d[param_pair[1]]=Y[i,j]
        background_data[i,j]=model.predict(d.view(1,-1))[0]

fig = plt.figure(figsize=[9, 9])
ax = fig.add_subplot(221)
im=ax.imshow(background_data)
add_colorbar(im)
ax.set_title('forward velocity V control of the agent', fontsize=20)
ax.set_xlabel('{}'.format(paramnames[param_pair[0]]), fontsize=15)
ax.set_ylabel('{}'.format(paramnames[param_pair[1]]), fontsize=15)



# action distribution------------------------------
def return_vw_distribution(env, model,number_trials=10, is1d=False):
    env.reset()
    theta=env.theta
    phi=env.phi
    pos= env.goalx if is1d else [env.goalx,env.goaly] 
    theta=env.reset_task_param()
    v=[]
    w=[]
    d=[]
    mu=[]
    cov=[]
    vv=[]
    vw=[]
    if is1d:
        for trial_num in range(number_trials):
            env.reset(theta=theta.detach(), phi=phi.detach(), goal_position=pos)
            # env.reset()
            done=False
            vs=[]
            ds=[]
            mus=[]
            covs=[]
            vvs=[]
            while not done:
                action,_=model.predict(env.decision_info)
                decision_info,_,done,_=env.step(action)
                vs.append(action[0])
                vvs.append(env.s[1])
            # print(vs, ws)
            print('trial finished at : ',env.episode_time)
            v.append(vs)
            vv.append(vvs)
    else:
        for trial_num in range(number_trials):
            env.reset(theta=theta.detach(), phi=phi.detach(), goal_position=pos)
            # env.reset()
            done=False
            vs=[]
            ws=[]
            ds=[]
            mus=[]
            covs=[]
            vvs=[]
            vws=[]
            while not done:
                action,_=model.predict(env.decision_info)
                decision_info,_,done,_=env.step(action)
                vs.append(action[0])
                ws.append(action[1])
                vvs.append(env.s[3])
                vws.append(env.s[4])
                ds.append(decision_info[0].tolist()[3:3+16])
                mus.append(env.b[:2])
                covs.append(env.P[:2,:2])
            # print(vs, ws)
            print('trial finished at : ',env.episode_time)
            v.append(vs)
            w.append(ws)
            d.append(ds)
            mu.append(mus)
            cov.append(covs)
            vv.append(vvs)
            vw.append(vws)
    return v, w, vv, vw


def plot_vw_distribution(env, model, number_trials=10, is1d=False):
    v,w, vv, vw=return_vw_distribution(env, model,number_trials=number_trials,is1d=is1d)
    fig = plt.figure(figsize=[9, 9])
    ax = fig.add_subplot(221)
    ax1 = fig.add_subplot(222)
    ax.set_title('forward velocity V control of the agent', fontsize=20)
    ax.set_xlabel('time steps, dt={}'.format(env.dt), fontsize=15)
    ax.set_ylim([-1.1,1.1])



    ax2 = fig.add_subplot(223)
    ax2.set_xlabel('time steps, dt={}'.format(env.dt), fontsize=15)
    ax2.set_ylabel('v', fontsize=15)
    ax2.set_ylim([-1.1,1.1])

    for vs in v:
        ax.plot(vs)
    for vvs in vv:
        ax2.plot(vvs)

    if not is1d:
        ax1.set_title('angular velocity W control of the agent', fontsize=20)
        ax1.set_xlabel('time steps, dt={}'.format(env.dt), fontsize=15)
        ax.set_ylabel('control, range=[-1,1]', fontsize=15)
        ax1.set_ylabel('control, range=[-1,1]', fontsize=15)
        ax1.set_ylim([-1.1,1.1])
        ax3 = fig.add_subplot(224)
        ax3.set_xlabel('time steps, dt={}'.format(env.dt), fontsize=15)
        ax3.set_ylabel('w', fontsize=15)
        ax3.set_ylim([-1.1,1.1])
        for ws in w:
            ax1.plot(ws) 
        for vws in vw:
            ax3.plot(vws)


    return fig


def correct_rate(env, model, number_trials=100, task_info=None,correct_threshold=5):
    correct=0
    if task_info is list:
        for i in range(len(task_info)):
                env.reset(goal_position=task_info['pos'],
                            phi=task_info['phi'],
                            theta=task_info['theta'])
                done=False
                while not done:
                    action,_=model.predict(env.decision_info)
                    decision_info,_,done,_=env.step(action)
                    # fig.savefig("{}.png".format(env.episode_time))
                if env.episode_reward>5:
                    correct=correct+1

    else:
        for i in range(number_trials):
            if task_info is dict:
                env.reset(goal_position=task_info['pos'],
                            phi=task_info['phi'],
                            theta=task_info['theta'])
            else:
                env.reset()
                done=False
                while not done:
                    action,_=model.predict(env.decision_info)
                    decision_info,_,done,_=env.step(action)
                    # fig.savefig("{}.png".format(env.episode_time))
                if env.episode_reward>correct_threshold:
                    correct=correct+1
    return correct/number_trials


def get_wrong_trial(env,model, number_trials=100):
    tasks=[]
    while len(tasks)<number_trials:
        task_info={}
        while task_info=={}:
            done=False
            while not done:
                action,_=model.predict(env.decision_info)
                decision_info,_,done,_=env.step(action)
                if env.episode_reward<5:
                    task_info['pos']=[env.goalx,env.goaly]
                    task_info['phi']=env.phi
                    task_info['theta']=env.theta
        tasks.append(task_info)
    return tasks
    # v w and controls for some trials    

#------------------------------------------------------
def plot_belief(env,title='title',**kwargs):
    f1=plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.set_ylim(-1.5,1.5)
    ax.set_xlim(-1.5,1.5)
    pos=env.b[:2,0].detach()
    cov=env.P[:2,:2].detach()
    # plt.plot(pos[0],pos[1],'o')
    # print('test',title)
    if kwargs.get('title'):
        print('test',kwargs.get('title'))
        title=kwargs.get('title')
    plt.title(title)
    plt.plot(env.s.detach()[0,0],env.s.detach()[1,0],'ro')
    plt.plot([pos.detach()[0],pos.detach()[0]+env.decision_info.detach()[0,0]*np.cos(env.b.detach()[2,0]+env.decision_info.detach()[0,1]) ],
    [pos.detach()[1],pos.detach()[1]+env.decision_info.detach()[0,0]*np.sin(env.b.detach()[2,0]+env.decision_info.detach()[0,1])],'g')
    plt.quiver(pos.detach()[0], pos.detach()[1],np.cos(env.b.detach()[2,0].item()),np.sin(env.b.detach()[2,0].item()), color='r', scale=10)
    plot_cov_ellipse(cov, pos, nstd=2,ax=ax)
    # plot_cov_ellipse(np.diag([1,1])*0.05, [env.goalx,env.goaly], nstd=1, ax=ax)
    plot_circle(np.eye(2)*env.phi[8,0].item(),[env.goalx,env.goaly],ax=ax,color='y')
    return f1


def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, color=None, ax=None,alpha=0.5, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        figure=plt.figure(figsize=(10,10))
        ax = plt.gca()
        ax.set_ylim(-1.5,1.5)
        ax.set_xlim(-1.5,1.5)


    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    if color is not None:
        ellip.set_color(color)
    ellip.set_alpha(alpha)
    ax.add_artist(ellip)
    return ellip
    

def plot_circle(cov, pos, color=None, ax=None, **kwargs):

    if ax is None:
        figure=plt.figure(figsize=(10,10))
        ax = plt.gca()
        ax.set_ylim(-1.5,1.5)
        ax.set_xlim(-1.5,1.5)
    assert cov[0,0]==cov[1,1]
    r=cov[0,0]
    c = Circle(pos,r)
    if color is not None:
        c.set_color(color)
    c.set_alpha(0.5)
    ax.add_artist(c)
    return c


def overlap_prob(r, cov, mu ):
    # normalizing_z=2*pi*r**2
    rop=cov.copy()
    rop[0,0]=rop[0,0]+r**2
    rop[1,1]=rop[0,0]+r**2

    # normalizing_z=1/2/pi/np.sqrt(np.linalg.det(rop))


    if type(mu) == list:
        vector_mu=np.asarray(mu).reshape((2,1))
    # expectation=np.exp((-1/2/pi/r**2) * vector_mu.transpose()@np.linalg.inv(rop)@vector_mu)
    expectation=np.exp(-1/2* vector_mu.transpose()@np.linalg.inv(rop)@vector_mu)

    return expectation


def overlap_intergration(r, cov, mu, bins=20):
    xrange=[mu[0]-r*1.1,mu[0]+r*1.1]
    yrange=[mu[1]-r*1.1,mu[1]+r*1.1]
    P=0
    a=np.zeros((bins,bins))
    b=np.zeros((bins,bins))
    xs=np.linspace(xrange[0],xrange[1],bins)
    ys=np.linspace(yrange[0],yrange[1],bins)

    for i in range(bins):
        for j in range(bins):
            if (xs[i]-mu[0])**2+(ys[j]-mu[1])**2<=r**2:                
                expectation=( (1/2/pi/np.sqrt(np.linalg.det(cov)))
                * np.exp(-1/2
                    * (np.array([xs[i],ys[j]]).reshape(1,2)@np.linalg.inv(cov)@np.array([xs[i],ys[j]]).reshape(2,1)) ))
                P=P+expectation*4*r*r/bins/bins
                a[i,j]=(expectation*4*r*r/bins/bins)[0]
                b[i,j]=1

    return P


def overlap_mc(r, cov, mu, nsamples=1000):
    # xrange=[-cov[0,0],cov[0,0]]
    # yrange=[-cov[1,1],cov[1,1]]
    # xrange=[mu[0]-r*1.1,mu[0]+r*1.1]
    # yrange=[mu[1]-r*1.1,mu[1]+r*1.1]

    check=[]
    xs, ys = np.random.multivariate_normal(-mu, cov, nsamples).T
    # plot_overlap(r,cov,mu,title=None)
    for i in range(nsamples):
        # plt.plot(xs[i],ys[i],'.')
        if (xs[i])**2+(ys[i])**2<=r**2:                
            check.append(1)
        else:
            check.append(0)
    P=np.mean(check)
    return P


def plot_overlap(r,cov,mu,title=None):
    f1=plt.figure(figsize=(10,10))
    ax = plt.gca()
    ax.set_ylim(-1.5,1.5)
    ax.set_xlim(-1.5,1.5)
    if title is not None:
        ax.title.set_text(str(title))
    plot_cov_ellipse(cov,[0,0],nstd=1,ax=ax)
    plot_circle(np.eye(2)*r,mu,ax=ax,color='r')
    return f1

# mu=[0,0]
# plot_cov_ellipse(cov,[0.,0.])
# plot_cov_ellipse(np.eye(2)*0.1,mu,nstd=1)

# cov=np.array([[0.03,0.03],[0.03,0.05]])
# cov=cov*0.6
# cov=cov*10
# r=0.2
# for i in np.linspace(-0.5,0.5,19):
#     for j in np.linspace(-0.5,0.5,19):
#         plot_overlap(r,cov,[i,j],title=str(str(overlap_intergration(r,cov,[i,j]))+str(overlap_prob(r,cov,[i,j]))))
#         plt.savefig("{}_{}.png".format(str(i),str(j)))

# plot belief with stable baselines agent
# env.reset(
#     pro_noise_stds=torch.Tensor([-0.05*8,-pi/80*8]),
#     obs_noise_stds=torch.Tensor([0.05*8,pi/80*8]))
# env.reset(phi=phi, theta=phi)
# env.reset(theta=theta.detach(), phi=phi.detach(), goal_position=pos)
# while True:
#     env.reset(
#         # pro_noise_stds=torch.Tensor([0.2,0.2])
#             # pro_gains=torch.Tensor([0.5,1]),
#             # obs_gains=torch.Tensor([0.51,1])
#     )
#     done=False
#     # phi=env.phi
#     while not done:
#         action,_=model.predict(env.decision_info)
#         decision_info,_,done,_=env.step(action)
#         # fig=plot_belief(env,title=('action',action,env.phi),kwargs={'title':action})
#         # fig.savefig("{}.png".format(env.episode_time))
#         # print(env.decision_info[0,:2],action)        
#         # print(env.s[:,0],env.phi[:-2,0])
#         # print(env.episode_time)
#     # if env.caculate_reward()==0.:
#     #     break
#     print('final reward ',env.caculate_reward(), env.episode_time)



# plot belief with torch agent, to make sure translate correctly
# import policy_torch
# agent = policy_torch.copy_mlp_weights(model,layers=[128,128])
# env.reset()
# while not env.stop:
#     action = agent(env.decision_info)[0]
#     decision_info,done=env(action, env.theta)
#     fig=plot_belief(env,title=(action,env.phi),kwargs={'title':action})
#     # fig.savefig("{}.png".format(env.episode_time))




# belief distribution of single trial
fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot()
ax.set_xlim([0,1])
ax.set_ylim([-0.5,0.5])
for covs, mus in zip(cov, mu):
    for onecov,onemu in zip(covs, mus):
        plot_cov_ellipse(onecov, onemu, ax=ax,nstd=2,alpha=0.1)

fig = plt.figure(figsize=[8, 8])
ax = fig.add_subplot()
ax.set_xlim([0,1])
ax.set_ylim([-0.5,0.5])
for covs, mus in zip(cov, mu):
    for onecov,onemu in zip(covs, mus):
        ax.scatter(onemu[0],onemu[1])



# # testing decision info distribution 
number_trials=10
d_entry=3
t_entry=9
entry_ls=[]
for trial in range(number_trials):
    entry_ls.append(d[trial][t_entry][d_entry])
plt.plot(entry_ls)

# testing decision info growth 
d_entry=5
trial_entry=7
entry_ls=[]
for t in range(len(d[trial_entry])):
    entry_ls.append(d[trial_entry][t][d_entry])
plt.plot(entry_ls)


# mu=torch.Tensor([ 0.2194, 0.2194])
# cov=torch.Tensor([[5.0709e-05, 4.5831e-06],
#         [4.5831e-06, 5.9715e-07]])
# bins=20
# r=0.1873
# mu=torch.Tensor([ 0., -0.])

# xrange=[mu[0]-goal_radius, mu[0]+goal_radius]
# yrange=[mu[1]-goal_radius, mu[1]+goal_radius]
# P=0
v=[]
w=[]
for i in range(1000):
    decision_info= env.decision_info
    decision_info[:,:9]=env.reset_task_param().view(1,-1)
    action,_=model.predict(decision_info)
    v.append(action[0])
    w.append(action[1])
plt.hist(v,bins=100)
plt.plot([true_action[0],true_action[0]],[0,99],color='r')
plt.hist(w,bins=100)
plt.plot([true_action[1],true_action[1]],[0,99],color='r')


true_action,_=model.predict(env.decision_info)
decision_info,_,done,_=env.step(true_action)



# plot policy surface
def plot_policy_surfaces(decision_info,model):
    r_range=[0.0,1.0]
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
    for ri in range(len(r_labels)):
        for ai in range(len(a_labels)):
            action,_=model.predict(decision_info)
            decision_info[:,0]=r_labels[ri]
            decision_info[:,1]=a_labels[ai]
            action,_=model.predict(decision_info)
            policy1_data_v[ri,ai]=action[0]
            policy1_data_w[ri,ai]=action[1]

    fig, ax = plt.subplots(1, 2,
            gridspec_kw={'hspace': 0.4, 'wspace': 0.2},figsize=(16,16))
    # fig.suptitle('{} and {} policy surface'.format('rua',fontsize=40))
    ax[0].set_title('forward velocity  '+'tau : '+str( env.phi[9]),fontsize=24)
    ax[0].set_xlabel('relative angle',fontsize=18)
    ax[0].set_ylabel('relative distance',fontsize=18)
    im1=ax[0].imshow(policy1_data_v,origin='lower',vmin=-1.,vmax=1.,extent=[a_labels[0],a_labels[-1],r_labels[0],r_labels[-1]])
    # add_colorbar(im1)
    ax[1].set_title('ang velocity',fontsize=24)
    ax[1].set_xlabel('relative angle',fontsize=18)
    ax[1].set_xlabel('relative distance',fontsize=18)
    im2=ax[1].imshow(policy1_data_w,origin='lower',vmin=-1.,vmax=1.,extent=[a_labels[0],a_labels[-1],r_labels[0],r_labels[-1]])
    add_colorbar(im2)
    return fig

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

# test policy surface when change cov
from FireflyEnv.env_utils import *

env.reset()

decision_info= env.decision_info
true_action,_=model.predict(env.decision_info.detach())
_,_,_,_=env.forward(true_action, env.phi)
decision_info[:,5:5+15]=vectorLowerCholesky(env.P)
plot_policy_surfaces(decision_info.detach(),model)

true_action,_=model.predict(env.decision_info)
_,_,_,_=env.step(true_action)
plot_policy_surfaces(env.decision_info,model)


env.episode_time=env.episode_time+10
print(env.episode_time)
env.decision_info[:,2:4]=env.decision_info[:,2:4]*0+1

decision_info=env.wrap_decision_info()
plot_policy_surfaces(decision_info,model)


# ax.set_xlim([0,1])
# ax.set_ylim([-0.5,0.5])


    # for a in ax.flat:
    #     a.set(xlabel='relative angle', ylabel='relative distance')
    # # for a in ax.flat:
    # #     a.label_outer()

    # ax[2,0].text(-1.3,+5.5,'time {}'.format(str(belief.tolist()[4])),fontsize=24)
    # ax[2,0].text(-3.3,-2.5,'theta {}'.format(str(['{:.2f}'.format(x) for x in (belief.tolist()[20:])])),fontsize=20)
    # ax[2,0].text(-2.3,1.5,'scale bar,  -1                0                +1',fontsize=24)

    # ax[2,0].imshow((np.asarray(list(range(10)))/10).reshape(1,-1))
    # ax[2,0].axis('off')

    # ax[2,1].set_title('P matrix',fontsize=24)
    # ax[2,1].imshow(inverseCholesky(belief.tolist()[5:20]))
    # ax[2,1].axis('off')

    # plt.savefig('./policy plots/{} and {} policy surface {}.png'.format(torch_model1.name,torch_model2.name,name_index))




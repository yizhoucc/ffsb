from cProfile import label
from tkinter import PhotoImage
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import numpy as np
from plot_ult import plotoverhead_simple, quickallspine, quicksave, quickspine, run_trial, similar_trials2this, vary_theta,vary_theta_ctrl,vary_theta_new


def calc_svm_decision_boundary(svm_clf, xmin, xmax):
    """Compute the decision boundary"""
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    xx = np.linspace(xmin, xmax, 200)
    yy = -w[0]/w[1] * xx - b/w[1]
    return xx, yy

# Computes the decision boundary of a trained classifier
db_xx, db_yy = calc_svm_decision_boundary(clf, -35, 35)

# Rotate the decision boundary 90° to get perpendicular axis
neg_yy = np.negative(db_yy) 

neg_slope = -1 / -clf.coef_[0][0]
bias = clf.intercept_[0] / clf.coef_[0][1]
ortho_db_yy = neg_slope * db_xx - bias


plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
ax = plt.gca()
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

plt.plot(db_xx,db_yy)

plt.plot(db_xx,ortho_db_yy)





# vary action cost w from healthy --------------------------------------------------------------------

theta_init,theta_final=theta.clone(),theta.clone()
theta_init[8]=0
theta_final[8]=1


vary_theta(agent, env, phi, theta_init, theta_final,5,etask=[0.5,0.3], ntrials=10)

vary_theta_ctrl(agent, env, phi, theta_init, theta_final,3,etask=[0.5,0.3])





# vary ASD to healthy --------------------------------------------------------------------

# load theta distribution
logls=['Z:/human/fixragroup','Z:/human/clusterpaperhgroup']
monkeynames=['ASD', 'Ctrl' ]

mus,covs,errs=[],[],[]
thetas=[]
for inv in logls:
    finaltheta,finalcov, err=process_inv(inv,ind=60,removegr=False)
    mus.append(np.array(finaltheta).reshape(-1))
    covs.append(finalcov)
    errs.append(err)
    thetas.append(finaltheta)
thetas=torch.tensor(mus)
theta_init=thetas[0]
theta_final=thetas[1]

alltag=[]
alltheta=[]
loglls=[]
with open(logls[0],'rb') as f:
    log=pickle.load(f)
    res=[l[2] for l in log[19:99]]
    for r in res:
        for point in r:
            alltheta.append(point[0]) # theta
            loglls.append(point[1])
            alltag.append(0)
with open(logls[1],'rb') as f:
    log=pickle.load(f)
    res=[l[2] for l in log[19:99]]
    for r in res:
        for point in r:
            alltheta.append(point[0]) # theta
            loglls.append(point[1])
            alltag.append(1)

alltheta=np.array(alltheta)
alltag=np.array(alltag)

# compute svm
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(alltheta, alltag)
w = clf.coef_[0] # the normal vector
midpoint=(mus[0]+mus[1])/2
lb=np.array([0,0,0,0,0,0,0.129,0,0,0,0])
hb=np.array([1,2,1,1,1,1,0.131,2,2,1,1])




# vary ASD to healthy
vary_theta(agent, env, phi, theta_init, theta_final,5,etask=[0.5,0.3], ntrials=10)
vary_theta_ctrl(agent, env, phi, theta_init, theta_final,3,etask=[0.5,0.3])


# vary small gain to correct
theta_init=midpoint.copy();theta_final=midpoint.copy()
theta_init[1]=0
theta_final[1]=2
theta_init,theta_final=torch.tensor(theta_init).view(-1,1).float(),torch.tensor(theta_final).view(-1,1).float()
vary_theta_new(agent, env, phi, theta_init, theta_final,5,etask=[1,1], ntrials=20)
quicksave('0 wgain to 2')


# ax=multimonkeytheta(monkeynames, mus, covs, errs, )
# ax.set_yticks([0,1,2])
# ax.plot(np.linspace(-1,9),[2]*50)
# ax.get_figure()



theta_init=np.min(midpoint-lb)/w[np.argmin(hb-midpoint)]*-w*5+midpoint
theta_final=np.min(midpoint-lb)/w[np.argmin(midpoint-lb)]*w*5+midpoint
theta_init,theta_final=torch.tensor(theta_init).view(-1,1).float(),torch.tensor(theta_final).view(-1,1).float()

# make the gains to correct.
midpoint[:2]=[0.5,1.6]
ww=w.copy()
ww[:2]=[0,0]
ww[9:]=[0,0]
ww[6]=0
ww[7:9]=[0,0]
scaler=1/40
theta_init=midpoint-ww*scaler
theta_final=ww*scaler+midpoint
theta_init,theta_final=torch.tensor(theta_init).view(-1,1).float(),torch.tensor(theta_final).view(-1,1).float()


vary_theta(agent, env, phi, theta_init, theta_final,5,etask=[0.5,0.5], ntrials=50,plotallb=False)

# path overhead
two_theta_overhead(agent, env, phi, thetas, labels=['ASD model','Ctrl model'],etask=[0.5,0.5],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=50)
vary_theta_new(agent, env, phi, thetas[0], thetas[1],3,etask=[1,1], ntrials=20)


two_theta_curvaturehist(agent, env, phi, thetas, labels=['ASD model','Ctrl model'],etask=[0.5,0.5],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=50)


two_theta_overhead(agent, env, phi, [theta_init, theta_final], labels=['ASD model','Ctrl model'],etask=[0.5,0.5],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=99)

two_theta_curvaturehist(agent, env, phi, [theta_init, theta_final], labels=['ASD model','Ctrl model'],etask=[0.5,0.5],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=50)


vary_theta(agent, env, phi, theta_init, theta_final,3,etask=[2,2], ntrials=20)
vary_theta_new(agent, env, phi, theta_init, theta_final,3,etask=[1,1], ntrials=20)

two_theta_overhead(agent, env, phi, thetas, labels=['ASD model','Ctrl model'],etask=[1,1],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=20)


atheta=thetas[0]
awctheta=thetas[0].clone()
awctheta[8]=1
vary_theta(agent, env, phi, atheta, awctheta, 3, etask=[0.6,0.5], ntrials=20)

env.episode_len=40






lowcost=torch.tensor([
        [0.8],
        [0.7],

        [0.7],
        [0.3],
        [0.7],
        [0.9],

        [1.3000e-01],

        [0.5],
        [0.5],
        [1.0000e-1],
        [1.0000e-1]])

highcost=torch.tensor([
        [0.8],
        [1.57],

        [0.7],
        [0.9],
        [0.7],
        [0.3],

        [1.3000e-01],

        [0.5],
        [0.5],
        [1.0000e-1],
        [1.0000e-1]])


# two_theta_overhead(agent, env, phi, [lowcost, highcost], labels=['us obs','use prediction'],etask=[0.6,0.5],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=50)


vary_theta_new(agent, env, phi, lowcost, highcost,3,etask=[1,1], ntrials=20,plotallb=False)







lowcost=torch.tensor([
        [0.5],
        [0.7],

        [0.5],
        [0.5],
        [0.5],
        [0.9],

        [1.3000e-01],

        [0.5],
        [0.0],
        [1.0000e-1],
        [1.0000e-1]])

highcost=torch.tensor([
        [0.5],
        [0.7],

        [0.5],
        [0.5],
        [0.5],
        [0.5],

        [1.3000e-01],

        [0.5],
        [0.9],
        [1.0000e-1],
        [1.0000e-1]])

vary_theta(agent, env, phi, lowcost, highcost,5,etask=[0.6,0.5], ntrials=20,plotallb=False)

two_theta_overhead(agent, env, phi, [lowcost, highcost], labels=['high noise','high cost'],etask=[0.6,0.5],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=50)



two_theta_curvaturehist(agent, env, phi, [lowcost, highcost], labels=['ASD model','Ctrl model'],etask=[0.5,0.5],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=50)




# when varying, plot human data with model response
datapath=Path("Z:/human/hgroup")
with open(datapath, 'rb') as f:
    hstates, hactions, htasks = pickle.load(f)
datapath=Path("Z:/human/agroup")
with open(datapath, 'rb') as f:
    astates, aactions, atasks = pickle.load(f)

thetask=[0.5,0.3]
healthytheta=thetas[0].view(-1,1)
indls=similar_trials2this(htasks,thetask,ntrial=5)
substates=[hstates[i] for i in indls]
subactions=[hactions[i] for i in indls]
modelstates,modelactions=run_trials(agent, env, phi, healthytheta, thetask,ntrials=10)
plotoverhead_simple(substates,thetask,color='b',label='label',ax=None)
plotoverhead_simple(modelstates,thetask,color='b',label='label',ax=None)

for a in subactions:
    plt.plot(a,color='b')
for a in modelactions:
    plt.plot(a,color='r')


asdtheta=thetas[1].view(-1,1)
indls=similar_trials2this(atasks,thetask,ntrial=5)
substates=[astates[i] for i in indls]
subactions=[aactions[i] for i in indls]
modelstates,modelactions=run_trials(agent, env, phi, asdtheta, thetask,ntrials=10)
plotoverhead_simple(substates,thetask,color='b',label='label',ax=None)
plotoverhead_simple(modelstates,thetask,color='r',label='label',ax=None)

for a in subactions:
    plt.plot(a,color='b')
for a in modelactions:
    plt.plot(a,color='r')


# vary two param and plot path
nplots=5
theta_inita,theta_final=lowcost, highcost
res=[]
smallgain=torch.tensor([
        [0.5],
        [0.7],

        [0.009],
        [0.009],
        [0.009],
        [0.009],

        [1.3000e-01],

        [0.5],
        [0.0],
        [1.0000e-1],
        [1.0000e-1]])
phi=torch.tensor([
        [1],
        [1.57],
        [0.009],
        [0.009],
        [0.009],
        [0.009],
        [1.3000e-01],
        [0.5],
        [0.0],
        [1.0000e-1],
        [1.0000e-1]])
def varyparampath(theta,thetask=[0.5,0.5],ntrials=3):
    states,actions=run_trials(agent, env, phi, theta, thetask,ntrials=ntrials)
    sharedt=min([len(a)-1 for a in actions])
    sharedxy=np.mean(np.array(torch.stack([s[:sharedt,:2] for s in states])),axis=0)
    return sharedxy

gridreso=3
dx,dy=np.zeros((11)),np.zeros((11))
dx[5]=1 # obs noise w
dy[8]=1 # cost w
X=np.linspace(0,2,gridreso)
Y=np.linspace(0,2,gridreso)
paramls=[]
cs=[]
for j in range(gridreso):
    for i in range(gridreso):
        theta=torch.tensor(dx*X[i]+dy*Y[j]+np.array(smallgain).reshape(-1)).float().view(-1,1)
        paramls.append(theta)
        c=np.array([1,0,0])*X[i]/2+np.array([0,0,1])*Y[j]/2 
        cs.append(c)
res=[varyparampath(each,ntrials=30) for each in paramls]
# mints=min([len(r[0]) for r in res])
xy=np.array([s[:] for s in res])
with initiate_plot(2,2,300) as f:
    ax=f.add_subplot(111)
    for z,c in zip(xy,cs):
        ax.plot(z[:,0]*2,z[:,1]*2,color=c,alpha=0.7)
        # ax.scatter(z[-1,0]*200,z[-1,1]*200,color=color,edgecolor='none',alpha=0.7)
    quickspine(ax)
    ax.axis('equal')
    ax.set_xlabel('world x [m]')
    ax.set_ylabel('world y [m]')

with initiate_plot(2,2,300) as f:
    ax=f.add_subplot(111)
    ax.imshow(np.array(cs).reshape((gridreso,gridreso,3)),origin='lower',extent=[0,2,0,2])
    quickallspine(ax)
    ax.set_xticks([0,2])
    ax.set_yticks([0,2])
    ax.set_xlabel('noise')
    ax.set_ylabel('cost')
    
xy=np.array([s[:] for s in res])
with initiate_plot(2,2,300) as f:
    ax=f.add_subplot(111)
    _cmap=sns.color_palette("rocket", n_colors=5)
    for z,c in zip(xy,_cmap):
            ax.plot(z[:,0]*200,z[:,1]*200,color=c)
            ax.scatter(z[-1,0]*200,z[-1,1]*200,color=c)
    norm = plt.Normalize(2,0)
    sm = plt.cm.ScalarMappable(cmap="rocket", norm=norm)
    sm.set_array([])
    plt.colorbar(sm,label='w control cost')
    ax.set_xlabel('world x [cm]')
    ax.set_ylabel('world y [cm]')
    quickspine(ax)
    ax.axis('equal')
plt.show()


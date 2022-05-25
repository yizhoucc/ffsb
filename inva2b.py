from cProfile import label
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import numpy as np
from torch import scatter

# we create 40 separable points
X, y = make_blobs(n_samples=400, centers=2, random_state=6)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(X, y)



# # plot support vectors
# ax.scatter(
#     clf.support_vectors_[:, 0],
#     clf.support_vectors_[:, 1],
#     s=100,
#     linewidth=1,
#     facecolors="none",
#     edgecolors="k",
# )
# plt.show()

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


# # get the separating hyperplane
# w = clf.coef_[0]
# a = -w[0] / w[1]
# xx = np.linspace(-5, 5)
# yy = a * xx - (clf.intercept_[0]) / w[1]


# plot the decision function

# plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
# ax = plt.gca()
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
    

ax=multimonkeytheta(monkeynames, mus, covs, errs, )
ax.set_yticks([0,1,2])
ax.plot(np.linspace(-1,9),[2]*50)
ax.get_figure()

# load theta distribution
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
thetas=torch.tensor(mus)

theta_init=np.min(midpoint-lb)/w[np.argmin(hb-midpoint)]*-w*3+midpoint
theta_final=np.min(midpoint-lb)/w[np.argmin(midpoint-lb)]*w*3+midpoint
theta_init,theta_final=torch.tensor(theta_init).view(-1,1).float(),torch.tensor(theta_final).view(-1,1).float()


theta_init=thetas[0]
theta_final=thetas[1]


vary_theta(agent, env, phi, theta_init, theta_final,3,etask=[0.6,0.5], ntrials=20)


two_theta_overhead(agent, env, phi, thetas, labels=['ASD model','Ctrl model'],etask=[0.6,0.5],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=20)



vary_theta(agent, env, phi, theta_init, theta_final,3,etask=[2,2], ntrials=20)


two_theta_overhead(agent, env, phi, thetas, labels=['ASD model','Ctrl model'],etask=[1,0.7],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=20)


atheta=thetas[0]
awctheta=thetas[0].clone()
awctheta[8]=1
vary_theta(agent, env, phi, atheta, awctheta, 3, etask=[0.6,0.5], ntrials=20)

env.episode_len=40






lowcost=torch.tensor([[5.0000e-01],
        [1.5708e+00],

        [5.0000e-01],
        [0.9],
        [5.0000e-01],
        [0.2],

        [1.3000e-01],

        [0.2],
        [0],
        [1.0000e-1],
        [1.0000e-1]])
highcost=torch.tensor([[5.0000e-01],
        [1.5708e+00],

        [5.0000e-01],
        [0.2],
        [5.0000e-01],
        [0.9],

        [1.3000e-01],

        [0.2],
        [1],
        [1.0000e-1],
        [1.0000e-1]])

two_theta_overhead(agent, env, phi, [lowcost, highcost], labels=['lowcost','highcost'],etask=[0.6,0.6],initv=0.,initw=0.,mkactions=None, pert=None,ntrials=20)


vary_theta(agent, env, phi, lowcost, highcost,3,etask=[0.7,0.5], ntrials=20)

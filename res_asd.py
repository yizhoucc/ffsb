
# for res_asd figures
import numpy as np
from plot_ult import * 
from scipy import stats 


# ASD theta bar ------------------------
logls=['Z:/human/fixragroup','Z:/human/clusterpaperhgroup']
monkeynames=['ASD', 'Ctrl' ]
mus,covs,errs=[],[],[]
for inv in logls:
    finaltheta,finalcov, err=process_inv(inv,ind=60)
    mus.append(finaltheta)
    covs.append(finalcov)
    errs.append(err)

ax=multimonkeytheta(monkeynames, mus, covs, errs, )
ax.set_yticks([0,1,2])
ax.plot(np.linspace(-1,9),[2]*50)
ax.get_figure()



# % behavioral small gain prior, and we confirmed it ------------------------

# load human without feedback data
datapath=Path("Z:/human/wohgroup")
with open(datapath, 'rb') as f:
    hstates, hactions, htasks = pickle.load(f)

datapath=Path("Z:/human/woagroup")
with open(datapath, 'rb') as f:
    astates, aactions, atasks = pickle.load(f)
# get the side tasks (stright trials do not have curvature)
res=[]
for task in htasks:
    d,a=xy2pol(task, rotation=False)
    # if  env.min_angle/2<=a<env.max_angle/2:
    if a<=-pi/5*0.7 or a>=pi/5*0.7:
        res.append(task)
sidetasks=np.array(res)

ares=np.array([s[-1].tolist() for s in astates])
# radial and angular distance response
ardist=np.linalg.norm(ares[:,:2],axis=1)
aadist=np.arctan2(ares[:,1],ares[:,0])
# radial and angular distance target
atasksda=np.array([xy2pol(t,rotation=False) for t in atasks])
artar=atasksda[:,0]
aatar=atasksda[:,1] # hatar=np.arctan2(htasks[:,1],htasks[:,0])
artarind=np.argsort(artar)
aatarind=sorted(aatar)


hres=np.array([s[-1].tolist() for s in hstates])
# radial and angular distance response
hrdist=np.linalg.norm(hres[:,:2],axis=1)
hadist=np.arctan2(hres[:,1],hres[:,0])
# radial and angular distance target
htasksda=np.array([xy2pol(t,rotation=False) for t in htasks])
hrtar=htasksda[:,0]
hatar=htasksda[:,1] # hatar=np.arctan2(htasks[:,1],htasks[:,0])
hrtarind=np.argsort(hrtar)
hatarind=sorted(hatar)


# plot the radial error and angular error

with initiate_plot(4,2,300) as f:
    ax=f.add_subplot(121)
    ax.scatter(hatar,hadist,s=1,alpha=0.5,color='b')
    ax.scatter(aatar,aadist,s=1,alpha=0.2,color='r')
    ax.set_xlim(-0.7,0.7)
    ax.set_ylim(-2,2)
    ax.plot([-1,1],[-1,1],'k',alpha=0.5)
    ax.set_xlabel('target angle')
    ax.set_ylabel('response angle')
    # ax.axis('equal')
    quickspine(ax)

    ax=f.add_subplot(122)
    ax.scatter(hrtar,hrdist,s=1,alpha=0.5,color='b')
    ax.scatter(artar,ardist,s=1,alpha=0.3,color='r')
    # ax.plot([.5,3],[.5,3],'k',alpha=0.5)
    ax.set_xlim(.5,3)
    ax.set_ylim(0.5,5)
    ax.plot([0,3],[0,3],'k',alpha=0.5)
    ax.set_xlabel('target distance')
    ax.set_ylabel('response distance')
    quickspine(ax)
    # ax.axis('equal')
    plt.tight_layout()

# plot the inferred gains (biases?)

# or plot the model reproduced radial error and angular error





# per subject ------------------------

# behavior 
datapath=Path("Z:/human/hgroup")
with open(datapath, 'rb') as f:
    hstates, hactions, htasks = pickle.load(f)

datapath=Path("Z:/human/agroup")
with open(datapath, 'rb') as f:
    astates, aactions, atasks = pickle.load(f)

filename='Z:/human/fbsimple.mat'
data=loadmat(filename)

# seperate into two groups
hdata,adata=[],[]
for d in data:
    if d['name'][0]=='A':
        adata.append(d)
    else:
        hdata.append(d)
print('we have these number of health and autiusm subjects',len(hdata),len(adata))

hsublen=[len(eachsub['targ']['r']) for eachsub in hdata]
asublen=[len(eachsub['targ']['r']) for eachsub in adata]
hcumsum=np.cumsum(hsublen)
acumsum=np.cumsum(asublen)

for isub in range(len(hdata)):
    s=0 if isub==0 else hcumsum[isub-1]
    e=hcumsum[isub]
    substates=hstates[s:e]
    subtasks=htasks[s:e]
    ax=quickoverhead_state(substates,subtasks)
    print(invres['h'][isub][0])


for isub in range(len(adata)):
    s=0 if isub==0 else acumsum[isub-1]
    e=acumsum[isub]
    substates=astates[s:e]
    subtasks=atasks[s:e]
    ax=quickoverhead_state(substates,subtasks)
    print(invres['a'][isub][0])


# load inv data
numhsub,numasub=25,14
foldername='persub1cont'
logs={'a':'Z:/human/fixragroup','h':'Z:/human/clusterpaperhgroup'}

invres={'a':[],'h':[]}
for isub in range(numhsub):
    dataname="hsub{}".format(str(isub))
    savename=Path("Z:/human/{}".format(foldername))/"invhsub{}".format(str(isub))
    if savename.is_file():
        invres['h'].append(process_inv(savename,ind=31, usingbest=True))
for isub in range(numasub):
    dataname="asub{}".format(str(isub))
    savename=Path("Z:/human/{}".format(foldername))/"invasub{}".format(str(isub))
    if savename.is_file():
        invres['a'].append(process_inv(savename,ind=31, usingbest=True))

# plot the inv res
with initiate_plot(4,2,300) as fig:
    ax=fig.add_subplot(111)
    quickspine(ax)
    ax.set_xticks(list(range(10)))
    ax.set_xticklabels(theta_names, rotation=45, ha='right')
    ax.set_ylabel('inferred param value')
    x=np.array(list(range(10)))
    for log in invres['h']:
        # log=invres['a'][0]
        y=log[0]
        plt.scatter(x,y,color='b',alpha=0.1, edgecolors='none',label='control')
    for log in invres['a']:
        # log=invres['a'][0]
        y=log[0]
        plt.scatter(x+0.3,y,color='r',alpha=0.1, edgecolors='none',label='ASD')
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    leg=ax.legend(by_label.values(), by_label.keys(),loc='upper right',bbox_to_anchor=(0,0))
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    # plot the group mean 
    finaltheta,_, err=process_inv('Z:/human/clusterpaperhgroup',ind=60)
    plt.errorbar(list(range(10)), finaltheta.view(-1), yerr = err, err = None, ls='none', color='b') 
    finaltheta,_, err=process_inv('Z:/human/fixragroup',ind=60)
    plt.errorbar(np.array(list(range(10)))+0.3, finaltheta.view(-1), yerr = err, err = None, ls='none', color='r') 




hy=np.array([np.array(log[0].view(-1)) for log in invres['h']])
ay=np.array([np.array(log[0].view(-1)) for log in invres['a']])

plt.errorbar(x, np.mean(hy,axis=0), np.std(hy,axis=0))
plt.errorbar(x+0.3, np.mean(ay,axis=0), np.std(ay,axis=0))


# biased degree v
''' bias*(p/p+o) '''
biash=(pi/2-hy[:,0])/(pi/2) * (hy[:,2]/(hy[:,2]+hy[:,4]))
biasa=(pi/2-ay[:,0])/(pi/2) * (ay[:,2]/(ay[:,2]+ay[:,4]))
plt.figure(figsize=(1,3))
plt.scatter([0]*len(biash), biash,alpha=0.1)
plt.scatter([1]*len(biasa), biasa,alpha=0.1)

# biased degree w
biash=(pi/2-hy[:,1])/(pi/2) * (hy[:,3]/(hy[:,3]+hy[:,5]))
biasa=(pi/2-ay[:,1])/(pi/2) * (ay[:,3]/(ay[:,3]+ay[:,5]))
plt.figure(figsize=(1,3))
plt.scatter([0]*len(biash), biash,alpha=0.1)
plt.scatter([1]*len(biasa), biasa,alpha=0.1)

# cost w
plt.figure(figsize=(1,3))
plt.scatter([0]*len(biash), hy[:,7],alpha=0.3)
plt.scatter([1]*len(biasa), ay[:,7],alpha=0.3)

stats.ttest_ind(biash,biasa)
stats.ttest_ind(hy[:,1],ay[:,1])
stats.ttest_ind(hy[:,7],ay[:,7])

npsummary(biash)
npsummary(biasa)

# use svm to tell two group apart
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
from plot_ult import plotoverhead_simple, quickallspine, quickspine, run_trial, similar_trials2this, vary_theta,vary_theta_ctrl,vary_theta_new

X=[]
Y=[]
for log in invres['h']:
    X.append(np.array(log[0].view(-1)))
    Y.append(0)
for log in invres['a']:
    X.append(np.array(log[0].view(-1)))
    Y.append(1)
X=np.array(X)
Y=np.array(Y)

clf = svm.SVC(kernel="linear", C=1000)
clf.fit(X, Y)

def calc_svm_decision_boundary(svm_clf, xmin, xmax):
    """Compute the decision boundary"""
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]
    xx = np.linspace(xmin, xmax, 200)
    yy = -w[0]/w[1] * xx - b/w[1]
    return xx, yy

db_xx, db_yy = calc_svm_decision_boundary(clf, -35, 35)
neg_yy = np.negative(db_yy) 

neg_slope = -1 / -clf.coef_[0][0]
bias = clf.intercept_[0] / clf.coef_[0][1]
ortho_db_yy = neg_slope * db_xx - bias

plt.scatter(X[:, 5], X[:, 7], c=Y, s=30)
ax = plt.gca()


plt.plot(db_xx,db_yy)

plt.plot(db_xx,ortho_db_yy)

plt.plot(np.abs(clf.coef_[0]))
plt.plot(np.zeros_like(clf.coef_[0]))


pxy,ev,evect=pca(X)

pxy=pxy[:,:2]

im=plt.scatter(pxy[:,0],pxy[:,1],s=5,c=Y)
plt.colorbar(im)



thetas=[np.array(mu.view(-1)) for mu in mus]

plt.plot((ay-thetas[0]).T)
plt.plot((hy-thetas[1]).T)

np.mean((ay-thetas[0]).T,axis=1)

np.mean((hy-thetas[1]),axis=0)
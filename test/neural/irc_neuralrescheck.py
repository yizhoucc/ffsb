

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import numpy as np
import pickle
from pathlib import Path
import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
os.chdir('../..')
print(os.getcwd())
import random
import configparser
from plot_ult import *
import scipy.interpolate as interpolate

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def splineDesign(knots, x, ord=4, der=0, outer_ok=False):
    """Reproduces behavior of R function splineDesign() for use by ns(). See R documentation for more information.

    Python code uses scipy.interpolate.splev to get B-spline basis functions, while R code calls C.
    Note that der is the same across x."""
    knots = np.array(knots, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    xorig = x.copy()
    not_nan = ~np.isnan(xorig)
    nx = x.shape[0]
    knots.sort()
    nk = knots.shape[0]
    need_outer = any(x[not_nan] < knots[ord - 1]) or any(x[not_nan] > knots[nk - ord])
    in_x = (x >= knots[0]) & (x <= knots[-1]) & not_nan

    if need_outer:
        if outer_ok:
            # print('knots do not contain the data range')

            out_x = ~all(in_x)
            if out_x:
                x = x[in_x]
                nnx = x.shape[0]
            dkn = np.diff(knots)[::-1]
            reps_start = ord - 1
            if any(dkn > 0):
                reps_end = max(0, ord - np.where(dkn > 0)[0][0] - 1)
            else:
                reps_end = np.nan  # this should give an error, since all knots are the same
            idx = [0] * (ord - 1) + list(range(nk)) + [nk - 1] * reps_end
            knots = knots[idx]
        else:
            raise ValueError("the 'x' data must be in the range %f to %f unless you set outer_ok==True'" % (
            knots[ord - 1], knots[nk - ord]))
    else:
        reps_start = 0
        reps_end = 0
    if (not need_outer) and any(~not_nan):
        x = x[in_x]
    idx0 = reps_start
    idx1 = len(knots) - ord - reps_end
    cycleOver = np.arange(idx0, idx1)
    m = len(knots) - ord
    v = np.zeros((cycleOver.shape[0], len(x)), dtype=np.float64)
    # v = np.zeros((m, len(x)))

    d = np.eye(m, len(knots))
    for i in range(cycleOver.shape[0]):
        v[i] = interpolate.splev(x, (knots, d[cycleOver[i]], ord - 1), der=der)
        # v[i] = interpolate.splev(x, (knots, d[i], ord - 1), der=der)

    # before = np.sum(xorig[not_nan] < knots[0])
    # after = np.sum(xorig[not_nan] > knots[-1])
    design = np.zeros((v.shape[0], xorig.shape[0]), dtype=np.float64)
    for i in range(v.shape[0]):
    #     design[i, before:xorig.shape[0] - after] = v[i]
        design[i,in_x] = v[i]


    return design.transpose()

def convolve_neuron(spk, trial_idx, bX):
    kernel_len  = bX.shape[1]
    agument_spk_size = spk.shape[0] + 2 * kernel_len * np.unique(trial_idx).shape[0]
    agument_spk = np.zeros(agument_spk_size)
    reverse = np.zeros(agument_spk.shape[0],dtype=bool)
    cc = 0
    for tr in np.unique(trial_idx):
        sel = trial_idx == tr
        agument_spk[cc:cc+sel.sum()] = spk[sel]
        reverse[cc:cc+sel.sum()] = True
        cc += sel.sum() + 2 * kernel_len
    modelX = np.zeros((spk.shape[0],bX.shape[1]))
    for k in range(bX.shape[1]):
        xsm = np.convolve(agument_spk, bX[:,k],'same')
        modelX[:,k] = xsm[reverse]
    return modelX

def convolve_loop(spks, trial_idx, bX):
    modelX = np.zeros((spks.shape[1], spks.shape[0]*bX.shape[1]))
    cc = 0
    for neu in range(spks.shape[0]):
        print(neu)
        modelX[:,cc:cc+bX.shape[1]] = convolve_neuron(spks[neu], trial_idx, bX)
        cc += bX.shape[1]
    return modelX

def splitdata(s, mask=None):
    n = len(s)
    if mask is not None:
        pass
    else:
        mask = np.array(random.sample(range(0, n), n//10*9))

    negmask = np.zeros(len(s), dtype=bool)
    negmask[mask] = True
    train = s[mask]
    test = s[negmask]
    return train, test, mask



# load belief and neural data -------------------
config = configparser.ConfigParser()
config.read_file(open('privateconfig'))
resdir=config['Datafolder']['data']
resdir = Path(resdir)/'neuraltest'


with open(resdir/'res/0220collapsemodelbelief', 'rb') as f:
    res = pickle.load(f)

y_ = res['y']
X = {k: res[k] for k in ['rad_vel', 'ang_vel', 'x_monk', 'y_monk']}
trial_idx = res['trial_idx']
beliefs = res['belief']
covs = res['cov']
s = np.vstack([v for v in X.values()])
s = s.T

s.shape
beliefs.shape
y_.shape

# define a b-spline
kernel_len = 7 # should be about +- 325ms 
knots = np.hstack(([-1.001]*3, np.linspace(-1.001,1.001,5), [1.001]*3))
tp = np.linspace(-1.,1.,kernel_len)
bX = splineDesign(knots, tp, ord=4, der=0, outer_ok=False)
with initiate_plot(3,2,200) as f:
    ax=f.add_subplot(111)
    plt.plot(bX)
    plt.title('B-spline kernel')
    quickspine(ax)
    plt.xticks([0,kernel_len-1])
    ax.set_xticklabels([-kernel_len*50, kernel_len*50])
    plt.xlabel('time, ms')
    plt.ylabel('coef')


modelX = convolve_loop(y_.T, trial_idx, bX) # ts, neurons
pos_xy = np.hstack((X['x_monk'].reshape(-1,1), X['y_monk'].reshape(-1,1))) # ts, xy

# remove bad data
non_nan = ~np.isnan(pos_xy.sum(axis=1))
modelX = modelX[non_nan]
pos_xy = pos_xy[non_nan]
belief_xy = beliefs[:,[0,1]][non_nan]

# make sure belief is right
ind=0
ind+=1
trialind=list(set(trial_idx[non_nan]))[ind]
xy=pos_xy[trial_idx[non_nan]==trialind]
plt.plot(xy[:,0], xy[:,1])
xy=belief_xy[trial_idx[non_nan]==trialind]
plt.plot(-xy[:,1], xy[:,0])
plt.scatter(0,0, color='k', marker='v')
plt.axis('equal')




model = Ridge()
clf = GridSearchCV(model, {'alpha':[0.001,0.01,0.1,1,10,100,1000]})
clf.fit(modelX, pos_xy)

model = Ridge()
clf = GridSearchCV(model, {'alpha':[0.001,0.01,0.1,1,10,100,1000]})
clf.fit(modelX, belief_xy)


from sklearn import linear_model
linreg = linear_model.LinearRegression()
linreg.fit(modelX, belief_xy)
print('linear regression score',linreg.score(modelX, belief_xy))
pred  = linreg.predict(modelX)

every=5
with initiate_plot(5, 3, 200) as f:
    ax=f.add_subplot(121)
    plt.scatter(pred[::every,1],belief_xy[::every,1],s=1, alpha=0.3)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('pos x')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)
    ax=f.add_subplot(122)
    plt.scatter(pred[::every,0],belief_xy[::every,0],s=1, alpha=0.3)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.title('pos y')
    ax.axis('equal')
    ax.plot(ax.get_ylim(),ax.get_ylim(),'k')
    quickspine(ax)

    plt.tight_layout()




# linear fit ----------------------------------
b = beliefs
b[[0,1,3]]=b[[0,1,3]]*500
b[[2,4]]=b[[2,4]]*180/pi
s = np.vstack([v for v in X.values()])
s[np.isnan(s) == True] = 0
s = s.T
y = y.T
b = b.T



# neural encoding belief
trainx, testx, mask = splitdata(y)

trainy, testy, _ = splitdata(b[:,[0,1,3,4]], mask=mask)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy)
y_pred = reg.predict(testx)
print("Mean squared error: %.2f" % mean_squared_error(testy, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(testy, y_pred))

plt.axis('equal')
plt.scatter(b[:,1], b[:,0], s=0.1)
y_pred = reg.predict(y)
plt.scatter(y_pred[:,1], y_pred[:,0], s=1)
plt.show()

plt.scatter(b[:,4], b[:,3], s=0.1)
y_pred = reg.predict(y)
plt.scatter(y_pred[:,3], y_pred[:,2], s=1)
plt.show()

plt.imshow(reg.coef_.reshape(20,-1))
plt.show()

trainy, testy, _ = splitdata(s, mask=mask)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy)
y_pred = reg.predict(testx)
print("Mean squared error: %.2f" % mean_squared_error(testy, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(testy, y_pred))

plt.axis('equal')
plt.scatter(s[:,2], s[:,3], s=0.1)
y_pred = reg.predict(y)
plt.scatter(y_pred[:,2], y_pred[:,3], s=1)
plt.show()

plt.scatter(s[:,1], s[:,0], s=0.1)
y_pred = reg.predict(y)
plt.scatter(y_pred[:,1], y_pred[:,0], s=1)
plt.show()


plt.imshow(reg.coef_.reshape(20,-1))




# test decoding position only
trainx, testx, mask = splitdata(y)

trainy, testy, _ = splitdata(b[:,[0,1]], mask=mask)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy, covs[mask][:,0,0]**0.5)
y_pred = reg.predict(testx)
print("Mean squared error: %.2f" % mean_squared_error(testy, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(testy, y_pred))

with initiate_plot(2,2,200) as fig:
    ax=fig.add_subplot(111)
    plt.axis('equal')
    plt.scatter(b[:,1], b[:,0], s=0.1, label='belief position')
    y_pred = reg.predict(y)
    plt.scatter(y_pred[:,1], y_pred[:,0], s=1, label='neural encoded belief position')
    plt.xlim(-400,400)
    plt.ylim(-100,500)
    plt.xlabel('world x, cm')
    plt.ylabel('world y, cm')
    quickspine(ax)
    # quickleg(ax)
    plt.show()

    
trainy, testy, _ = splitdata(s[:,[2,3]], mask=mask)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy)
y_pred = reg.predict(testx)
print("Mean squared error: %.2f" % mean_squared_error(testy, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(testy, y_pred))

with initiate_plot(2,2,200) as fig:
    ax=fig.add_subplot(111)
    plt.axis('equal')
    plt.scatter(s[:,2], s[:,3], s=0.1)
    y_pred = reg.predict(y)
    plt.scatter(y_pred[:,0], y_pred[:,1], s=1)
    plt.xlabel('world x, cm')
    plt.ylabel('world y, cm')
    quickspine(ax)
    plt.show()

    




# test decoding vel only
trainx, testx, mask = splitdata(y)

trainy, testy, _ = splitdata(b[:,[3,4]], mask=mask)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy)
y_pred = reg.predict(testx)
print("Mean squared error: %.2f" % mean_squared_error(testy, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(testy, y_pred))

plt.axis('equal')
plt.scatter(b[:,4], b[:,3], s=0.1)
y_pred = reg.predict(y)
plt.scatter(y_pred[:,1], y_pred[:,0], s=1)
plt.show()


trainy, testy, _ = splitdata(s[:,[0,1]], mask=mask)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy)
y_pred = reg.predict(testx)
print("Mean squared error: %.2f" % mean_squared_error(testy, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(testy, y_pred))

plt.axis('equal')
plt.scatter(s[:,1], s[:,0], s=0.1)
y_pred = reg.predict(y)
plt.scatter(y_pred[:,1], y_pred[:,0], s=1)
plt.show()


# numbers
trainx, testx, mask = splitdata(y)

trainy, testy, _ = splitdata(b[:,[0,1]], mask=mask)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy, covs[mask][:,0,0]**0.5)
y_pred = reg.predict(testx)
print("Mean squared error: %.2f" % mean_squared_error(testy, y_pred),sum((abs(testy-y_pred))/abs(testy+1e-3)))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(testy, y_pred))


trainy, testy, _ = splitdata(s[:,[2,3]], mask=mask)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy)
y_pred = reg.predict(testx)
print("Mean squared error: %.2f" % mean_squared_error(testy, y_pred),sum((abs(testy-y_pred))/abs(testy+1e-3)))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(testy, y_pred))

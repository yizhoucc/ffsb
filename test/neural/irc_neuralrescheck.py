

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
import numpy as np
import pickle
from pathlib import Path
import os
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
os.chdir('../..')
print(os.getcwd())
import random

# load belief and neural data -------------------
resdir = 'Z:/neuraltest/res'
resdir = Path(resdir)


with open(resdir/'0202collapsemodelbelief', 'rb') as f:
    res = pickle.load(f)


y = res['y']
X = {k: res[k] for k in ['rad_vel', 'ang_vel', 'x_monk', 'y_monk']}
trial_idx = res['trial_idx']
beliefs = res['belief']
covs = res['cov']


# linear fit ----------------------------------
b = beliefs
s = np.vstack([v for v in X.values()])
s[np.isnan(s) == True] = 0
s = s.T
y = y.T
b = b.T


def splitdata(s, mask=None):
    n = len(s)
    if mask is not None:
        pass
    else:
        mask = np.array(random.sample(range(0, n), n//10))
    train = s[~mask]
    test = s[mask]
    return train, test, mask


trainx, testx, mask = splitdata(s)
trainy, testy, _ = splitdata(y, mask=mask)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy)
y_pred = reg.predict(testx)
print("Mean squared error: %.2f" % mean_squared_error(testy, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(testy, y_pred))

trainx, testx, mask = splitdata(y)
trainy, testy, _ = splitdata(s, mask=mask)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy)
y_pred = reg.predict(testx)
print("Mean squared error: %.2f" % mean_squared_error(testy, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(testy, y_pred))


trainx, testx, mask = splitdata(b)
trainy, testy, _ = splitdata(y, mask=mask)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy)
y_pred = reg.predict(testx)
print("Mean squared error: %.2f" % mean_squared_error(testy, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(testy, y_pred))

trainx, testx, mask = splitdata(b)
trainy, testy, _ = splitdata(y, mask=mask)
reg = linear_model.LinearRegression()
reg.fit(trainx, trainy)
y_pred = reg.predict(testx)
print("Mean squared error: %.2f" % mean_squared_error(testy, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(testy, y_pred))

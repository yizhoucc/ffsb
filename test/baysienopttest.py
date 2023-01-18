# # example of the test problem
# from math import sin
# from math import pi
# from numpy import arange
# from numpy import argmax
# from numpy.random import normal
# from matplotlib import pyplot
 
# # objective function
# def objective(x, noise=0.1):
# 	noise = normal(loc=0, scale=noise)
# 	return (x**2 * sin(5 * pi * x)**6.0) + noise
 



# example of bayesian optimization for a 1d function from scratch
from math import sin,cos
from math import pi
from numpy import arange, exp
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot

# objective function
def objective(x, noise=0.4):
	noise1 = normal(loc=-0, scale=noise)
	return ( cos(5 * pi * x)**6.0)+exp(-0.5*(x-0.5)**2/0.2)  +noise1


# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)

# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
	# calculate the best surrogate score found so far
	yhat, _ = surrogate(model, X)
	best = max(yhat)
	# calculate mean and stdev via surrogate function
	mu, std = surrogate(model, Xsamples)
	mu = mu[:, 0]
	# calculate the probability of improvement
	probs = norm.cdf((mu - best) / (std+1E-9))
	return probs

# optimize the acquisition function
def opt_acquisition(X, y, model):
	# random search, generate random samples
	Xsamples = random(100)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model)
	# locate the index of the largest scores
	ix = argmax(scores)
	return Xsamples[ix, 0]

# plot real observations vs surrogate function
def plot(X, y, model):
	# scatter plot of inputs and real objective function
	pyplot.scatter(X, y)
	# line plot of surrogate function across domain
	Xsamples = asarray(arange(0, 1, 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples, _ = surrogate(model, Xsamples)
	pyplot.plot(Xsamples, ysamples)
	# # show the plot
	# pyplot.show()

# plot ground truth
gX = arange(0, 1, 0.01)
# sample the domain without noise
gy = [objective(x, 0) for x in gX]
# sample the domain with noise
ynoise = [objective(x) for x in gX]
# find best result
ix = argmax(gy)
print('Optima: x=%.3f, y=%.3f' % (gX[ix], gy[ix]))
# plot the points with noise
pyplot.scatter(gX, ynoise,label='samples',color='tab:blue')
# plot the points without noise
pyplot.plot(gX, gy,label='ground truth',color='orange')
pyplot.legend()

# sample the domain sparsely with noise
X = random(100)
y = asarray([objective(x) for x in X])
# reshape into rows and cols
X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)
# plot before hand
plot(X, y, model)
# find best result
ix = argmax(y)
print('inital result before fit : x=%.3f, y=%.3f' % (X[ix], y[ix]))
# grid-based sample of the domain [0,1]
pyplot.title('before fit')
# show the plot
pyplot.show()


# perform the optimization process
for i in range(100):
	# select the next point to sample
	x = opt_acquisition(X, y, model)
	# sample the point
	actual = objective(x)
	# summarize the finding
	est, _ = surrogate(model, [[x]])
	# print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
	# add the data to the dataset
	X = vstack((X, [[x]]))
	y = vstack((y, [[actual]]))
	# update the model
	model.fit(X, y)

# plot all samples and the final surrogate function
plot(X, y, model)
# best result
ix = argmax(y)
print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
pyplot.plot(gX, gy,label='ground truth',color='orange')
pyplot.legend()
pyplot.title('after fit')
# show the plot
pyplot.show()


model

from plot_ult import *
from sklearn import linear_model





# simulation data
truew=3
trueb=30
x=torch.linspace(0,200,3000)
y=truew*x+trueb
y=torch.normal(y,60)
x,y=x.view(-1,1), y.view(-1,1)
plt.scatter(x,y)
plt.show()

# ground truth package.
linreg = linear_model.LinearRegression()
linreg.fit(x,y)
linreg.coef_
linreg.intercept_

# visualization of ground truth package
ax=plt.subplot()
thispred=linreg.predict(x)[:,0]
thistrue=np.array(y.view(-1))
predvstrue1(thispred,thistrue,ax)
vmin,vmax=limplot(ax)
ax.plot([vmin,vmax],[vmin,vmax],'k')
plt.xlabel('pred')
plt.ylabel('true')
plt.title('fit quanlity')
plt.show()

ax=plt.subplot()
predvstrue1(np.array(x.view(-1)),np.array(y.view(-1)),ax)
ax.plot(x, linreg.coef_*np.array(x)+linreg.intercept_,'k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('best fit line vs ground truthdata cloud')
plt.show()
print(linreg.coef_,linreg.intercept_)


# custom method -------------------
import torch
from torch.autograd import Variable

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, LBFGS, SGD
from torch.utils.data import Dataset, DataLoader


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

inputDim = 1        # takes variable 'x' 
outputDim = 1       # takes variable 'y'
learningRate = 0.01 
epochs = 10

model = linearRegression(inputDim, outputDim)
##### For GPU #######
if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.MSELoss() 
optimizer = LBFGS(model.parameters(), history_size=9, max_iter=4)
max_norm=22
losslog=[]
for epoch in range(epochs):
    running_loss = 0.0


    x_ = Variable(x, requires_grad=True)
    y_ = Variable(y)

    def closure():
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x_)

        # Compute loss
        loss = criterion(y_pred, y_)

        # Backward pass
        loss.backward()

        return loss

    # Update weights
    optimizer.step(closure)

    # Update the running loss
    loss = closure()

    print(loss)
    losslog.append(loss.clone().detach())
    # print('epoch {}, loss {}'.format(epoch, loss.item()))

plt.plot(losslog)
plt.show()

min(losslog)
criterion(torch.tensor(model.linear.weight*x+model.linear.bias), torch.tensor(y))
criterion(torch.tensor(linreg.coef_*np.array(x)+linreg.intercept_), torch.tensor(y))

with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(x.cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(x)).data.numpy()
    # print(predicted)

ax=plt.subplot()
predvstrue1(np.array(x.view(-1)),np.array(y.view(-1)),ax)
# plt.plot(x, y, 'go', label='True data', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, predicted, label='fitted custom model', color='k',alpha=1)
plt.plot(x, linreg.coef_*np.array(x)+linreg.intercept_, label='fitted package model', color='b',alpha=1)
plt.plot(x, truew*x+trueb, label='hidden truth', color='k',alpha=1)
quickleg(ax,bbox_to_anchor=(1,1.5))
plt.title('custom method has similar perforance as package')
plt.show()

print('custom model', model.linear.weight.clone().detach(),model.linear.bias.clone().detach())
print('groundtruth', truew, trueb)



# custom method with uncertainty (likelihood loss) -------------------

inputDim = 1        # takes variable 'x' 
outputDim = 1       # takes variable 'y'
learningRate = 0.01 
epochs = 10

model = linearRegression(inputDim, outputDim)

def likelihoodloss(pred, mus, covs, ndim=2):
    p=torch.distributions.multivariate_normal.MultivariateNormal(mus,covariance_matrix=covs)
    genloss=-torch.mean(p.log_prob(pred))
    # genloss=-torch.mean(torch.clip(p.log_prob(target),-10,3))
    return genloss


criterion = likelihoodloss
optimizer = LBFGS(model.parameters(), history_size=9, max_iter=4)
max_norm=22
losslog=[]
for epoch in range(epochs):
    running_loss = 0.0


    x_ = Variable(x, requires_grad=True)
    y_ = Variable(y)

    def closure():
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x_)

        # Compute loss
        loss = criterion(y_pred, y_, torch.ones((1,1)))

        # Backward pass
        loss.backward()

        return loss

    # Update weights
    optimizer.step(closure)

    # Update the running loss
    loss = closure()

    print(loss)
    losslog.append(loss.clone().detach())
    # print('epoch {}, loss {}'.format(epoch, loss.item()))

plt.plot(losslog)
plt.show()

# min(losslog)
# criterion(torch.tensor(model.linear.weight*x+model.linear.bias), torch.tensor(y))
# criterion(torch.tensor(linreg.coef_*np.array(x)+linreg.intercept_), torch.tensor(y))

with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(x.cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(x)).data.numpy()
    # print(predicted)

ax=plt.subplot()
predvstrue1(np.array(x.view(-1)),np.array(y.view(-1)),ax)
# plt.plot(x, y, 'go', label='True data', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, predicted, label='fitted custom model', color='k',alpha=1)
plt.plot(x, linreg.coef_*np.array(x)+linreg.intercept_, label='fitted package model', color='b',alpha=1)
plt.plot(x, truew*x+trueb, label='hidden truth', color='k',alpha=1)
quickleg(ax,bbox_to_anchor=(1,1.5))
plt.title('custom method has similar perforance as package')
plt.show()
print('package model', model.linear.weight.clone().detach(),model.linear.bias.clone().detach())
print('custom model', linreg.coef_,linreg.intercept_)
print('groundtruth', truew, trueb)





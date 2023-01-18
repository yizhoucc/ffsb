# testing hessian
import torch
from torch.autograd.functional import hessian
import matplotlib.pyplot as plt
import np

#----------- likelihood----------------------------


# create a cov
# the cov is narrower at x, wider at y. positively correrlated.
cov=torch.tensor([[0.3,0.5],
                [0.5,2]])

# if x at 0,0, the hessian is -1*cov!
h1=-1*hessian(lambda x:torch.exp(-0.5*(x.t()@cov@x)),torch.ones(2)*0)
# if x at 0,1, we should be more certain at x than y.
h2=-1*hessian(lambda x:torch.exp(-0.5*(x.t()@cov@x)),torch.tensor([0,0.7]))

# non perfect inference
fig=plt.figure(figsize=(5,5))
ax=fig.add_subplot(111)
plot_cov_ellipse(cov,[0,0],nstd=1,ax=ax)
plot_cov_ellipse(cov,[0,0],nstd=2,ax=ax)
plot_cov_ellipse(cov,[0,0],nstd=3,ax=ax)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.scatter(0,1,color='red')
# plot hessian 
plot_cov_ellipse(h2,[0,1],nstd=1,ax=ax,color='orange')
# plot cov by inverse hessian
plot_cov_ellipse(torch.inverse(h2),[0,1],nstd=0.3,ax=ax,color='red')
ax.scatter(6,6,color='red',label='inferrence uncertainty')
ax.scatter(6,6,color='tab:blue',label='orignal distribution')
ax.scatter(6,6,color='orange',label='heissian')
ax.legend()


# perfect inference
fig=plt.figure(figsize=(5,5))
ax=fig.add_subplot(111)
plot_cov_ellipse(cov,[0,0],nstd=1,ax=ax)
plot_cov_ellipse(cov,[0,0],nstd=2,ax=ax)
plot_cov_ellipse(cov,[0,0],nstd=3,ax=ax)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
# plot cov by inverse hessian
plot_cov_ellipse(torch.inverse(h1),[0,0],nstd=0.3,ax=ax,color='red')
ax.scatter(6,6,color='red',label='perfect inferrence uncertainty')
ax.scatter(6,6,color='tab:blue',label='orignal distribution')
# ax.scatter(6,6,color='orange',label='heissian')
ax.legend()

#-----------log likelihood----------------------------

# if x at 0,0, the hessian is -1*cov!
invcov=torch.inverse(cov)
h1=-1*hessian(lambda x:(-0.5*(x.t()@invcov@x)),torch.ones(2)*1e-8)
# if x at 0,1, we should be more certain at x than y.
h2=-1*hessian(lambda x:(-0.5*(x.t()@invcov@x)),torch.tensor([-100,-60.]))

# non perfect inference
fig=plt.figure(figsize=(5,5))
ax=fig.add_subplot(111)
# plot_cov_ellipse(cov,[0,0],nstd=1,ax=ax)
# plot_cov_ellipse(cov,[0,0],nstd=2,ax=ax)
# plot_cov_ellipse(cov,[0,0],nstd=3,ax=ax)
m = np.array([[0.],[0.]])  # defining the mean of the Gaussian (mX = 0.2, mY=0.6)
cov_inv = np.linalg.inv(cov)  # inverse of covariance matrix
cov_det = np.linalg.det(cov)  # determinant of covariance matrix
# Plotting
x = np.linspace(-9, 9)
y = np.linspace(-9, 9)
X,Y = np.meshgrid(x,y)
coe = 1.0 / ((2 * np.pi)**2 * cov_det)**0.5
Z =  (-0.5 * (cov_inv[0,0]*(X-m[0])**2 + (cov_inv[0,1] + cov_inv[1,0])*(X-m[0])*(Y-m[1]) + cov_inv[1,1]*(Y-m[1])**2))
ax.contourf(X,Y,Z)

ax.set_xlim(-9,9)
ax.set_ylim(-9,9)

# plot hessian 
# plot_cov_ellipse(h2,[0,1],nstd=1,ax=ax,color='orange')
# plot cov by inverse hessian
plot_cov_ellipse(torch.inverse(h2),[-6,-6.],nstd=1,ax=ax,color='red')
plot_cov_ellipse(torch.inverse(h1),[0,0],nstd=1,ax=ax,color='orange')
ax.scatter(33,33,color='red',label='inferrence uncertainty')
ax.scatter(33,33,color='tab:blue',label='orignal distribution')
ax.scatter(33,33,color='orange',label='perfect inferrence uncertainty')
ax.legend()
ax.set_title('log likelihood surface',fontsize=16)


# perfect inference
fig=plt.figure(figsize=(5,5))
ax=fig.add_subplot(111)
# plot_cov_ellipse(cov,[0,0],nstd=1,ax=ax)
# plot_cov_ellipse(cov,[0,0],nstd=2,ax=ax)
# plot_cov_ellipse(cov,[0,0],nstd=3,ax=ax)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
# plot cov by inverse hessian
plot_cov_ellipse(torch.inverse(h1),[0,0],nstd=0.3,ax=ax,color='red')
ax.scatter(6,6,color='red',label='perfect inferrence uncertainty')
ax.scatter(6,6,color='tab:blue',label='orignal distribution')
# ax.scatter(6,6,color='orange',label='heissian')
ax.legend()








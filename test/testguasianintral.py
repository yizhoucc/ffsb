
from matplotlib import pyplot as plt
import torch
from numpy import pi
import numpy as np



# plot uncertainty
uncertainty=torch.tensor([[0.2,0],[0,0.1]])*0.01
m = np.array([[0.1],[-0.1]])*0.9   # the origin
r=0.13


fig=plt.figure(figsize=(5,5))
ax=fig.add_subplot(111)
cov_inv = np.linalg.inv(uncertainty)  # inverse of covariance matrix
cov_det = np.linalg.det(uncertainty)  # determinant of covariance matrix
x = np.linspace(-0.3, 0.3,55)
y = np.linspace(-0.3, 0.3,55)
X,Y = np.meshgrid(x,y)
coe = 1.0 / ((2 * np.pi)**2 * cov_det)**0.5
Z = np.exp(-0.5 * (cov_inv[0,0]*X**2 + (cov_inv[0,1] + cov_inv[1,0])*X*Y + cov_inv[1,1]*Y**2))*0.6/55
# for row in (range(Z.shape[0])):
#     for col in (range(Z.shape[1])):
#         if Z[row,col]<0.1:
#             Z[row,col]=0.

ax.set_xlim(-0.3,0.3)
ax.set_ylim(-0.3,0.3)
c=ax.contourf(X,Y,Z)
plt.colorbar(c)



# plot goal
circle1 = plt.Circle(m,r, color='r',fill=False)
ax.add_patch(circle1)

# calculate the prob and make it title
reward=torch.tensor([0.])
rew_std = 0.065
mu = torch.Tensor(m)
R = torch.eye(2)*0.065**2 
P = torch.tensor(uncertainty)
S = R+P
alpha = -0.5 * mu.t() @ S.inverse() @ mu
reward = torch.exp(alpha) /2 / pi /torch.sqrt(S.det())
# normalization -> to make max reward as 1
mu_zero = torch.zeros(1,2)
alpha_zero = -0.5 * mu_zero @ R.inverse() @ mu_zero.t()
reward_zero = torch.exp(alpha_zero) /2 / pi /torch.sqrt(R.det())
reward = reward/reward_zero

# # sampling method:
# reward2=torch.tensor([0.])
# for row in (range(X.shape[0])):
#     for col in (range(X.shape[1])):
#         x=X[row,col]
#         y=Y[row,col]
#         if (x-m[0])**2+(y-m[1])**2<0.13**2:
#             reward2+=Z[row,col]
# + '  {0:.2f}'.format(reward2.item())


ax.set_title('reward prob:{0:.2f}'.format(reward.item()),fontsize=14)

plt.show()


Zin=Z.copy()+0.001

for row in (range(X.shape[0])):
    for col in (range(X.shape[1])):
        x=X[row,col]
        y=Y[row,col]
        if (x-m[0])**2+(y-m[1])**2>0.13**2:
            Zin[row,col]=0.
        else:
            Z[row,col]=0.

fig=plt.figure(figsize=(5,5))
output = fig.add_subplot(111, projection = '3d') 
output.plot_surface(X, Y, Z, rstride = 2, cstride = 1, cmap = plt.cm.Blues_r,alpha=0.9)
output.plot_surface(X, Y, Zin, rstride = 2, cstride = 1,cmap=plt.cm.Reds)
output.set_xlabel('x')                        
output.set_ylabel('y')
output.set_zlabel('prob')
plt.show()


%gui qt
import os
os.environ["QT_API"] = "pyqt5"
from mayavi import mlab


def v2_mayavi(transparency):
    from mayavi import mlab

    fig = mlab.figure(size=(999,999))
    X, Y = np.mgrid[-0.3:0.3:0.01, -0.3:0.3:0.01]
    # Z = np.exp(-0.5 * (cov_inv[0,0]*X**2 + (cov_inv[0,1] + cov_inv[1,0])*X*Y + cov_inv[1,1]*Y**2))*0.6/55
    # s = surf(X, Y, Z,colormap='Blues')
    ax_ranges = [-0.3, 0.3, -0.3, 0.3, 0, 0.02]
    ax_scale = [1.0, 1.0, 50.0]
    ax_extent = ax_ranges * np.repeat(ax_scale, 2)

    surf3 = mlab.surf(X, Y, Z, colormap='Blues')
    surf4 = mlab.surf(X, Y, Zin, colormap='Oranges')
    surf3.actor.actor.scale = ax_scale
    surf4.actor.actor.scale = ax_scale
    mlab.view(1, 1, 1, [-1.5, -1.6, -0.3])
    mlab.outline(surf3, color=(.7, .7, .7), extent=ax_extent)
    mlab.axes(surf3, color=(.7, .7, .7), extent=ax_extent,
              ranges=ax_ranges,
              xlabel='x', ylabel='y', zlabel='z')

    if transparency:
        surf3.actor.property.opacity = 0.5
        surf4.actor.property.opacity = 0.5
        fig.scene.renderer.use_depth_peeling = 1

v2_mayavi(False)




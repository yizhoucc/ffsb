# for visualizing group inverse and sub inverse.
'''
if sub inverse produce ridges on logll surface, then group inverse may not be able to find the correct solution.
'''
import numpy as np
import matplotlib.pyplot as plt

from plot_ult import quickallspine


# produce some sub inverse ellispse

paramspace=[0,1]
subjectrange=[0.3,0.7]
covanglerange=[0.6,2]
covratiorange=[3,10]
nsub=5
reso=100


fig=plt.figure()
ax=fig.add_subplot(121)
ax.set_xlim(paramspace)
ax.set_ylim(paramspace)
ax.axis('equal')
X,Y=np.meshgrid(np.linspace(paramspace[0],paramspace[1], reso),np.linspace(paramspace[0],paramspace[1], reso))
pos = np.dstack((X, Y))

groudtruth=[]
for isub in range(nsub):
    mu=np.random.uniform(low=subjectrange[0],high=subjectrange[1],size=(2,))
    covangle=np.random.uniform(low=covanglerange[0],high=covanglerange[1])
    covratio=np.random.uniform(low=covratiorange[0],high=covratiorange[1])
    cov=np.diag([1,covratio**2])/100
    c, s = np.cos(covangle), np.sin(covangle)
    rotation = np.array(((c, -s), (s, c)))
    cov=rotation@cov@rotation.T
    surface=multivariate_normal(mu,cov)
    subZ=surface.pdf(pos)
    groudtruth.append(subZ)
    im=ax.contourf(X,Y,subZ, alpha=0.3)
plt.colorbar(im, label='psedo likelihood',alpha=1)
    # plot_cov_ellipse(cov, pos=mu, nstd=1, color=None, ax=ax,edgecolor=None,alpha=0.1)
ax.set_title('subject inverse')
quickallspine(ax)
ax.set_xlabel('parameter')
ax.set_ylabel('parameter')

groupZ=np.sum(groudtruth, axis=0)
ax=fig.add_subplot(122, sharey=ax)
im=ax.contourf(X,Y,groupZ, alpha=1)
ax.set_title('group inverse')
quickallspine(ax)
ax.set_xlabel('parameter')
ax.set_ylabel('parameter')
ax.axis('equal')
plt.colorbar(im, label='psedo likelihood')

plt.tight_layout()



from torch import float32
from plot_ult import *
import numpy as np
from matplotlib.pyplot import axes, bar, plot



# trial structure
ind=np.random.randint(0,len(actions))
vs=actions[ind][:,0]
ws=actions[ind][:,1]
ts=np.arange(len(vs))*0.1

with initiate_plot(4, 2, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(211)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-1,1)
    ax.fill_betweenx([-1,2],-0.05,0,alpha=0.3,color='blue')
    ax.set_ylabel('forward v [a.u.]', fontsize=9)
    ax.plot(ts,vs)
    
    ax = fig.add_subplot(212)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(ts,ws)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel('time [s]')
    ax.set_ylabel('angular w [a.u.]', fontsize=9)
    ax.set_ylim(-1,1)
    ax.fill_betweenx([-1,2],-0.05,0,alpha=0.3,color='blue')

# fig1, all overhead
print('loading data')
datapath=Path("Z:\\bruno_normal\\packed")
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
# print('process data')
# states, actions, tasks=monkey_data_downsampled(df,factor=0.0025)
# print('done process data')

overheaddf_tar(df[:1000])
overheaddf_path(df,list(range(1000)))

# fig2, training vs time
agentcommon='re1re1re1repaper_3_199_199_'
log=[]
for i in range(1,100,4):
    # load the agent check points
    agentname=agentcommon+str(i)
    thisagent=TD3.load('trained_agent/'+agentname).actor.mu.cpu()
    with suppress():
        # eval
        philist=[
    torch.tensor([[0.5],
            [pi/2],
            [0.5],
            [0.5],
            [0.5],
            [0.5],
            [0.13],
            [0.5],
            [0.5],
            [0.5],
            [0.5],
    ]),torch.tensor([[0.5],
            [pi/2],
            [0.5],
            [0.5],
            [0.5],
            [0.5],
            [0.13],
            [0.9],
            [0.9],
            [0.9],
            [0.9],
    ]),torch.tensor([[0.5],
            [pi/2],
            [0.9],
            [0.9],
            [0.9],
            [0.9],
            [0.13],
            [0.5],
            [0.5],
            [0.5],
            [0.5],
    ])
    ]
        tasklist=[[0.9,0.1],[0.7,0.4],[0.7,-0.4],[0.6,0]]
        thislog=[]
        for p in philist:
            for t in tasklist:
                env.reset(goal_position=t, phi=p, theta=p)
                done=False
                while not done:
                    _,_,done,_=env.step(thisagent(env.decision_info))
                thislog.append((env.trial_sum_reward-env.trial_sum_cost))
    log.append([np.mean(thislog),np.std(thislog)])
    print(agentname,log[-1] )
with initiate_plot(3, 2, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(211)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot([l[0] for l in log])


# fig 3
#  logll vs gen
with open(data_path/'longonlypacked_schro_pert', 'rb') as f:
    slog = pickle.load(f)
log=slog
optimizer=log[-1][0]
res=[l[2] for l in log]
loglls=[]
for r in res:
    genloglls=[]
    for point in r:
        genloglls.append(point[1])
    loglls.append(np.mean(genloglls))
# gen=[[i]*optimizer.population_size for i in range(optimizer.generation)]
gen=list(range(len(res)))
gen=torch.tensor(gen).flatten()
loglls=torch.tensor(loglls).flatten()

with initiate_plot(3, 2, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(211)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(gen,loglls,alpha=0.8)
    plt.xlabel('generations')
    plt.ylabel('- log likelihood')

# zoom in of last 1/4
with initiate_plot(3, 2, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(211)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(gen[-len(gen)//2:],loglls[-len(gen)//2:],alpha=0.8)
    plt.xlabel('generations')
    plt.ylabel('- log likelihood')

# slice pc
print('loading data')
datapath=Path("Z:\\schro_pert\\packed_schro_pert")
with open(datapath,'rb') as f:
    df = pickle.load(f)
df=datawash(df)
df=df[df.category=='normal']
df=df[df.target_r>180]
print('process data')
states, actions, tasks=monkey_data_downsampled(df[110:120],factor=0.0025)
print('done process data')

@ray.remote
def getlogll(x):
    with torch.no_grad():
        return  monkeyloss_(agent, actions, tasks, phi, torch.tensor(x).t(), env, action_var=0.01,num_iteration=1, states=states, samples=5,gpu=False).item()

with open(data_path/'longonlypacked_schro_pert', 'rb') as f:
    slog = pickle.load(f)
log=slog
res=[l[2] for l in log]
allsamples=[]
alltheta=[]
loglls=[]
for r in res:
    for point in r:
        alltheta.append(torch.tensor(point[0]))
        loglls.append(torch.tensor(point[1]))
        allsamples.append([point[1],point[0]])
alltheta=torch.stack(alltheta)
logllsall=torch.tensor(loglls).flatten()
allsamples.sort(key=lambda x: x[0])
allthetameans=np.array([l[0]._mean for l in log])
alltheta=alltheta[50:]
logllsall=logllsall[50:]
mu=np.mean(np.asfarray(alltheta),0).astype('float32')
score, evectors, evals = pca(np.asfarray(alltheta))
x=score[:,0] # pc1
y=score[:,1] # pc2
z=logllsall
npixel=9

finalcov=log[-1][0]._C
realfinalcov=np.cov(np.array([l[0] for l in log[-1][2]]).T)

finaltheta=log[-1][0]._mean
pcfinaltheta=((finaltheta.reshape(-1)-mu)@evectors).astype('float32')
finalcovpc=(evectors.T@finalcov@evectors).astype('float32')
realfinalcovpc=(evectors.T@realfinalcov@evectors).astype('float32')
pccov=evectors.T@(np.cov(alltheta.T))@evectors
allthetameanspc=((allthetameans-mu)@evectors).astype('float32')
with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(111)
    plot_cov_ellipse(pccov[:2,:2], pcfinaltheta[:2], alpha_factor=1,nstd=1,ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    s = ax.scatter(x,y, c=z,alpha=0.5,edgecolors='None', cmap='jet')
    ax.set_xlabel('projected parameters')
    ax.set_ylabel('projected parameters')
    c = fig.colorbar(s, ax=ax)
    ax.locator_params(nbins=3, axis='y')
    ax.locator_params(nbins=3, axis='x')
    # c.clim(min(np.log(loglls)), max(np.log(loglls))) 
    c.set_label('- log likelihood')
    # c.set_ticks([min((loglls)),max((loglls))])
    c.ax.locator_params(nbins=4)
    ax.set_xlim(xlow,xhigh)
    ax.set_ylim(ylow,yhigh)
    ax.plot(allthetameanspc[:,0],allthetameanspc[:,1])

xlow,xhigh=pcfinaltheta[0]-0.5,pcfinaltheta[0]+0.5
ylow,yhigh=pcfinaltheta[1]-0.5,pcfinaltheta[1]+0.5
# xlow,xhigh=ax.get_xlim()
xrange=np.linspace(xlow,xhigh,npixel)
# ylow,yhigh=ax.get_ylim()
yrange=np.linspace(ylow,yhigh,npixel)
X,Y=np.meshgrid(xrange,yrange)

background_data=np.zeros((npixel,npixel))
for i,u in enumerate(xrange):
    for j,v in enumerate(yrange):
        score=np.array([u,v])
        reconstructed_theta=score@evectors.transpose()[:2,:]+mu
        reconstructed_theta=reconstructed_theta.clip(1e-4,999)
        background_data[i,j]=getlogll(reconstructed_theta.reshape(1,-1).astype('float32'))

with initiate_plot(3, 3.5, 300) as fig, warnings.catch_warnings():
    warnings.simplefilter('ignore')
    ax = fig.add_subplot(111)
    plot_cov_ellipse(pccov[:2,:2], pcfinaltheta[:2], alpha_factor=1,nstd=1,ax=ax)
    im=ax.contourf(X,Y,background_data,cmap='jet')
    # c = add_colorbar(im)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('projected parameters')
    ax.set_ylabel('projected parameters')
    c = fig.colorbar(im, ax=ax)
    ax.locator_params(nbins=3, axis='y')
    ax.locator_params(nbins=3, axis='x')
    # c.clim(min(np.log(loglls)), max(np.log(loglls))) 
    c.set_label('- log likelihood')
    # c.set_ticks([min((loglls)),max((loglls))])
    c.ax.locator_params(nbins=4)
    ax.set_xlim(xlow,xhigh)
    ax.set_ylim(ylow,yhigh)
    ax.plot(allthetameanspc[:,0],allthetameanspc[:,1])




# vary eig vector by eig value
states, actions, tasks=monkey_data_downsampled(df[df.perturb_start_time.isnull()],factor=0.0025)
finalcov
phi=torch.tensor([[0.5000],
        [1.57],
        [0.01],
        [0.01],
        [0.01],
        [0.01],
        [0.13],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
])

finaltheta=torch.tensor(log[-1][0]._mean).view(-1,1)
# finaltheta=torch.cat([finaltheta[:6],finaltheta[-4:]])
finalcov=torch.tensor(log[-1][0]._C)
# finalcov = finalcov[torch.arange(finalcov.size(0))!=6] 
# finalcov = finalcov[:,torch.arange(finalcov.size(1))!=6] 
ev, evector=torch.eig(torch.tensor(finalcov),eigenvectors=True)
ev=ev[:,0]
ev,esortinds=ev.sort(descending=False)
evector=evector[:,esortinds]

firstevector=evector[:,0].float().view(-1,1)
firstev=ev[0]


ind=np.random.randint(0,len(tasks))
# ind=461 of schro non pert data
theta_init=finaltheta-firstevector*firstev*2
theta_final=finaltheta+firstevector*firstev*2
env.debug=True

diagnose_plot_theta(agent, env, phi, theta_init, theta_final,5,etask=tasks[ind],initv=actions[ind][0][0],initw=actions[ind][0][1],mkactions=actions[ind])






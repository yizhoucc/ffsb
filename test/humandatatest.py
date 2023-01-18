

import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from monkey_functions import down_sampling, down_sampling_
from numpy import pi, save
import pickle
import torch
from FireflyEnv import ffacc_real
from Config import Config
import mat73

arg = Config()
env=ffacc_real.FireFlyPaper(arg)
env.debug=True
env.terminal_vel=0.0 # prevent the asseertion of stop error\


# load the mat
def loadmat(filename,key='subjects'):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    # def _check_keys(data):
    #     '''
    #     checks if entries in dictionary are mat-objects. If yes
    #     todict is called to change them to nested dictionaries
    #     '''
    #     for key in data:
    #         print(key,type(data[key]))
    #         if isinstance(data[key], spio.matlab.mio5_params.mat_struct):
    #             data[key] = _todict(data[key])
    #     return data

    def _check_keys(data):
        res=[]
        for d in data:
            if isinstance(d, spio.matlab.mio5_params.mat_struct):
                d = _todict(d)
            res.append(d)
        return res

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    try:
        data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    except:
        data=mat73.loadmat(filename)
    data=data[key]
    data=_check_keys(data)
    return data


# filename='Z:/human/wofbv6.mat'
# data=loadmat(filename)

# # seperate into two groups
# hdata,adata=[],[]
# for d in data:
#     if d['name'][0]=='A':
#         adata.append(d)
#     else:
#         hdata.append(d)
# print('we have these number of health and autiusm subjects',len(hdata),len(adata))


# process the data into inverse ready format ----------------------------


# # test 
# sub=hdata[0]

# # tasks 
# worldscale=200
# r=np.array(sub['targ']['r'])/worldscale
# theta=np.array(sub['targ']['theta'])
# tasks=np.vstack([r*np.cos(theta),r*np.sin(theta)]).T
# # target positions
# plt.scatter(r*np.cos(theta),r*np.sin(theta))
# plt.gca().set_aspect('equal', adjustable='box')
# # stop position (check)
# r=np.array(sub['resp']['r'])/worldscale
# theta=np.array(sub['resp']['theta'])
# tasks=np.vstack([r*np.cos(theta),r*np.sin(theta)]).T
# tasks=tasks.astype('float32')
# # target positions
# plt.scatter(r*np.cos(theta),r*np.sin(theta))
# plt.gca().set_aspect('equal', adjustable='box')

# # actions
# ntrial=len(sub['vel']['v'])
# actions=[]
# for i in range(ntrial):
#     epactions=np.vstack([down_sampling(sub['vel']['v'][i],orignal_dt=0.0012*8)/worldscale,down_sampling(sub['vel']['w'][i],orignal_dt=0.0012*8)/90]).T
#     actions.append(torch.tensor(epactions).float())

# # check 
# plt.plot(actions[55])
# curmax=0.
# for a in actions:
#     curmax=max(curmax,np.max(a[:,1]))


# states
def run_trial(env,given_action,pert=None,
            phi=torch.tensor([[1],   
                [pi/2],   
                [1e-8],   
                [1e-8],   
                [0.0],   
                [0.0],   
                [0.13],   
                [0.5],   
                [0.5],   
                [0.5],   
                [0.5]])):

    def _collect():
        epstates.append(env.s)
    # saves
    epstates=[]
    env.reset(phi=phi)
    t=0
    while t<len(given_action):
        env.step(given_action[t]) 
        _collect()
        t+=1
    env.step(given_action[-1]) 
    _collect()
    return torch.stack(epstates, axis=0)[1:,:,0]

# env.terminal_vel=0.0 # prevent the asseertion of stop error
# states=[]
# for a in actions:
#     epstates=run_trial(env,a)
#     states.append(epstates)

# # check if the states align with stop and xy
# ind=np.random.randint(0,len(actions))
# plt.plot(states[ind][:,0],states[ind][:,1],label='model reconstruction')
# plt.plot(np.asarray(sub['traj']['yt'][ind])/200,np.asarray(sub['traj']['xt'][ind])/200,label='true')
# r=np.array(sub['resp']['r'])[ind]/worldscale
# theta=np.array(sub['resp']['theta'])[ind]
# plt.scatter(r*np.cos(theta),r*np.sin(theta),label='actual stop')
# r=np.array(sub['targ']['r'])[ind]/worldscale
# theta=np.array(sub['targ']['theta'])[ind]
# plt.scatter(r*np.cos(theta),r*np.sin(theta),label='target')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.xlabel('world x, 2m')
# plt.ylabel('world y, 2m')
# plt.legend()
# plt.show()

# plt.plot(actions[ind])
# plt.plot(states[ind][:,2])
# head=[0]
# for wctrl in actions[ind][:,1]:
#     head.append(head[-1]+0.1*pi/2*wctrl)
# plt.plot(head)    

# np.sum(actions[ind][:,0])*0.1
# np.sum(sub['vel']['v'][ind])*(0.0012*8)/200

# np.sum(actions[ind][:,1])*0.1
# np.sum(sub['vel']['w'][ind])*(0.0012*8)/90


# import neo
# filename=['Z:/human/woFB/A1_woFB_25-Oct-2017_9-52_smr.smr']

# for idx, file_name in enumerate(filename):
#     seg_reader = neo.io.Spike2IO(filename=file_name).read_segment()



# len(states[0])
# len(actions[0])

# # save the processed human data
# packed=(states, actions, tasks)
# import pickle
# with open('humantest', 'wb') as handle:
#     pickle.dump(packed, handle, protocol=pickle.HIGHEST_PROTOCOL)





# # tasks 
# worldscale=200
# r=np.array(sub['targ']['r'])/worldscale
# theta=np.array(sub['targ']['theta'])
# tasks=np.vstack([r*np.cos(theta),r*np.sin(theta)]).T
# # target positions
# plt.scatter(r*np.cos(theta),r*np.sin(theta))
# plt.gca().set_aspect('equal', adjustable='box')
# # stop position (check)
# r=np.array(sub['resp']['r'])/worldscale
# theta=np.array(sub['resp']['theta'])
# tasks=np.vstack([r*np.cos(theta),r*np.sin(theta)]).T
# tasks=tasks.astype('float32')
# # target positions
# plt.scatter(r*np.cos(theta),r*np.sin(theta))
# plt.gca().set_aspect('equal', adjustable='box')


# # actions
# ntrial=len(sub['vel']['v'])
# actions=[]
# for i in range(ntrial):
#     epactions=np.vstack([down_sampling(sub['vel']['v'][i],orignal_dt=0.0012*8)/worldscale,down_sampling(sub['vel']['w'][i],orignal_dt=0.0012*8)/90]).T
#     actions.append(torch.tensor(epactions).float())

# # check 
# plt.plot(actions[55])


# curmax=0.
# for a in actions:
#     curmax=max(curmax,np.max(a[:,1]))





# states

# env.terminal_vel=0.0 # prevent the asseertion of stop error
# states=[]
# for a in actions:
#     epstates=run_trial(env,a)
#     states.append(epstates)


# offical process and save asd -----------------------------------------------------------
def process_human_data(subjectls, env):
    taskall,actionall,stateall=[],[],[]
    for sub in subjectls:

        # tasks 
        worldscale=200
        r=np.array(sub['targ']['r'])/worldscale
        theta=np.array(sub['targ']['theta'])
        tasks=np.vstack([r*np.cos(theta),r*np.sin(theta)]).T
        if taskall!=[]:
            taskall=np.vstack([taskall,tasks])
        else:
            taskall=tasks

        # actions
        ntrial=len(sub['vel']['v'])
        actions=[]
        for i in range(ntrial):
            epactions=np.vstack([down_sampling(sub['vel']['v'][i],orignal_dt=0.0012*8)/worldscale,down_sampling(sub['vel']['w'][i],orignal_dt=0.0012*8)/90]).T
            actions.append(torch.tensor(epactions).float())
        actionall+=actions

        # states
        def run_trial(env,given_action,pert=None,
                    phi=torch.tensor([[1],   
                        [pi/2],   
                        [1e-8],   
                        [1e-8],   
                        [0.0],   
                        [0.0],   
                        [0.13],   
                        [0.5],   
                        [0.5],   
                        [0.5],   
                        [0.5]])):

            def _collect():
                epstates.append(env.s)
            # saves
            epstates=[]
            env.reset(phi=phi)
            t=0
            while t<len(given_action):
                env.step(given_action[t]) 
                _collect()
                t+=1
            env.step(given_action[-1]) 
            _collect()
            return torch.stack(epstates, axis=0)[1:,:,0]
        env.terminal_vel=0.0 # prevent the asseertion of stop error
        states=[]
        for a in actions:
            epstates=run_trial(env,a)
            states.append(epstates)
        stateall+=states
    return stateall,actionall,taskall


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

# make human data and save
states, actions, tasks=process_human_data(hdata, env)
# save the processed human data
packed=(states, actions, tasks)

# save as one
# with open('Z:/human/hgroup', 'wb') as handle:
#     pickle.dump(packed, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save per sub
for i in range(len(adata)):
    savename="asub{}".format(str(i))
    states, actions, tasks=process_human_data([adata[i]], env)
    packed=(states, actions, tasks)
    with open('Z:/human/{}'.format(savename), 'wb') as handle:
        pickle.dump(packed, handle, protocol=pickle.HIGHEST_PROTOCOL)


[len(eachsub['targ']['r']) for eachsub in hdata]
len(states)



with open('Z:/human/hgroup', 'rb') as f:
    df = pickle.load(f)



# offical process and save turn off data ---------------------------------------------------------



# env.terminal_vel=0.0 # prevent the asseertion of stop error
# states=[]
# for a in actions:
#     epstates=run_trial(env,a)
#     states.append(epstates)

# # check if the states align with stop and xy
# ind=np.random.randint(0,len(actions))
# plt.plot(states[ind][:,0],states[ind][:,1],label='model reconstruction')
# plt.plot(np.asarray(sub['traj']['yt'][ind])/200,np.asarray(sub['traj']['xt'][ind])/200,label='true')
# r=np.array(sub['resp']['r'])[ind]/worldscale
# theta=np.array(sub['resp']['theta'])[ind]
# plt.scatter(r*np.cos(theta),r*np.sin(theta),label='actual stop')
# r=np.array(sub['targ']['r'])[ind]/worldscale
# theta=np.array(sub['targ']['theta'])[ind]
# plt.scatter(r*np.cos(theta),r*np.sin(theta),label='target')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.xlabel('world x, 2m')
# plt.ylabel('world y, 2m')
# plt.legend()
# plt.show()

# plt.plot(actions[ind])
# plt.plot(states[ind][:,2])
# head=[0]
# for wctrl in actions[ind][:,1]:
#     head.append(head[-1]+0.1*pi/2*wctrl)
# plt.plot(head)    

# np.sum(actions[ind][:,0])*0.1
# np.sum(sub['vel']['v'][ind])*(0.0012*8)/200

# np.sum(actions[ind][:,1])*0.1
# np.sum(sub['vel']['w'][ind])*(0.0012*8)/90


# import neo
# filename=['Z:/human/woFB/A1_woFB_25-Oct-2017_9-52_smr.smr']

# for idx, file_name in enumerate(filename):
#     seg_reader = neo.io.Spike2IO(filename=file_name).read_segment()



# len(states[0])
# len(actions[0])

# # save the processed human data
# packed=(states, actions, tasks)
# import pickle
# with open('humantest', 'wb') as handle:
#     pickle.dump(packed, handle, protocol=pickle.HIGHEST_PROTOCOL)





# # tasks 
# worldscale=200
# r=np.array(sub['targ']['r'])/worldscale
# theta=np.array(sub['targ']['theta'])
# tasks=np.vstack([r*np.cos(theta),r*np.sin(theta)]).T
# # target positions
# plt.scatter(r*np.cos(theta),r*np.sin(theta))
# plt.gca().set_aspect('equal', adjustable='box')
# # stop position (check)
# r=np.array(sub['resp']['r'])/worldscale
# theta=np.array(sub['resp']['theta'])
# tasks=np.vstack([r*np.cos(theta),r*np.sin(theta)]).T
# tasks=tasks.astype('float32')
# # target positions
# plt.scatter(r*np.cos(theta),r*np.sin(theta))
# plt.gca().set_aspect('equal', adjustable='box')


# # actions
# ntrial=len(sub['vel']['v'])
# actions=[]
# for i in range(ntrial):
#     epactions=np.vstack([down_sampling(sub['vel']['v'][i],orignal_dt=0.0012*8)/worldscale,down_sampling(sub['vel']['w'][i],orignal_dt=0.0012*8)/90]).T
#     actions.append(torch.tensor(epactions).float())

# # check 
# plt.plot(actions[55])


# curmax=0.
# for a in actions:
#     curmax=max(curmax,np.max(a[:,1]))





# states

# env.terminal_vel=0.0 # prevent the asseertion of stop error
# states=[]
# for a in actions:
#     epstates=run_trial(env,a)
#     states.append(epstates)



# ---------- turn off obs data ------------------------
# load data -----------------------------------------
filename='/data/human/turnoffdata/JP.mat'
data=mat73.loadmat(filename)


# play with data and see -----------------------------------------
data.keys()
data['data'].keys()
data['dis_data'].keys()

# the on lenth for each trial
plt.plot(data['dis_data']['stimDur'])


# the targets
cond=np.where(data['data']['on_off']==1)[0] # when on
plt.scatter(data['data']['FFx'][cond],data['data']['FFz'][cond],color='r',alpha=0.3)
plt.xlabel('data ffx')
plt.ylabel('data ffz')
plt.axis('equal')

# target with paths
cond=np.where(data['data']['on_off']==1)[0]
plt.scatter(data['data']['FFx'][cond],data['data']['FFz'][cond],color='r',alpha=0.3)
plt.plot(data['data']['posX'][:],data['data']['posZ'][:],alpha=0.3)
plt.xlabel('data ffx(red) and posX(blue)')
plt.ylabel('data ffz(red) and posZ(blue)')
plt.axis('equal')



# extract -----------------------------------------

# sample rate? use 1/100 100hz for now
plt.plot((data['data']['trial_time']))
plt.plot(np.diff(data['data']['trial_time']))
np.mean(np.diff(data['data']['trial_time']))

plt.plot(data['data']['end_trial']-data['data']['start_trial'])
plt.plot(data['data']['stop_trial']-data['data']['start_trial'])

plt.plot(data['data']['start_trial'][1:]-data['data']['end_trial'][:-1])

# get the index to split trials
trialx, trialz=[],[] # states
trialv, trialw=[],[] # actions
for start, end in zip(data['data']['start_trial'].astype(int),data['data']['end_trial'].astype(int)):
    # -1 because reset at the end
    trialx.append(data['data']['posX'][start:end-1])
    trialz.append(data['data']['posZ'][start:end-1])
    v=data['data']['linear_velocity'][start:end-1]/2
    cond=np.where(v>0.01) # condition for start
    trialv.append(v[cond])
    trialw.append((data['data']['angular_velocity'][start:end-1]/90)[cond])



# overhead path
ind=0
plt.scatter(trialx[ind], trialz[ind],c=np.linspace(0,1,len(trialz[ind])),alpha=0.3)
plt.colorbar()
plt.plot(trialx[ind][:],trialz[ind][:])
ind+=1

# control curve
ind=0
plt.scatter(list(range(len(trialw[ind]))),trialv[ind],c=np.linspace(0,1,len(trialw[ind])),alpha=0.3)
plt.scatter(list(range(len(trialw[ind]))),trialw[ind],c=np.linspace(0,1,len(trialw[ind])),alpha=0.3)
plt.colorbar()
plt.plot(trialv[ind])
plt.plot(trialw[ind])
ind+=1


ind=0
plt.scatter(trialw[ind],trialv[ind],c=np.linspace(0,1,len(trialw[ind])),alpha=0.9)
plt.colorbar()
plt.plot(trialw[ind],trialv[ind],alpha=0.3)
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
ind+=1


# this downsampling is good
orignal_dt=0.012
downsampleddata=down_sampling_(trialw[ind],orignal_dt=orignal_dt)
plt.plot(downsampleddata)
plt.plot(np.array(list(range(len(trialw[ind]))))*orignal_dt, trialw[ind])
ind+=1



# down sample the controls an see if it reproduce the states
statey=[down_sampling_(trialx[ind],orignal_dt=orignal_dt) for ind in range(len(trialw))]
statex=[down_sampling_(trialz[ind],orignal_dt=orignal_dt) for ind in range(len(trialw))]
controlv=[np.array(down_sampling_(trialv[ind],orignal_dt=orignal_dt)) for ind in range(len(trialw))]
controlw=[np.array(down_sampling_(trialw[ind],orignal_dt=orignal_dt)) for ind in range(len(trialw))]

actions=[ torch.tensor([v,w]).float().T for v, w in zip(controlv, controlw) ]

def run_trial(env,given_action,
            phi=torch.tensor([[2],   
                [pi/2],   
                [1e-8],   
                [1e-8],   
                [0.0],   
                [0.0],   
                [0.13],   
                [0.5],   
                [0.5],   
                [0.5],   
                [0.5]])):

    def _collect():
        epstates.append(env.s)
    # saves
    epstates=[]
    env.reset(phi=phi)
    t=0
    while t<len(given_action):
        env.step(given_action[t]) 
        _collect()
        t+=1
    env.step(given_action[-1]) 
    _collect()
    return torch.stack(epstates, axis=0)[1:,:,0]

env.terminal_vel=0.0 # prevent the asseertion of stop error\
env.debug=True
states=[]
for a in actions[:55]:
    epstates=run_trial(env,a)
    states.append(epstates)

# it reproduce the same states and v w. there is a lag in dt but we will remove the reaction time so no prob
ind=0
plt.plot(states[ind][:,0],states[ind][:,1])
plt.plot(statex[ind],statey[ind])
plt.axis('equal')
ind+=1

ind=0
plt.plot(states[ind][:,3])
plt.plot(controlv[ind])
plt.plot(states[ind][:,4])
plt.plot(controlw[ind])
ind+=1

# save -----------------------------------------
def process_turnoffdata(data, env):
    trialv, trialw=[],[] # actions
    for start, end in zip(data['data']['start_trial'].astype(int),data['data']['end_trial'].astype(int)):
        v=data['data']['linear_velocity'][start:end-1]/2
        cond=np.where(v>0.01) # condition for start
        trialv.append(v[cond])
        trialw.append((data['data']['angular_velocity'][start:end-1]/90)[cond])

    controlv=[np.array(down_sampling_(trialv[ind],orignal_dt=0.012)) for ind in range(len(trialw))]
    controlw=[np.array(down_sampling_(trialw[ind],orignal_dt=0.012)) for ind in range(len(trialw))]

    actions=[ torch.tensor([v,w]).float().T for v, w in zip(controlv, controlw) ]

    states=[]
    for a in actions:
        epstates=run_trial(env,a)
        states.append(epstates)

    tasks=[]
    for start, end in zip(data['data']['start_trial'].astype(int),data['data']['end_trial'].astype(int)):
        tasks.append([data['data']['FFz'][start],data['data']['FFx'][start]])
    tasks=np.array(tasks).astype('float32')


    stimdur=(data['dis_data']['stimDur']*10).astype(int)

    return states, actions, tasks, stimdur

from pathlib import Path
folder=Path('/data/human/turnoffdata')


env=ffacc_real.FireFlyPaper(arg)
env.debug=True
env.terminal_vel=0.0 # prevent the asseertion of stop error\


for matfile in folder.glob('*.mat'):
    print(matfile)
    data=mat73.loadmat(matfile)
    states, actions, tasks, stimdur=process_turnoffdata(data, env)
    packed=(states, actions, tasks, stimdur)

    savename=matfile.parent/matfile.name[:-4]
    
    with open(savename, 'wb') as handle:
        pickle.dump(packed, handle, protocol=pickle.HIGHEST_PROTOCOL)






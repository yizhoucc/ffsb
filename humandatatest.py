

import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from monkey_functions import down_sampling
from numpy import pi, save
import pickle
import torch
from FireflyEnv import ffacc_real
from Config import Config
arg = Config()
env=ffacc_real.FireFlyPaper(arg)

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
            print(d,type(d))
            if isinstance(d, spio.matlab.mio5_params.mat_struct):
                each = _todict(d)
            res.append(each)
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

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    data=data['subjects']
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


# offical process and save -----------------------------------------------------------
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


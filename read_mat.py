import os
import h5py
import numpy as np
import pickle
# f= h5py.File("C:/Users/24455/Desktop/Data.mat", 'r')


def mat2pickle(filename,savename):
    f= h5py.File(filename, 'r')
    number_subjects=f['subject']['trials'].shape[0]
    subjects=[]
    for subject_index in range(number_subjects):
        ref=f['subject']['trials'][subject_index][0]
        subject=f[ref]
        number_trials=subject['continuous'].shape[0]
        trials=[]
        for trial_index in range(number_trials):
            trial_dict={}
            for meta in subject.keys():
                trial_dict[meta]={}
                for item in f[(subject[meta])[trial_index,0]].keys():
                    if f[(subject[meta])[trial_index,0]][item].shape==(1,1):
                        trial_dict[meta][item]=f[(subject[meta])[trial_index,0]][item][0,0]
                    else:
                        trial_dict[meta][item]=f[(subject[meta])[trial_index,0]][item][:]
            trials.append(trial_dict)
        subjects.append(trials)
    pickle.dump(subjects, open(savename, "wb" ))
    print('done')







# list(f['subject'].keys())
# f.name
subject_index=0
trial_index=0
ref=f['subject']['trials'][subject_index][0]
# ref=f['subject/trials'][subject_index][0]
subject1=f[ref]
subject1['continuous'].keys()
continuous.keys()
f.keys()
f['subject'].keys()
subject1.keys()
# continuous
continuous_list=['v','w']
continuous_dict={}
continuous=f[(subject1['continuous'])[trial_index,0]]
for item in continuous_list:
    continuous_dict[item]=continuous[item][0,:].tolist()


number_subjects=f['subject']['trials'].shape[0]
subjects=[]
for subject_index in range(number_subjects):
    ref=f['subject']['trials'][subject_index][0]
    subject=f[ref]
    number_trials=subject['continuous'].shape[0]
    trials=[]
    for trial_index in range(number_trials):
        trial_dict={}
        for meta in subject.keys():
            trial_dict[meta]={}
            for item in f[(subject[meta])[trial_index,0]].keys():
                if f[(subject[meta])[trial_index,0]][item].shape==(1,1):
                    trial_dict[meta][item]=f[(subject[meta])[trial_index,0]][item][0,0]
                else:
                    trial_dict[meta][item]=f[(subject[meta])[trial_index,0]][item][:]
        trials.append(trial_dict)
    subjects.append(trials)
pickle.dump(subjects, open( "C:/Users/24455/Desktop/data.p", "wb" ))




human_data = pickle.load(open( "C:/Users/24455/Desktop/data.p", "rb"))

vs=human_data[0][0]['continuous']['JS_linear']
import matplotlib.pyplot as plt 
plt.plot(vs.reshape(-1))
from scipy import signal
sos = signal.butter(3, 40, 'low', fs=100, output='sos')
filtered = signal.sosfilt(sos, vs.reshape(-1))
plt.plot(filtered)
plt.plot([a for i,a in enumerate(filtered) if i%20==0])

def down_sample(data,dt=0.2 ,samplingf=60, order=3,cutofff=20 ):
    new_samplingf=1/dt # 5 if dt=0.2
    # filter
    sos = signal.butter(order, cutofff, 'low', fs=samplingf, output='sos')
    if data.shape[0] == 1:
        filtered = signal.sosfilt(sos, data.reshape(-1))
    else:
        filtered = signal.sosfilt(sos, data)
    # down sample
    downsampled=[a for i,a in enumerate(filtered) if i%(samplingf/new_samplingf)==0]
    return downsampled

# plt.plot(down_sample(vs))
a=human_data[0][1001]['continuous']['JS_linear']
human_data[0][1000]['prs']['r_sub']
plt.plot(a.reshape(-1))

for trial_index in range(1200,1300):
    print(human_data[0][trial_index]['prs']['r_tar'])
    
for trial_index in range(1200,1300):
    a=human_data[0][trial_index]['continuous']['xmp']#-human_data[0][trial_index]['prs']['fireflyposx']
    plt.plot(a.reshape(-1))

for trial_index in range(1200,1300):
    sx=human_data[0][trial_index]['continuous']['xmp']#-human_data[0][trial_index]['prs']['fireflyposx']
    sy=human_data[0][trial_index]['continuous']['ymp']#-human_data[0][trial_index]['prs']['fireflyposy']
    plt.plot(sx.reshape(-1),sy.reshape(-1))

ax=human_data[0][trial_index]['continuous']['xmp']-human_data[0][trial_index]['prs']['fireflyposx']
ay=human_data[0][trial_index]['continuous']['ymp']-human_data[0][trial_index]['prs']['fireflyposy']
ax=-96
ay=18

def get_task_info(data,subject_index, trial_index,goal_radius):
    goalx=data[subject_index][trial_index]['prs']['fireflyposx']
    goaly=data[subject_index][trial_index]['prs']['fireflyposy']
    tau=data[subject_index][trial_index]['prs']['tau']
    vgain=data[subject_index][trial_index]['prs']['vmax']
    wgain=data[subject_index][trial_index]['prs']['wmax']



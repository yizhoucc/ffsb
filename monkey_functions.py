import pickle
from PIL.Image import ROTATE_90
import torch
import pandas
import numpy as np
from numpy import pi
import warnings
from matplotlib import pyplot as plt
from scipy.signal import resample

# testing imports


# from __future__ import division
# import numpy as np
# import matplotlib.pyplot as plt

# forwarddf=df[df.target_x<50][df.target_x>-50]
# dataf=[]
# for d in forwarddf.action_w:
#     dataf.append(d)

# ldf=df[df.target_x>90]
# rdf=df[df.target_x<-90]
# datas=[]
# for d in ldf.action_w:
#     datas.append(d)
# for d in rdf.action_w:
#     datas.append(d)


# maxlen=0
# for d in dataf:
#     maxlen=max(maxlen,len(d))


# p=[]
# for d in datas:
#     if len(d)<maxlen:
#         d+=[0.]*(maxlen-len(d))
#     ps = np.abs(np.fft.fft(d))**2
#     time_step = 1 / 10
#     freqs = np.fft.fftfreq(len(d), time_step)
#     p.append(ps.tolist())

# p=np.array(p)
# meanp=np.mean(p,0)
# idx = np.argsort(freqs)
# plt.plot(freqs[idx], meanp[idx])
# plt.xticks(rotation = 45)
# plt.xlabel('Hz')
# plt.ylabel('power specturm')

# from scipy import signal
# ys=[]
# for d in dataf:
#     if len(d)<maxlen:
#         d+=[0.]*(maxlen-len(d))
#     d=np.array(d)
#     f, Pxx_den = signal.welch(d, 10)
#     ys.append(Pxx_den.tolist())
# ys=np.array(ys)
# meanp=np.mean(ys,0)
# plt.plot(f, meanp)
# # plt.ylim([0.5e-3, 1])
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()
from operator import itemgetter 


def datawash(df):
    from pandas import concat
    trialtypes={}
    df=df[df.trial_dur<4]
    df=concat([df[df.category=='skip'], df[df.category=='normal']])
    trialtypes[1]=df[df.full_on == False]
    
    return df


def dfdownsample(df,filename):
    index=0
    while index<df.shape[0]:
        df.pos_x[index]=down_sampling(df.pos_x[index], 0.0012,0.1)
        df.pos_y[index]=down_sampling(df.pos_y[index], 0.0012,0.1)
        df.head_dir[index]=down_sampling(df.head_dir[index], 0.0012,0.1)
        df.pos_r[index]=down_sampling(df.pos_r[index], 0.0012,0.1)
        df.pos_v[index]=down_sampling(df.pos_v[index], 0.0012,0.1)
        df.pos_w[index]=down_sampling(df.pos_w[index], 0.0012,0.1)
        df.relative_radius[index]=down_sampling(df.relative_radius[index], 0.0012,0.1)
        df.relative_angle[index]=down_sampling(df.relative_angle[index], 0.0012,0.1)
        df.time[index]=down_sampling(df.time[index], 0.0012,0.1)
        df.action_v[index]=down_sampling(df.action_v[index], 0.0012,0.1)
        df.action_w[index]=down_sampling(df.action_w[index], 0.0012,0.1)
        index+=1
    df.to_pickle(filename)


def load_monkey_data(filename, partial_obs=True):

    infile = open('monkey_data/{}.pkl'.format(filename),'rb')
    df = pickle.load(infile)
    # throw out full obs trials
    if partial_obs:
        df=df[df.full_on==False]
    return df


def data_iter(batch_size,states,actions,tasks):
    num_samples=len(tasks)
    allinds=list(range(num_samples))
    np.random.shuffle(allinds)
    for i in range(0, num_samples, batch_size):
        batchinds=torch.tensor(allinds[i:min(i+batch_size, num_samples)])
        yield itemgetter(*batchinds)(states),itemgetter(*batchinds)(actions),itemgetter(*batchinds)(tasks)


def monkey_data_downsampled_(df,factor=0.0025):
    states = []
    actions = []
    tasks=[]
    index=0
    while index<df.shape[0]:
        try:
            if len(df.action_v[index])>2:
                # process data.
                # all angles into radius scale -pi/2
                # x=y, y=-x (rotation)
                xs=convert_unit(torch.tensor(df.pos_y[index], dtype=torch.float), factor=factor)
                ys=-convert_unit(torch.tensor(df.pos_x[index], dtype=torch.float), factor=factor)
                hs=pi/180*torch.tensor(df.head_dir[index], dtype=torch.float)-pi/2
                vs=convert_unit(torch.tensor(df.pos_v[index], dtype=torch.float),factor=factor)
                ws=pi/180*torch.tensor(df.pos_w[index], dtype=torch.float)
                state=torch.stack([xs,ys,hs,vs,ws]).t()
                
                wctrls=torch.tensor(df.action_w[index]).float()
                vctrls=torch.tensor(df.action_v[index]).float()
                action=[vctrls,wctrls]
                action=torch.stack(action).t()

                task=[convert_unit(df.target_y[index],factor=factor),-convert_unit(df.target_x[index],factor=factor)]

                states.append(state)
                actions.append(action)
                tasks.append(task)
        except:
            index+=1
            continue
        index+=1
    return states, actions, tasks


def monkey_data_downsampled(df,factor=0.0025):
    states = []
    actions = []
    tasks=[]
    index=0
    while index<df.shape[0]:
        trial=df.iloc[index]
        try:
            if len(trial.action_v)>2:
                xs=convert_unit(torch.tensor(trial.pos_y, dtype=torch.float), factor=factor)
                ys=-convert_unit(torch.tensor(trial.pos_x, dtype=torch.float), factor=factor)
                hs=pi/180*torch.tensor(trial.head_dir, dtype=torch.float)-pi/2
                vs=convert_unit(torch.tensor(trial.pos_v, dtype=torch.float),factor=factor)
                ws=pi/180*torch.tensor(trial.pos_w, dtype=torch.float)
                state=torch.stack([xs,ys,hs,vs,ws]).t()
                
                wctrls=torch.tensor(trial.action_w).float()
                vctrls=torch.tensor(trial.action_v).float()
                action=[vctrls,wctrls]
                action=torch.stack(action).t()

                task=[convert_unit(trial.target_y,factor=factor),-convert_unit(trial.target_x,factor=factor)]

                states.append(state)
                actions.append(action)
                tasks.append(task)
        except:
            print('no',index)
            index+=1
            continue
        index+=1
    return states, actions, tasks


def monkey_trajectory(df,new_dt=0.1, goal_radius=65,factor=0.005):
    #------prepare saving vars-----
    states = [] # true location
    actions = [] # action
    tasks=[] # vars that define an episode
    orignal_dt=get_frame_rate(df, default_rate=0.0012)
    index=0
    while index<df.shape[0]:
        try:
            # state=[
            #     [convert_unit(y,factor=factor),convert_unit(x,factor=factor)] for x, y in zip(down_sampling(df.px[index], orignal_dt, new_dt),down_sampling(df.py[index], orignal_dt, new_dt))
            #     ]
            xs=convert_unit(torch.tensor(down_sampling(df.pos_y[index], orignal_dt, new_dt), dtype=torch.float), factor=factor)
            ys=-convert_unit(torch.tensor(down_sampling(df.pos_x[index], orignal_dt, new_dt), dtype=torch.float), factor=factor)
            initx=xs[0]
            inity=ys[0]
            xs=xs-initx
            ys=ys-inity
            hs=torch.stack(down_sampling((pi/180*torch.tensor(df.head_dir[index], dtype=torch.float)), orignal_dt, new_dt)).float()-pi/2
            vs=convert_unit(torch.stack(down_sampling((torch.tensor(df.pos_v[index], dtype=torch.float)), orignal_dt, new_dt)),factor=factor).float()
            ws=-pi/180*torch.stack(down_sampling((torch.tensor(df.pos_w[index], dtype=torch.float)), orignal_dt, new_dt)).float()
            state=torch.stack([xs,ys,hs,vs,ws])
            # df.action_v[index]
            # df.real_v[index]
            # df.action_w[index]
            # down_sampling(torch.tensor(df.real_w[index])*pi/180,orignal_dt, new_dt)
            # plt.plot(df.py[index],df.px[index])
            # plt.plot(df.FFY[index],df.FFX[index],'-o')
            # plt.plot(df.px[index],df.py[index])
            # plt.plot(df.FFX[index],df.FFY[index],'-o')
            # plt.plot(180/pi*torch.atan2(df.FFY[index]-torch.tensor((df.py[index])),df.FFX[index]-torch.tensor((df.px[index]))))
            # plt.plot(torch.atan2(torch.tensor((df.px[index])-df.FFX[index]),torch.tensor((df.py[index])-df.FFY[index]))-torch.tensor(df.real_relative_angle[index]*pi/180))
            # np.arctan((down_sampling(df.px[index], orignal_dt, 0.2)-df.FFX[index])/(down_sampling(df.py[index], orignal_dt, 0.2)-df.FFY[index]))-down_sampling(df.real_relative_angle[index]*pi/180, orignal_dt, 0.2)            
            task=[[ convert_unit(df.target_y[index],factor=factor)-initx,-convert_unit(df.target_x[index],factor=factor)-inity],convert_unit(goal_radius,factor=factor)]
            action=[[v.astype('float32'),w.astype('float32')] for v, w in zip(down_sampling(df.action_v[index], orignal_dt, new_dt),down_sampling(df.action_w[index], orignal_dt, new_dt))]
            states.append(state)
            actions.append(action)
            tasks.append(task)
            index+=1
        except:
            index+=1
            continue
    return states, actions, tasks


def plot_monkey_trial(df,index):
    try:
        fig = plt.figure(figsize=[24, 8])

        ax1 = fig.add_subplot(131)
        ax1.set_xlabel('world x, cm')
        ax1.set_ylabel('world y, cm')
        ax1.set_title('monkey location xy')
        ax1.plot(df.px[index], df.py[index])
        goalcircle = plt.Circle((df.FFX[index],df.FFY[index]), 65, color='y', alpha=0.5)
        ax1.add_patch(goalcircle)
        ax1.set_ylim([-30,200])
        ax1.set_yticks(np.linspace(-30,200,10))
        ax1.set_xlim([-100,100])
        ax1.axis('equal')
        
        ax2 = fig.add_subplot(132)
        ax2.set_xlabel('time, 0.0012 s')
        ax2.set_ylabel('forward v, cm/s')
        ax2.set_ylim([-50,210])
        ax2.set_yticks(np.linspace(-50,200,10))
        ax2.set_title('monkey action v')
        ax2.plot(df.real_v[index])
        
        ax3 = fig.add_subplot(133)
        ax3.set_xlabel('time, 0.0012 s')
        ax3.set_ylabel('angular w, ang/s')
        ax3.set_ylim([-90,90])
        ax3.set_yticks(np.linspace(-90,90,10))
        ax3.set_title('monkey action w')
        ax3.plot(df.real_w[index])
    except KeyError:
        print('this trial (trial number: {}) might be a easy trial. try another.'.format(index))
        return None
    return fig


def monkey_action_distribution(df):
    intial_w = df.real_w.map(lambda x: x[0])
    intial_v = df.real_v.map(lambda x: x[0])

    fig = plt.figure(figsize=[16, 8])
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('angular w, ang/s')
    ax1.set_ylabel('number of trials')
    ax1.set_title('inital angular velocity distribution')
    ax1.hist(intial_w)
    # ax1.set_ylim([-30,200])
    # ax1.set_yticks(np.linspace(-30,200,10))
    # ax1.set_xlim([-100,100])

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('forward v, cm/s')
    ax2.set_ylabel('number of trials')
    ax2.set_title('inital velocity distribution')
    ax2.hist(intial_v)
    # ax2.set_ylim([-50,200])
    # ax2.set_yticks(np.linspace(-50,200,10))

    return fig


def monkey_action_correlation(df):
    intial_y = df.py.map(lambda x: x[0])
    intial_w = df.real_w.map(lambda x: x[0])
    intial_v = df.real_v.map(lambda x: x[0])

    fig = plt.figure(figsize=[16, 8])
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('start y position, cm')
    ax1.set_ylabel('angular w, ang/s')
    ax1.set_title('initial w vs intial y position')
    ax1.scatter(intial_y, intial_w)
    # ax1.set_ylim([-30,200])
    # ax1.set_yticks(np.linspace(-30,200,10))
    # ax1.set_xlim([-100,100])

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('start y position, cm')
    ax2.set_ylabel('forward v, cm/s')
    ax2.set_title('initial v vs intial y position')
    ax2.scatter(intial_y,intial_v)
    # ax2.set_ylim([-50,200])
    # ax2.set_yticks(np.linspace(-50,200,10))

    return fig


def down_sampling_(data_array, orignal_dt=0.0012, new_dt=0.1, continues_data=True):
    # TODO, moving avg for action data.
    result=[]
    index=0
    result_i=[] # int indices
    while True:
        try:
            correction=index*new_dt/orignal_dt-index*new_dt//orignal_dt
            # int part
            result.append(data_array[int((index)*new_dt//orignal_dt)]
            # decimal part (still not precise because of rounding errors)
                +correction*(data_array[int((index)*new_dt//orignal_dt)+1]-data_array[int((index)*new_dt//orignal_dt)]))
            result_i.append(int((index)*new_dt//orignal_dt))
            index+=1
        except IndexError:
            break
    return result


def down_sampling(data_array, orignal_dt=0.0012, new_dt=0.1):
    # data array is in  timestamp x (feature) format
    return resample(data_array, int(len(data_array)*orignal_dt/new_dt))


def get_frame_rate(df, default_rate=0.0012):

    index=0
    while True:
        try:
            frame_rate=(df.time[index][-1]-df.time[index][0])/len(df.time[index])
            break
        except KeyError:
            index+=1
    if (frame_rate-default_rate)/default_rate<1e-4:
        return default_rate
    else:
        warnings.warn('the frame rate of the data is {}, which is not the same as default {}'.format(frame_rate, default_rate))
        return(frame_rate)


def convert_unit(monkey_data, factor=0.005):
    return monkey_data*factor


def get_vlim(df):
    min_v=110
    max_v=0
    index=0
    while index<=df.ep.iloc[-1]:
        try:
            min_v=min(min_v, min(df.real_v[index]))
            max_v=max(max_v,max(df.real_v[index]))
            index+=1
        except KeyError:
            index+=1
            continue
    print(min_v,max_v)
    return min_v,max_v


def get_wlim(df):
    min_w=110
    max_w=0
    index=0
    while index<=df.ep.iloc[-1]:
        try:
            min_w=min(min_w, min(df.real_w[index]))
            max_w=max(max_w,max(df.real_w[index]))
            index+=1
        except KeyError:
            index+=1
            continue

    print(min_w,max_w)
    return min_w, max_w


def pack_data(datapath, expression):
    # new way to use unpacked data
    # datapath=Path("Z:\\bruno_pert")
    # sessions=list(datapath.glob('*ds'))
    # df=None
    # for session in sessions:
    #     with open(session,'rb') as f:
    #         df_ = pickle.load(f)
    #     if df is None:
    #         df=df_
    #     else:
    #         df=df.append(df_)
    # del df_
    print('loading data')
    files=list(datapath.glob(expression))
    df=None
    for each in files:
        with open(each,'rb') as f:
            _df = pickle.load(f)
        if df is None:
            df=_df
        else:
            df=df.append(_df)
    print('done loading data')
    return df
    

def bypass(input):
    return input


class MonkeyDataExtractor():

    # ruiyi example
    # data_path=Path.cwd().parents[1]/'mkdata/bcm/bruno'
    # ext=MonkeyDataExtractor(folder_path=data_path)
    # trajectory=ext()
    # trajectory.to_pickle(data_path/'test.pkl')

    def __init__(self, folder_path):
        self.monkey_class = 'BCM'
        self.folder_path = folder_path
        self.smr_full_file_path = sorted(self.folder_path.glob('*.smr'))
        self.log_full_file_path = [file.parent / (file.stem + '.log') 
                                   for file in self.smr_full_file_path]
        if self.monkey_class == 'NYU':
            self.marker_memo = {'file_start': 1, 'trial_start': 2, 'trial_end': 3,
                            'juice' :4, 'perturb_start': 8}
        else:
            self.marker_memo = {'file_start': 1, 'trial_start': 2, 'trial_end': 3,
                            'juice' :4, 'perturb_start': 8, 'perturb_start2': 5}
            
        self.y_offset = 32.5
    
    def __call__(self, downsample_fun=bypass, saving_fun=bypass, returndata=True):
        if self.monkey_class == 'NYU':
            self.nyu_extract_smr()
            self.nyu_extract_log()
            self.nyu_segment()
        else:
            self.bcm_extract_smr()
            self.bcm_extract_log()
            self.bcm_segment()
            downsample_fun(self.monkey_trajectory)
            saving_fun(self.monkey_trajectory)
        if returndata:
            return self.monkey_trajectory
            
    def nyu_extract_smr(self):
        channel_signal_all = []
        marker_all = []
        
        for idx, file_name in enumerate(self.smr_full_file_path):
            seg_reader = neo.io.Spike2IO(filename=file_name).read_segment()
            
            if idx == 0: # only get sampling rate once
                self.SAMPLING_RATE = seg_reader.analogsignals[0].sampling_rate.item()
             
            # Sometimes the length across channels varies a bit
            analog_length = min([i.size for i in seg_reader.analogsignals])
            channel_signal = np.ones((analog_length, seg_reader.size['analogsignals'] + 1))
            
            channel_names = []
            for ch_idx, ch_data in enumerate(seg_reader.analogsignals):
                channel_signal[:, ch_idx] = ch_data.as_array()[:analog_length].T
                channel_names.append(ch_data.annotations['channel_names'][0])
            
            # Add a time channel
            channel_signal[:, -1] = seg_reader.analogsignals[0].times[:analog_length]
            channel_names.append('Time') 
            
            channel_signal_all.append(pd.DataFrame(channel_signal, columns=channel_names))
            
            marker_channel_idx = [idx for idx, value 
                                    in enumerate(seg_reader.events)
                                    if value.name == 'marker'][0]
            marker_key, marker_time = (
                seg_reader.events[marker_channel_idx].get_labels().astype('int'),
                seg_reader.events[marker_channel_idx].as_array())
            marker = {'key': marker_key, 'time': marker_time}
            marker_all.append(marker)
            
            self.channel_signal_all = channel_signal_all
            self.marker_all = marker_all
            
    def nyu_extract_log(self):
        log_data_all = []
        
        for file_name in self.log_full_file_path:
            with open(file_name, 'r', encoding='UTF-8') as content:
                log_content = content.readlines()
            
            floor_density = []
            perturb_vpeak = []; perturb_wpeak = []; perturb_start_time_ori = []
            full_on = []; target_x = []; target_y = []
            for line in log_content:
                if 'Joy Stick Max Velocity' in line:
                    gain_v = float(line.split(': ')[1])
                    
                if 'Joy Stick Max Angular Velocity' in line:
                    gain_w = float(line.split(': ')[1])
                    
                if 'Perturb Max Velocity' in line:
                    perturb_vpeakmax = float(line.split(': ')[1])
                    
                if 'Perturb Max Angular Velocity' in line:
                    perturb_wpeakmax = float(line.split(': ')[1])
                    
                if 'Perturbation Sigma' in line:
                    perturb_sigma = float(line.split(': ')[1])
                    
                if 'Perturbation Duration' in line:
                    perturb_dur = float(line.split(': ')[1])
                    
                if 'Floor Density' in line:
                    content_temp = float(line.split(': ')[1])
                    floor_density.append(content_temp)
                    
                if 'Perturbation Linear Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_vpeak.append(content_temp)
                    
                if 'Perturbation Angular Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_wpeak.append(- content_temp)
                    
                if 'Perturbation Delay Time' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_start_time_ori.append(content_temp / 1000)  # ms to s
                    
                if 'Firefly Full On' in line:
                    content_temp = bool(int(line.split(': ')[1]))
                    full_on.append(content_temp)
                
                if 'Position x/y(cm)' in line:
                    content_temp_x, content_temp_y = line.split(': ')[1].split(' ')
                    target_x.append(float(content_temp_x))
                    # Monkey data's y positions are reversed.
                    target_y.append(- float(content_temp_y) + self.y_offset)
                
            log_data_all.append({'gain_v': gain_v, 'gain_w': gain_w,
                                 'perturb_vpeakmax': perturb_vpeakmax, 'perturb_wpeakmax': perturb_wpeakmax,
                                 'perturb_sigma': perturb_sigma, 'perturb_dur': perturb_dur,
                                 'floor_density': floor_density,
                                 'perturb_vpeak': perturb_vpeak, 'perturb_wpeak': perturb_wpeak,
                                 'perturb_start_time_ori': perturb_start_time_ori,
                                  'full_on': full_on, 'target_x': target_x, 'target_y': target_y})
            
            self.log_data_all = log_data_all
            
    def nyu_segment(self, lazy_threshold=4000, skip_threshold=400, skip_r_threshold=30, 
                    crazy_threshold=200,
                    medfilt_kernel=5, v_threshold=1, reward_boundary=65, 
                    perturb_corr_threshold=50):
        # lazy_threshold (data points): Trial is too long.
        # skip_threshold (data points): Trial is too short.
        # skip_r_threshold (cm): Monkey did not move a lot.
        # crazy_threshold (cm): Monkey stopped too far.
        # v_threshold (cm/s): Velocity threshold for start and end.
        # reward_boundary (cm): Rewarded when stop inside this circular boundary.
        # perturb_corr_threshold (data points): Corrected perturbation start index should not be too biased.
        
        gain_v = []; gain_w = []; perturb_vpeakmax = []; perturb_wpeakmax = []
        perturb_sigma = []; perturb_dur = []; perturb_vpeak = []; perturb_wpeak = []
        perturb_v = []; perturb_w = []; perturb_v_gauss = []; perturb_w_gauss = []
        perturb_start_time = []; perturb_start_time_ori = []
        floor_density = []; pos_x = []; pos_y = []
        head_dir = []; head_dir_end = []; pos_r = []; pos_theta = []
        pos_r_end = []; pos_theta_end = []
        pos_v = []; pos_w = []; target_x = []; target_y = []
        target_r = []; target_theta = []; full_on = []; rewarded = []
        relative_radius = []; relative_angle = []; time = []
        trial_dur = []; action_v = []; action_w = []
        relative_radius_end = []; relative_angle_end = []; category = []

        for session_idx, session_data in enumerate(self.channel_signal_all):
            log_data = self.log_data_all[session_idx]
            marker_data = self.marker_all[session_idx]
            start_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_start']]
            end_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_end']]
            perturb_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['perturb_start']]

            # segment trials
            for trial_idx in range(end_marker_times.size):
                trial_data = session_data[np.logical_and(
                    session_data.Time > start_marker_times[trial_idx],
                    session_data.Time < end_marker_times[trial_idx])].copy()
                
                # Use median filter kernel size as 5 to remove spike noise first.
                trial_data['ForwardV'] = medfilt(trial_data['ForwardV'], medfilt_kernel)
                trial_data['AngularV'] = medfilt(trial_data['AngularV'], medfilt_kernel)
                
                # cut non-moving head and tail
                moving_period = np.where(trial_data['ForwardV'].abs() > v_threshold)[0]
                if moving_period.size > 0:
                    start_idx = moving_period[0]
                    end_idx = moving_period[-1] + 2
                else:
                    start_idx = 0
                    end_idx = None
                  
                # store trial data
                trial_data = trial_data.iloc[start_idx : end_idx]
                trial_data['AngularV'] = - trial_data['AngularV']
                trial_data['MonkeyYa'] = np.cumsum(trial_data['AngularV']) / self.SAMPLING_RATE + 90
                trial_data['MonkeyX'] = np.cumsum(trial_data['ForwardV']
                                            * np.cos(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                trial_data['MonkeyY'] = np.cumsum(trial_data['ForwardV']
                                            * np.sin(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                
                gain_v.append(log_data['gain_v'])
                gain_w.append(log_data['gain_w'])
                perturb_vpeakmax.append(log_data['perturb_vpeakmax'])
                perturb_wpeakmax.append(log_data['perturb_wpeakmax'])
                perturb_vpeak.append(log_data['perturb_vpeak'][trial_idx])
                perturb_wpeak.append(log_data['perturb_wpeak'][trial_idx])
                perturb_start_time_ori.append(log_data['perturb_start_time_ori'][trial_idx])
                perturb_sigma.append(log_data['perturb_sigma'])
                perturb_dur.append(log_data['perturb_dur'])
                pos_x.append(trial_data['MonkeyX'].values)
                pos_y.append(trial_data['MonkeyY'].values)
                head_dir.append(trial_data['MonkeyYa'].values)
                head_dir_end.append(trial_data['MonkeyYa'].values[-1])
                floor_density.append(log_data['floor_density'][trial_idx])
                full_on.append(log_data['full_on'][trial_idx])

                rho, phi = cart2pol(pos_x[-1], pos_y[-1])
                pos_r.append(rho)
                pos_theta.append(np.rad2deg(phi))
                pos_r_end.append(rho[-1])
                pos_theta_end.append(np.rad2deg(phi[-1]))
                
                
                # determine if it is a perturbation trial
                perturb_start_time_temp = perturb_marker_times[(np.logical_and(
                                            perturb_marker_times > start_marker_times[trial_idx],
                                            perturb_marker_times < end_marker_times[trial_idx]))]
                if bool(perturb_start_time_temp.size):
                    assert perturb_start_time_temp.size == 1
                    pos_v.append(trial_data['ForwardV'].values)
                    pos_w.append(trial_data['AngularV'].values)
                    
                    # construct perturbation curves
                    perturb_xaxis = np.linspace(0, perturb_dur[-1], round(self.SAMPLING_RATE))
                    perturb_temp = norm.pdf(perturb_xaxis, loc=perturb_dur[-1] / 2, scale=perturb_sigma[-1])
                    perturb_temp /= perturb_temp.max()
                    perturb_v_temp = perturb_temp * perturb_vpeak[-1]
                    perturb_w_temp = perturb_temp * perturb_wpeak[-1]
                    perturb_v_gauss.append(perturb_v_temp)
                    perturb_w_gauss.append(perturb_w_temp)
                    
                    # use obvious angular perturbation curve as a template
                    if abs(perturb_wpeak[-1]) / perturb_wpeakmax[-1] > 0.1:
                        perturb_template = perturb_w_temp
                        original_vel = pos_w[-1]
                    else:
                        perturb_template = perturb_v_temp
                        original_vel = pos_v[-1]
                        
                    # use the template to do cross-correlation to find perturbation start time
                    perturb_start_idx_mark = int((perturb_start_time_temp
                                                - trial_data['Time'].values[0]) * self.SAMPLING_RATE)
                    perturb_start_idx_mark = np.clip(perturb_start_idx_mark, 0, None)
                    perturb_peak_idx = np.correlate(original_vel, perturb_template, mode='same').argsort()[::-1]
                    perturb_start_idx_corr = perturb_peak_idx - perturb_dur[-1] / 2 * self.SAMPLING_RATE
                    mask = (perturb_start_idx_corr > 0) \
                           & (perturb_start_idx_corr > perturb_start_idx_mark) \
                           & (perturb_start_idx_corr - perturb_start_idx_mark < perturb_corr_threshold)
                    
                    if mask.sum() == 0 or original_vel.size < perturb_template.size:
                        perturb_start_idx = np.clip(perturb_start_idx_mark, None, pos_v[-1].size - 1)
                    else:
                        perturb_start_idx = int(perturb_start_idx_corr[mask][0])
                    perturb_start_time.append(perturb_start_idx / self.SAMPLING_RATE)
                    
                    # get pure actions
                    perturb_v_full = np.zeros_like(pos_v[-1])
                    perturb_v_full[perturb_start_idx:perturb_start_idx + perturb_v_temp.size] = \
                                                            perturb_v_temp[:perturb_v_full.size - perturb_start_idx]
                    perturb_w_full = np.zeros_like(pos_w[-1])
                    perturb_w_full[perturb_start_idx:perturb_start_idx + perturb_w_temp.size] = \
                                                            perturb_w_temp[:perturb_w_full.size - perturb_start_idx]
                    
                    perturb_v.append(perturb_v_full); perturb_w.append(perturb_w_full)
                    action_v.append((pos_v[-1] - perturb_v_full).clip(-gain_v[-1], gain_v[-1]) / gain_v[-1])
                    action_w.append((pos_w[-1] - perturb_w_full).clip(-gain_w[-1], gain_w[-1]) / gain_w[-1])
                else:
                    pos_v.append(trial_data['ForwardV'].values.clip(-gain_v[-1], gain_v[-1]))
                    pos_w.append(trial_data['AngularV'].values.clip(-gain_w[-1], gain_w[-1]))
                    perturb_v_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_w_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_start_time.append(np.nan)
                    perturb_v.append(np.zeros_like(pos_v[-1])); perturb_w.append(np.zeros_like(pos_w[-1]))
                    action_v.append(pos_v[-1] / gain_v[-1])
                    action_w.append(pos_w[-1] / gain_w[-1])
                
                trial_data['Time'] -= trial_data['Time'].iloc[0]
                time.append(trial_data['Time'].values)
                trial_dur.append(trial_data['Time'].values[-1])
                target_x.append(log_data['target_x'][trial_idx])
                target_y.append(log_data['target_y'][trial_idx])
                tar_rho, tar_phi = cart2pol(target_x[-1], target_y[-1])
                target_r.append(tar_rho)
                target_theta.append(np.rad2deg(tar_phi))
                
                relative_r, relative_ang = get_relative_r_ang(
                                pos_x[-1], pos_y[-1], head_dir[-1], target_x[-1], target_y[-1])
                relative_radius.append(relative_r)
                relative_angle.append(np.rad2deg(relative_ang))
                relative_radius_end.append(relative_r[-1])
                relative_angle_end.append(np.rad2deg(relative_ang[-1]))
                rewarded.append(relative_r[-1] < reward_boundary)

                # Categorize trials
                if rewarded[-1]:
                    category.append('normal')
                else:
                    if trial_data['ForwardV'].size < skip_threshold or\
                       pos_r_end[-1] < skip_r_threshold:
                        category.append('skip')
                    elif trial_data['ForwardV'].size > lazy_threshold:
                        category.append('lazy')
                    elif relative_r[-1] > crazy_threshold:
                        category.append('crazy')
                    else:
                        category.append('normal')


        # Construct a dataframe   
        self.monkey_trajectory = pd.DataFrame().assign(gain_v=gain_v, gain_w=gain_w, 
                                 perturb_vpeakmax=perturb_vpeakmax, perturb_wpeakmax=perturb_wpeakmax,
                                 perturb_sigma=perturb_sigma, perturb_dur=perturb_dur,
                                 perturb_vpeak=perturb_vpeak, perturb_wpeak=perturb_wpeak,
                                 perturb_start_time=perturb_start_time,
                                 perturb_start_time_ori=perturb_start_time_ori,
                                 perturb_v_gauss=perturb_v_gauss, perturb_w_gauss=perturb_w_gauss,
                                 perturb_v=perturb_v, perturb_w=perturb_w,
                                 floor_density=floor_density, pos_x=pos_x,
                                 pos_y=pos_y, head_dir=head_dir, head_dir_end=head_dir_end,
                                 pos_r=pos_r, pos_theta=pos_theta, pos_r_end=pos_r_end,
                                 pos_theta_end=pos_theta_end, pos_v=pos_v, pos_w=pos_w, 
                                 target_x=target_x, target_y=target_y, target_r=target_r,
                                 target_theta=target_theta, full_on=full_on, rewarded=rewarded,
                                 relative_radius=relative_radius, relative_angle=relative_angle,
                                 time=time, trial_dur=trial_dur, 
                                 action_v=action_v, action_w=action_w, 
                                 relative_radius_end=relative_radius_end,
                                 relative_angle_end=relative_angle_end, category=category)

    def bcm_extract_smr(self):
        print('starting ext')
        channel_signal_all = []
        marker_all = []
        
        for idx, file_name in enumerate(self.smr_full_file_path):
            seg_reader = neo.io.Spike2IO(filename=str(file_name)).read_segment()
            
            if idx == 0: # only get sampling rate once
                self.SAMPLING_RATE = seg_reader.analogsignals[0].sampling_rate.item()
                
            # Sometimes the length across channels varies a bit
            analog_length = min([i.size for i in seg_reader.analogsignals])
            channel_signal = np.ones((analog_length, seg_reader.size['analogsignals']))
            
            channel_names = []
            # Do not read the last channel as it has a unique shape.
            for ch_idx, ch_data in enumerate(seg_reader.analogsignals[:-1]):
                channel_signal[:, ch_idx] = ch_data.as_array()[:analog_length].T
                channel_names.append(ch_data.annotations['channel_names'][0])
            
            # Add a time channel
            channel_signal[:, -1] = seg_reader.analogsignals[0].times[:analog_length]
            channel_names.append('Time') 
            
            channel_signal_all.append(pd.DataFrame(channel_signal,columns=channel_names))
            
            marker_channel_idx = [idx for idx, value 
                                    in enumerate(seg_reader.events)
                                    if value.name == 'marker'][0]
            marker_key, marker_time = (
                seg_reader.events[marker_channel_idx].get_labels().astype('int'),
                seg_reader.events[marker_channel_idx].as_array())
            marker = {'key': marker_key, 'time': marker_time}
            marker_all.append(marker)
            
            self.channel_signal_all = channel_signal_all
            self.marker_all = marker_all
            
    def bcm_extract_log(self):
        log_data_all = []
        
        for file_name in self.log_full_file_path:
            with open(file_name, 'r', encoding='UTF-8') as content:
                log_content = content.readlines()
                
            floor_density = []
            perturb_vpeak = []; perturb_wpeak = []
            perturb_start_time_ori = []
            full_on = []
            for line in log_content:
                if 'Joy Stick Max Velocity' in line:
                    gain_v = float(line.split(': ')[1])
                    
                if 'Joy Stick Max Angular Velocity' in line:
                    gain_w = float(line.split(': ')[1])
                    
                if 'Perturb Max Velocity' in line:
                    perturb_vpeakmax = float(line.split(': ')[1])
                    
                if 'Perturb Max Angular Velocity' in line:
                    perturb_wpeakmax = float(line.split(': ')[1])
                    
                if 'Perturbation Sigma' in line:
                    perturb_sigma = float(line.split(': ')[1])
                    
                if 'Perturbation Duration' in line:
                    perturb_dur = float(line.split(': ')[1])
                    
                if 'Floor Density' in line:
                    content_temp = float(line.split(': ')[1])
                    floor_density.append(content_temp)
                    
                if 'Perturbation Linear Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_vpeak.append(content_temp)
                    
                if 'Perturbation Angular Speed' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_wpeak.append(- content_temp)
                    
                if 'Perturbation Delay Time' in line:
                    content_temp = float(line.split(': ')[1])
                    perturb_start_time_ori.append(content_temp / 1000)
                    
                if 'Firefly Full ON' in line:
                    content_temp = bool(int(line.split(': ')[1]))
                    full_on.append(content_temp)
            
            if len(full_on) == 0: # Quigley's perturbation sessions
                full_on = [False] * len(floor_density)
            
            log_data_all.append({'gain_v': gain_v, 'gain_w': gain_w, 
                                 'perturb_vpeakmax': perturb_vpeakmax, 'perturb_wpeakmax': perturb_wpeakmax,
                                 'perturb_sigma': perturb_sigma, 'perturb_dur': perturb_dur,
                                 'floor_density': floor_density, 
                                 'perturb_vpeak': perturb_vpeak, 'perturb_wpeak': perturb_wpeak,
                                 'perturb_start_time_ori': perturb_start_time_ori,
                                 'full_on': full_on})
            self.log_data_all = log_data_all
            
    def bcm_segment(self, lazy_threshold=4000, skip_threshold=400, skip_r_threshold=30, 
                    crazy_threshold=200,
                    medfilt_kernel=5, v_threshold=1, reward_boundary=65,
                    target_r_range=[100, 400], target_theta_range=[55, 125], 
                    target_tolerance=1, perturb_corr_threshold=100):
        print('starting segmenting')
        # lazy_threshold (time): Trial is too long.

        # skip_threshold (time): Trial is too short.
        # and
        # skip_r_threshold (cm): Monkey did not move a lot.

        # crazy_threshold (cm): Monkey stopped too far, very wrong.

        # medfilt_kernel (data points): Remove spikes from raw data.
        # v_threshold (cm/s): Threshold for end point.
        # reward_boundary (cm): Rewarded when stop inside this circular boundary.
        # target_r_range (cm): Radius of target distribution.
        # target_theta_range (deg): Angle of target distribution.
        # target_tolerance (cm or deg): Max tolerance for targets out of distribution.
        # perturb_corr_threshold (data points): Corrected perturbation start index should not be too biased.

        gain_v = []; gain_w = []; perturb_vpeakmax = []; perturb_wpeakmax = []
        perturb_sigma = []; perturb_dur = []; perturb_vpeak = []; perturb_wpeak = []
        perturb_v = []; perturb_w = []; perturb_v_gauss = []; perturb_w_gauss = []
        perturb_start_time = []; perturb_start_time_ori = []
        floor_density = []; pos_x = []; pos_y = []
        head_dir = []; head_dir_end = []; pos_r = []; pos_theta = []; 
        pos_r_end = []; pos_theta_end = []
        pos_v = []; pos_w = []; target_x = []; target_y = []
        target_r = []; target_theta = []; full_on = []; rewarded = []
        relative_radius = []; relative_angle = []; time = []; 
        trial_dur = []; action_v = []; action_w = []; 
        relative_radius_end = []; relative_angle_end = []; category = []

        for session_idx, session_data in enumerate(self.channel_signal_all):
            log_data = self.log_data_all[session_idx]
            marker_data = self.marker_all[session_idx]
            start_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_start']]
            end_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['trial_end']]
            perturb_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['perturb_start']]
            if perturb_marker_times.size == 0:
                perturb_marker_times = marker_data['time'][
                            marker_data['key'] == self.marker_memo['perturb_start2']]

            # segment trials
            for trial_idx in range(end_marker_times.size):
                trial_data = session_data[np.logical_and(
                    session_data.Time > start_marker_times[trial_idx],
                    session_data.Time < end_marker_times[trial_idx])].copy()

                # Use median filter kernel size as 5 to remove spike noise first.
                trial_data['ForwardV'] = medfilt(trial_data['ForwardV'], medfilt_kernel)
                trial_data['AngularV'] = medfilt(trial_data['AngularV'], medfilt_kernel)
                
                # cut non-moving head and tail
                moving_period = np.where(trial_data['ForwardV'].abs() > v_threshold)[0]
                if moving_period.size > 0:
                    start_idx = moving_period[0]
                    end_idx = moving_period[-1] + 2
                else:
                    start_idx = 0
                    end_idx = None
                    
                # store trial data
                trial_data = trial_data.iloc[start_idx : end_idx]
                trial_data['AngularV'] = - trial_data['AngularV']
                trial_data['MonkeyYa'] = np.cumsum(trial_data['AngularV']) / self.SAMPLING_RATE + 90
                trial_data['MonkeyX'] = np.cumsum(trial_data['ForwardV']
                                            * np.cos(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                trial_data['MonkeyY'] = np.cumsum(trial_data['ForwardV']
                                            * np.sin(np.deg2rad(trial_data['MonkeyYa']))) / self.SAMPLING_RATE
                
                gain_v.append(log_data['gain_v'])
                gain_w.append(log_data['gain_w'])
                perturb_vpeakmax.append(log_data['perturb_vpeakmax'])
                perturb_wpeakmax.append(log_data['perturb_wpeakmax'])
                perturb_vpeak.append(log_data['perturb_vpeak'][trial_idx])
                perturb_wpeak.append(log_data['perturb_wpeak'][trial_idx])
                perturb_start_time_ori.append(log_data['perturb_start_time_ori'][trial_idx])
                perturb_sigma.append(log_data['perturb_sigma'])
                perturb_dur.append(log_data['perturb_dur'])
                pos_x.append(trial_data['MonkeyX'].values)
                pos_y.append(trial_data['MonkeyY'].values)
                head_dir.append(trial_data['MonkeyYa'].values)
                head_dir_end.append(trial_data['MonkeyYa'].values[-1])
                floor_density.append(log_data['floor_density'][trial_idx])
                full_on.append(log_data['full_on'][trial_idx])
                
                rho, phi = cart2pol(pos_x[-1], pos_y[-1])
                pos_r.append(rho)
                pos_theta.append(np.rad2deg(phi))
                pos_r_end.append(rho[-1])
                pos_theta_end.append(np.rad2deg(phi[-1]))
                
                
                # determine if it is a perturbation trial
                perturb_start_time_temp = perturb_marker_times[(np.logical_and(
                                            perturb_marker_times > start_marker_times[trial_idx],
                                            perturb_marker_times < end_marker_times[trial_idx]))]
                if bool(perturb_start_time_temp.size):
                    assert perturb_start_time_temp.size == 1
                    pos_v.append(trial_data['ForwardV'].values)
                    pos_w.append(trial_data['AngularV'].values)
                
                    # construct perturbation curves
                    perturb_xaxis = np.linspace(0, perturb_dur[-1], round(self.SAMPLING_RATE))
                    perturb_temp = norm.pdf(perturb_xaxis, loc=perturb_dur[-1] / 2, scale=perturb_sigma[-1])
                    perturb_temp /= perturb_temp.max()
                    perturb_v_temp = perturb_temp * perturb_vpeak[-1]
                    perturb_w_temp = perturb_temp * perturb_wpeak[-1]
                    perturb_v_gauss.append(perturb_v_temp)
                    perturb_w_gauss.append(perturb_w_temp)
                    
                    # use the more obvious perturbation curve as a template
                    corrcoef_v = np.correlate(pos_v[-1] - pos_v[-1].mean(), 
                                              perturb_v_temp - perturb_v_temp.mean(),
                                              mode='same') / (pos_v[-1].std() * perturb_v_temp.std())
                    corrcoef_w = np.correlate(pos_w[-1] - pos_w[-1].mean(), 
                                              perturb_w_temp - perturb_w_temp.mean(),
                                              mode='same') / (pos_w[-1].std() * perturb_w_temp.std())
                    if corrcoef_v.max() > corrcoef_w.max():
                        perturb_template = perturb_v_temp
                        original_vel = pos_v[-1]
                    else:
                        perturb_template = perturb_w_temp
                        original_vel = pos_w[-1]
                        
                    # use the template to do cross-correlation to find perturbation start time
                    perturb_start_idx_mark = int((perturb_start_time_temp
                                                - trial_data['Time'].values[0]) * self.SAMPLING_RATE)
                    perturb_start_idx_mark = np.clip(perturb_start_idx_mark, 0, None)
                    perturb_peak_idx = np.correlate(original_vel, perturb_template, mode='same').argsort()[::-1]
                    perturb_start_idx_corr = perturb_peak_idx - perturb_dur[-1] / 2 * self.SAMPLING_RATE
                    mask = (perturb_start_idx_corr > 0) \
                           & (perturb_start_idx_corr > perturb_start_idx_mark) \
                           & (perturb_start_idx_corr - perturb_start_idx_mark < perturb_corr_threshold)
                    
                    if mask.sum() == 0 or original_vel.size < perturb_template.size:
                        perturb_start_idx = np.clip(perturb_start_idx_mark, None, pos_v[-1].size - 1)
                    else:
                        perturb_start_idx = int(perturb_start_idx_corr[mask][0])
                    perturb_start_time.append(perturb_start_idx / self.SAMPLING_RATE)
                    
                    # get pure actions
                    perturb_v_full = np.zeros_like(pos_v[-1])
                    perturb_v_full[perturb_start_idx:perturb_start_idx + perturb_v_temp.size] = \
                                                            perturb_v_temp[:perturb_v_full.size - perturb_start_idx]
                    perturb_w_full = np.zeros_like(pos_w[-1])
                    perturb_w_full[perturb_start_idx:perturb_start_idx + perturb_w_temp.size] = \
                                                            perturb_w_temp[:perturb_w_full.size - perturb_start_idx]
                    
                    perturb_v.append(perturb_v_full); perturb_w.append(perturb_w_full)
                    action_v.append((pos_v[-1] - perturb_v_full).clip(-gain_v[-1], gain_v[-1]) / gain_v[-1])
                    action_w.append((pos_w[-1] - perturb_w_full).clip(-gain_w[-1], gain_w[-1]) / gain_w[-1])
                else:
                    pos_v.append(trial_data['ForwardV'].values.clip(-gain_v[-1], gain_v[-1]))
                    pos_w.append(trial_data['AngularV'].values.clip(-gain_w[-1], gain_w[-1]))
                    perturb_v_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_w_gauss.append(np.zeros(round(self.SAMPLING_RATE)))
                    perturb_start_time.append(np.nan)
                    perturb_v.append(np.zeros_like(pos_v[-1])); perturb_w.append(np.zeros_like(pos_w[-1]))
                    action_v.append(pos_v[-1] / gain_v[-1])
                    action_w.append(pos_w[-1] / gain_w[-1])
                

                trial_data['Time'] -= trial_data['Time'].iloc[0]
                time.append(trial_data['Time'].values)
                trial_dur.append(trial_data['Time'].values[-1])
                
                
                # target position is analog in BCM data, I bin target channels
                # and find the mode of bins
                targetx_bins = np.arange(my_floor(trial_data['FireflyX'].min(), 1),
                                         my_ceil(trial_data['FireflyX'].max(), 1), 0.1)
                targetx_idxes = np.digitize(trial_data['FireflyX'], targetx_bins)
                targetx_hist, _ = np.histogram(trial_data['FireflyX'], targetx_bins)
                try:
                    tar_x = trial_data['FireflyX'][
                                    targetx_idxes == targetx_hist.argmax()+1].mean()
                except: # when start_idx == end_idx, they are bad trials that not matter
                    tar_x = trial_data['FireflyX'].mean()

                targety_bins = np.arange(my_floor(trial_data['FireflyY'].min(), 1),
                                         my_ceil(trial_data['FireflyY'].max(), 1), 0.1)
                targety_idxes = np.digitize(trial_data['FireflyY'], targety_bins)
                targety_hist, _ = np.histogram(trial_data['FireflyY'], targety_bins)
                try:
                    tar_y = trial_data['FireflyY'][
                                    targety_idxes == targety_hist.argmax()+1].mean()
                except:
                    tar_y = trial_data['FireflyY'].mean()

                target_x.append(tar_x)
                target_y.append(- tar_y + self.y_offset)
                tar_rho, tar_phi = cart2pol(target_x[-1], target_y[-1])
                target_r.append(tar_rho)
                target_theta.append(np.rad2deg(tar_phi))

                relative_r, relative_ang = get_relative_r_ang(
                                pos_x[-1], pos_y[-1], head_dir[-1], target_x[-1], target_y[-1])
                relative_radius.append(relative_r)
                relative_angle.append(np.rad2deg(relative_ang))
                relative_radius_end.append(relative_r[-1])
                relative_angle_end.append(np.rad2deg(relative_ang[-1]))
                rewarded.append(relative_r[-1] < reward_boundary)

                #juice_time = marker_data['time'][marker_data['key'] == marker_memo['juice']]
                #j_marker = np.where(np.logical_and(juice_time > start_marker_times[trial_idx],
                #               juice_time < end_marker_times[trial_idx]))[0]

                # Categorize trials
                # Note that few targets in BCM data are out of the distribution
                # for unknown reason, I just label and ignore them.
                if target_r[-1] < target_r_range[0] - target_tolerance or\
                   target_r[-1] > target_r_range[1] + target_tolerance or\
                   target_theta[-1] < target_theta_range[0] - target_tolerance or\
                   target_theta[-1] > target_theta_range[1] + target_tolerance:
                    category.append('wrong_target')
                else:
                    if rewarded[-1]:
                        category.append('normal')
                    else:
                        if trial_data['ForwardV'].size < skip_threshold and\
                           pos_r_end[-1] < skip_r_threshold:
                            category.append('skip')
                        elif trial_data['ForwardV'].size > lazy_threshold:
                            category.append('lazy')
                        elif relative_r[-1] > crazy_threshold:
                            category.append('crazy')
                        else:
                            category.append('normal')

        # Construct a dataframe   
        self.monkey_trajectory = pd.DataFrame().assign(gain_v=gain_v, gain_w=gain_w, 
                                 perturb_vpeakmax=perturb_vpeakmax, perturb_wpeakmax=perturb_wpeakmax,
                                 perturb_sigma=perturb_sigma, perturb_dur=perturb_dur,
                                 perturb_vpeak=perturb_vpeak, perturb_wpeak=perturb_wpeak,
                                 perturb_start_time=perturb_start_time,
                                 perturb_start_time_ori=perturb_start_time_ori,
                                 perturb_v_gauss=perturb_v_gauss, perturb_w_gauss=perturb_w_gauss,
                                 perturb_v=perturb_v, perturb_w=perturb_w,
                                 floor_density=floor_density, pos_x=pos_x,
                                 pos_y=pos_y, head_dir=head_dir, head_dir_end=head_dir_end,
                                 pos_r=pos_r, pos_theta=pos_theta, pos_r_end=pos_r_end,
                                 pos_theta_end=pos_theta_end, pos_v=pos_v, pos_w=pos_w, 
                                 target_x=target_x, target_y=target_y, target_r=target_r,
                                 target_theta=target_theta, full_on=full_on, rewarded=rewarded,
                                 relative_radius=relative_radius, relative_angle=relative_angle,
                                 time=time, trial_dur=trial_dur, 
                                 action_v=action_v, action_w=action_w, 
                                 relative_radius_end=relative_radius_end,
                                 relative_angle_end=relative_angle_end, category=category)




# if __name__ == "__main__":
#     df=load_monkey_data('data')
#     monkey_action_correlation(df)
#     monkey_action_distribution(df)




'''


some concern about using the data where human stop - move = stop - move. 
when i try the task, i would do that to probe the gain of the task. so, when i move for a short time (it would be around 1 dt), i predict how far i would move and observe the optical flow. when i stop, im not expecting to reach the target, but to give myself some time to process the previous infomation (to estimate the gain, such that improve my next prediction).
as a result, the stop period could confuse the model. luckly its usually very short, and got diluted by the move period. if this is important, we can clean the data and throw those trials out.

some side thought about learning
so when first doing this task, we are 'fully' rely on the obs.
besides integrate obs when updating the belief, we also use obs to refine our prediction.
lets assume no prediction. obs is certered around truth, so subjet would stop around truth, at least in forward direction.
now assume gain is learned from belief. so it will be dblief/dt/action.
if this is true, then we will still have the stops center aroudn the truth
but this is not true. we do see stop distribution is biased.
so why is that?
'''

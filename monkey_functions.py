import pickle
from PIL.Image import ROTATE_90
import torch
import pandas
import numpy as np
from numpy import pi
import warnings
from matplotlib import pyplot as plt
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


def down_sampling(data_array, orignal_dt, new_dt, continues_data=True):
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
    




# if __name__ == "__main__":
#     df=load_monkey_data('data')
#     monkey_action_correlation(df)
#     monkey_action_distribution(df)



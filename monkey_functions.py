import pickle
import torch
import pandas
import numpy as np
from numpy import pi
import warnings
from matplotlib import pyplot as plt
# testing imports



def load_monkey_data(filename, partial_obs=True):

    infile = open('monkey_data/{}.pkl'.format(filename),'rb')
    df = pickle.load(infile)
    # throw out full obs trials
    if partial_obs:
        df=df[df.isFullOn==False]
    return df


def monkey_trajectory(df,new_dt=0.1, goal_radius=65,factor=0.005):
    #------prepare saving vars-----
    states = [] # true location
    actions = [] # action
    tasks=[] # vars that define an episode

    orignal_dt=get_frame_rate(df, default_rate=0.0012)

    index=0
    while index<=df.ep.iloc[-1]:
        try:
            # state=[
            #     [convert_unit(y,factor=factor),convert_unit(x,factor=factor)] for x, y in zip(down_sampling(df.px[index], orignal_dt, new_dt),down_sampling(df.py[index], orignal_dt, new_dt))
            #     ]
            xs=convert_unit(torch.tensor(down_sampling(df.py[index], orignal_dt, new_dt)), factor=factor)
            xs=xs-xs[0]
            ys=-convert_unit(torch.tensor(down_sampling(df.px[index], orignal_dt, new_dt)), factor=factor)
            # plt.plot(convert_unit(torch.tensor(down_sampling(df.py[index], orignal_dt, new_dt)), factor=factor)
            ys=ys-ys[0]
            # ,-convert_unit(torch.tensor(down_sampling(df.px[index], orignal_dt, new_dt)), factor=factor)
            # )
            hs=torch.stack(down_sampling((pi/180*torch.tensor(df.p_heading[index])), orignal_dt, new_dt)).float()-pi/4
            vs=convert_unit(torch.stack(down_sampling((torch.tensor(df.real_v[index])), orignal_dt, new_dt)),factor=factor).float()
            ws=-pi/180*torch.stack(down_sampling((torch.tensor(df.real_w[index])), orignal_dt, new_dt)).float()
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
            task=[[ convert_unit(df.FFY[index],factor=factor),convert_unit(df.FFX[index],factor=factor)],convert_unit(goal_radius,factor=factor)]
            action=[[convert_unit(v),-convert_unit(w)] for v, w in zip(down_sampling(df.real_v[index], orignal_dt, new_dt),down_sampling(df.real_w[index], orignal_dt, new_dt))]
            states.append(state)
            actions.append(action)
            tasks.append(task)
            index+=1
        except KeyError:
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
        ax1.set_ylim([-30,200])
        ax1.set_yticks(np.linspace(-30,200,10))
        ax1.set_xlim([-100,100])
        ax1.axis('equal')
        
        ax2 = fig.add_subplot(132)
        ax2.set_xlabel('time, 0.0012 s')
        ax2.set_ylabel('forward v, cm/s')
        ax2.set_ylim([-50,200])
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



# if __name__ == "__main__":
#     df=load_monkey_data('data')
#     monkey_action_correlation(df)
#     monkey_action_distribution(df)



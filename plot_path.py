# imports and define agent name

from stable_baselines import DDPG
import policy_torch
from torch import nn
import matplotlib.pyplot as plt
from Config import Config
from FireflyEnv import ffenv
import numpy as np 
from numpy import pi
import matplotlib.pyplot as plt
agent_name="DDPG_selu_skip1000000_9 26 9 38"
num_episode=2
arg=Config()
arg.std_range=[1e-4,1e-3,1e-4,1e-3]

env=ffenv.FireflyEnv(arg)
baselines_selu = DDPG.load(agent_name)
torch_model_selu = policy_torch.copy_mlp_weights(baselines_selu,layers=[256,256,64,32],act_fn=nn.functional.selu)
torch_model_selu.name='selu'

# baselines_relu = DDPG.load("DDPG_theta")
# torch_model_relu = policy_torch.copy_mlp_weights(baselines_relu,layers=[32,64])
# torch_model_relu.name='relu'

agent=torch_model_selu

# create saving vars
all_ep=[]


# for ecah episode,
for i in range(num_episode):
    ep_data={}
    ep_vt=[]
    ep_wt=[]
    ep_xt=[0.]
    ep_yt=[0.]
    ep_facing=[0.]
    ep_bxt=[0.]
    ep_byt=[0.]
    ep_br=[]
    ep_ba=[]
    # get goal position at start
    belief=env.reset()[0]
    r0=belief.tolist()[0]
    a0=-belief.tolist()[1]
    goalx=r0*np.sin(a0)
    goaly=r0*np.cos(a0)
    ep_data['goalx']=goalx
    ep_data['goaly']=goaly
    ep_data['a0']=a0
    ep_data['r0']=r0  


    # log the actions raw, v and w
    while not env.stop:
        action=agent(belief)
        belief=env(action)[0][0]
        ep_vt.append(action.tolist()[0]*arg.DELTA_T)
        ep_wt.append(action.tolist()[1]*arg.DELTA_T)
        ep_data['vt']=ep_vt
        ep_data['wt']=ep_wt
        ep_facing.append(ep_facing[-1]+ep_wt[-1])
        # convert the actions to x y, with start location as 0,0
        dx=ep_vt[-1]*np.sin(ep_facing[-1])
        dy=ep_vt[-1]*np.cos(ep_facing[-1])
        ep_xt.append(ep_xt[-1]+dx)
        ep_yt.append(ep_yt[-1]+dy)

        br=belief.tolist()[0]
        ba=-belief.tolist()[1]
        ep_br.append(br)
        ep_ba.append(ba)
        ep_bxt.append(-(br*np.sin(sum(ep_wt)+ba))+goalx)
        ep_byt.append((-br*np.cos(sum(ep_wt)+ba))+goaly)



        ep_data['xt']=ep_xt
        ep_data['yt']=ep_yt
        ep_data['bxt']=ep_bxt
        ep_data['byt']=ep_byt
        ep_data['brt']=ep_br
        ep_data['bat']=ep_ba
        # log the theta, etc for filtering 
        ep_data['theta']=env.theta.tolist()

    # save episode data dict to all data

    all_ep.append(ep_data)

# plot the actions
for i in range(1):
    plt.figure
    ep_xt=all_ep[i]['xt']
    ep_yt=all_ep[i]['yt']
    plt.plot(ep_xt,ep_yt,'ro-')
    plt.plot(all_ep[i]['bxt'],all_ep[i]['byt'],'bo-')
    plt.scatter(all_ep[i]['goalx'],all_ep[i]['goaly'])
    plt.savefig('path.png')

print('s')


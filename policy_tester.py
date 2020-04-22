from stable_baselines import DDPG
import policy_torch
from torch import nn
import matplotlib.pyplot as plt

def myhist(time_log):
    cts=[]
    for i in range(12):
        ct=0
        for t in time_log:
            if t==i:
                ct+=1
        cts.append(ct)
    return cts

# load baseline model
my_selu=act_fn=(lambda x, : 1.05070098*nn.functional.elu(x,alpha=1.67326324))
baselines_selu = DDPG.load("DDPG_selu2")
torch_model_selu = policy_torch.copy_mlp_weights(baselines_selu,layer1=128,layer2=128,act_fn=nn.functional.selu)
torch_model_selu.name='selu'

baselines_relu = DDPG.load("DDPG_theta")
torch_model_relu = policy_torch.copy_mlp_weights(baselines_relu,layer1=32,layer2=64)
torch_model_relu.name='relu'

# testing, get sample beliefs
from Config import Config
from FireflyEnv import ffenv
arg=Config()
env=ffenv.FireflyEnv(arg)

correct_count=0
time_log=[]
correct_time_log=[]
for i in range(999):
    belief=env.reset()[0]
    while not env.stop:
        action=torch_model_selu(belief)
        print(action)
        belief=env(action)[0][0]
        if env.rewarded:
            print('==========')
            correct_count+=1
            correct_time_log.append(env.time.item())
        if env.stop:
            time_log.append(env.time.item())
        # print(i,env.time)
        # print(belief)
        # print(action.tolist(), torch_model_selu(belief).tolist())
print('torch correct counts',correct_count)
fig, (ax1,ax2) = plt.subplots(1, 2,sharey=True)
ax1.set_title('all episode')
ax2.set_title('rewarded episode')
ax1.set(xlabel='episode length', ylabel='number of episodes')
ax2.set(xlabel='episode length')
ax1.plot(myhist(time_log))
print(myhist(correct_time_log))
print(myhist(time_log))
ax2.plot(myhist(correct_time_log))
plt.savefig('selu torch.png')




correct_count=0
time_log=[]
correct_time_log=[]
for i in range(999):
    belief=env.reset()[0]
    while not env.stop:
        action,_=baselines_selu.predict(belief.view(1,-1))
        belief=env(action)[0][0]
        if env.stop and env.time<11:
            correct_count+=1
            correct_time_log.append(env.time.item())
        if env.stop:
            time_log.append(env.time.item())
        # print(i,env.time)
        # print(belief)
        # print(action.tolist(), torch_model_selu(belief).tolist())
print('tf correct counts',correct_count)
fig, (ax1,ax2) = plt.subplots(1, 2,sharey=True)
ax1.set_title('all episode')
ax2.set_title('rewarded episode')
ax1.set(xlabel='episode length', ylabel='number of episodes')
ax2.set(xlabel='episode length')
ax1.plot(myhist(time_log))
ax2.plot(myhist(correct_time_log))
plt.savefig('selu tf.png')



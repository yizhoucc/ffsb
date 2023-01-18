from matplotlib.pyplot import xlabel
from stable_baselines3.td3.policies import MlpPolicy
import torch
from TD3_torch import TD3
import time
from stable_baselines3.common.noise import NormalActionNoise
from FireflyEnv import ffacc_real

action_noise = NormalActionNoise(mean=0., sigma=float(0.2))
env=ffacc_real.Smooth1dCrossFade()
modelname=None
# modelname='mdp1d_100000_10_21_18_27'

if modelname is None:
    model = TD3(MlpPolicy,
        env,
        buffer_size=int(1e6),
        batch_size=512,
        learning_rate=3e-4,
        learning_starts= 0,
        tau= 0.005,
        gamma= 0.99,
        train_freq = 10,
        gradient_steps = -1,
        n_episodes_rollout = 1,
        action_noise= action_noise,
        optimize_memory_usage = False,
        policy_delay = 2,
        target_policy_noise = 0.2,
        target_noise_clip = 0.5,
        tensorboard_log = None,
        create_eval_env = False,
        policy_kwargs = {'net_arch':[32,32,32]},
        verbose = 0,
        seed = None,
        device = "cpu",
        )
else:
    model = TD3.load('./trained_agent/'+modelname,
        env,
        buffer_size=int(1e6),
        batch_size=512,
        learning_rate=3e-4,
        learning_starts= 1000,
        tau= 0.005,
        gamma= 0.99,
        train_freq = 10,
        gradient_steps = -1,
        n_episodes_rollout = 1,
        action_noise= action_noise,
        optimize_memory_usage = False,
        policy_delay = 100,
        target_policy_noise = 0.2,
        target_noise_clip = 0.5,
        tensorboard_log = None,
        create_eval_env = False,
        # policy_kwargs = {'net_arch':[32,32,32]},
        verbose = 0,
        seed = None,
        device = "cpu",
        )
train_time=5000
# with pretraining of 1000 trials
def solution(env):
    d=env.s[0].item()
    g=env.phi[0].item()
    r=env.phi[1].item()
    c=2*(d-0.7*r)/((env.trial_len-2)*g*env.dt)
    return c, env.trial_len-2,d
def _policy(c,T,d,env):
    da=2*c/(env.trial_len-2)
    a=env.s[1].item()
    if env.s[0]<=d/2:
        newa=a-da
    else:
        newa=a+da
    return newa

def _bestpolicy(d,env):
    da=1/16*((env.reward_amout)/2/d**0.5)**(1/3)/env.phi[0].item()
    a=env.s[1].item()
    if env.s[0]<=d/2:
        newa=a-da
    else:
        newa=a+da
    return newa

# # replay buffer training, math solved optimal policy
# t=0
# while t<train_time:
#     env.reset()
#     d=env.s[0].item()
#     r=env.phi[1].item()
#     d=d-0.7*r
#     done=False
#     while not done and t<train_time:
#         obs=env.wrap_decision_info().numpy()
#         a=_bestpolicy(d,env)
#         print(a)
#         next_obs,reward,done,_=env.step(a)
#         model.replay_buffer.add(obs=a, next_obs=next_obs, action=a, reward=reward, done=done,)
#         t+=1
# replay buffer training, simple policy
t=0
while t<train_time:
    env.reset()
    c,T,d=solution(env)
    done=False
    while not done and t<train_time:
        obs=env.wrap_decision_info().numpy()
        a=_policy(c,T,d,env)+action_noise()
        # a=1.
        next_obs,reward,done,_=env.step(a)
        model.replay_buffer.add(obs=obs, next_obs=next_obs, action=a, reward=reward, done=done,)
        t+=1
# replay buffer, test
t=0
while t<train_time:
    env.reset()
    done=False
    while not done and t<train_time:
        obs=env.wrap_decision_info().numpy()
        # a=_policy(c,T,env)
        a=np.sin(env.trial_timer*0.1)
        next_obs,reward,done,_=env.step(a)
        reward=10.
        model.replay_buffer.add(obs=obs, next_obs=next_obs, action=a, reward=reward, done=done,)
        t+=1

model.policy_delay=6
model.train(gradient_steps=3000,batch_size=512,verbose=True)
# model.save('trained_agent/pretrained_triangle')
model.learn(100)

# # supervised policy
# mu=model.actor.mu
# opt=torch.optim.Adam(mu.parameters(), lr=0.001)
# for i in range(100):
#     x=torch.rand(6)
#     y=mu(x)
#     loss=y
#     print(loss)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
# for a in mu.parameters():
#     print(a.grad)

# opt=torch.optim.Adam(model.actor.mu.parameters(), lr=0.001)
opt=torch.optim.Adam([{'params':model.actor.mu.parameters()}], lr=0.001)
t=0
while t<train_time:
    env.reset()
    c,T=solution(env)
    done=False
    while not done and t<train_time:
        obs=env.wrap_decision_info().numpy()
        a=_policy(c,T,env)
        _,_,done,_=env.step(a)
        a_pred=model.actor.mu(torch.tensor(obs))
        loss=(a-a_pred)**2
        # actionls.append(a_pred.clone().detach().item())
        # print(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        t+=1
# cp trained actor to target
# with torch.no_grad():
#     # zip does not raise an exception if length of parameters does not match.
#     for param, target_param in zip_strict(model.actor.mu.parameters(), model.actor_target.mu.parameters()):
#         target_param.data.mul_(0.)
#         torch.add(target_param.data, param.data, alpha=1, out=target_param.data)
# for a in model.actor_target.mu.parameters():
#     print(a)
for a,b in zip(model.actor_target.mu.parameters(), model.actor.mu.parameters()):
    print(a==b)
# visualize trained actor
actionls=[]
target_actionls=[]
env.reset()
done=False
while not done:
    obs=env.wrap_decision_info().numpy()
    a_pred=model.actor.mu(torch.tensor(obs))
    targeta=model.actor_target.mu(torch.tensor(obs))
    _,_,done,_=env.step(a_pred)
    actionls.append(a_pred.clone().detach().item())
    target_actionls.append(targeta.clone().detach().item())
    t+=1
plt.plot(actionls,'blue')
plt.plot(target_actionls,'orange')
plt.ylabel('control')
plt.xlabel('time')



for i in range(20):  
    # env.cost_scale=0.05*i+0.2
    env.cost_scale=1
    env.rewardfun_ratio=1-0.05*i
    if modelname is None:
        namestr= ("trained_agent/precf_{}_{}_{}_{}_{}".format(train_time,i,
    str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
    ))
    else:
        namestr= ("trained_agent/{}_{}_{}_{}_{}".format(modelname,i,
        str(time.localtime().tm_mday),str(time.localtime().tm_hour),str(time.localtime().tm_min)
        ))
    model.learn(train_time)
    model.save(namestr)

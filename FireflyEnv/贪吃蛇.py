

from gym import spaces, Env
import torch
import numpy as np
from numpy import pi

class MultiFF(Env):

    def __init__(self,arg=None) -> None:
        arg=arg if arg is not None else {
            'vgain':1.,
            'wgain':0.7,
            'dt':0.1,
            'reward_amount':1,

        }
        self.world_size=1
        self.n_firefly=10
        self.reso=10
        self.reward_amount=arg['reward_amount']
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=low, high=high,shape=(self.reso,self.reso),dtype=np.int)

    def step(self,action=None,debug_info={},done=False):
        self.trial_timer+=1
        action=torch.tensor(action)
        self.state_step(action)
        reward=self.calculate_reward()
        self.wrap_decision_info()
        if self.countff()==0:
            done=True
            print(self.trial_timer)
        return self.decision_info, reward, done, debug_info

    def state_step(self,action):
        '''
        0 stay
        1 up
        2 down
        3 left
        4 right
        '''

        if action==3:
            dx=-1
        elif action==4:
            dx=1
        else:
            dx=0

        if action==1:
            dy=1
        elif action==2:
            dy=-1
        else:
            dy=0        

        self.agentx+=dx
        self.agenty+=dy
        self.clamp2edge()
        self.state2image()

    def clamp2edge(self,):
        self.agentx=torch.clamp(self.agentx,0, self.reso-1)
        self.agenty=torch.clamp(self.agenty,0, self.reso-1)
        

    def state2image(self,):
        for i in range(self.s.shape[0]):
            for j in range(self.s.shape[1]):
                if self.s[i,j]==2:
                    self.s[1,j]-=2.
        x,y=int(self.agentx),int(self.agenty)
        self.s[x,y]+=2.

    def reset(self):
        self.trial_timer=0
        self.s=torch.zeros(self.reso,self.reso)
        for i in range(self.n_firefly):
            self.add_ff()
        i=0
        while i<=1:
            agentxy=torch.rand(2)
            self.agentx,self.agenty=agentxy[0],agentxy[1]
            agentx,agenty=int(self.agentx//0.1),int(self.agenty//0.1)
            if self.s[agentx,agenty]==1:
                continue
            else:
                self.s[agentx,agenty]=2.
                i+=1 
        self.wrap_decision_info()
        return self.decision_info
    
    def calculate_reward(self,):
        reward=0.
        if self.reached_target():
            reward=self.reward_amount
            # print('reward', reward, 'nff', self.countff())
        return reward

    def wrap_decision_info(self,):
        self.decision_info=self.s

    def reached_target(self,state=None,):
        reached=False
        for i in range(self.s.shape[0]):
            for j in range(self.s.shape[1]):
                if self.s[i,j]==3:
                    self.s[i,j]-=1
                    # self.add_ff()
                    reached=True
        return reached

    def add_ff(self,):
        count=self.countff()
        while count < self.n_firefly:
            xy=torch.rand(2)
            x,y=xy[0],xy[1]
            x,y=int(x//0.1),int(y//0.1)
            if self.s[x,y]==0:
                self.s[x,y]+=1.
                count+=1 

    def countff(self,):
        count=0
        for i in range(self.s.shape[0]):
            for j in range(self.s.shape[1]):
                if self.s[i,j]==1:
                    count+=1
        return count

from stable_baselines3 import DQN
env=MultiFF()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000, log_interval=50)
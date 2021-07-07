

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
            'reward_amount':100,

        }
        self.world_size=1.
        self.n_firefly=10
        self.reso=10
        self.vgain=arg['vgain']
        self.wgain=arg['wgain']
        self.dt=arg['dt']
        self.reward_amount=arg['reward_amount']
        low=-np.inf
        high=np.inf
        self.action_space = spaces.Box(low=0., high=1.,shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high,shape=(self.reso,self.reso),dtype=np.int)

    def step(self,action=None,debug_info={},done=False):
        action=torch.tensor(action)
        self.state_step(action)
        reward=self.calculate_reward()
        self.wrap_decision_info()
        return self.decision_info, reward, done, debug_info

    def state_step(self,action):
        dx=torch.cos(self.agentheading)*action[0]*self.vgain*self.dt
        dy=torch.sin(self.agentheading)*action[0]*self.vgain*self.dt
        self.agentx+=dx.item()
        self.agenty+=dy.item()
        self.agentheading+=action[1]*self.wgain*self.dt
        self.clamp2edge()
        self.state2image()

    def clamp2edge(self,):
        self.agentx=torch.clamp(self.agentx,1e-6,1-1e-6)
        self.agenty=torch.clamp(self.agenty,1e-6,1-1e-6)
        self.agentheading=torch.clamp(self.agentheading,1e-6,2*pi-1e-6)
        

    def state2image(self,):
        for i in range(self.s.shape[0]):
            for j in range(self.s.shape[1]):
                if self.s[i,j]==2:
                    self.s[1,j]-=2.
        x,y=int(self.agentx//0.1),int(self.agenty//0.1)
        self.s[x,y]+=2.

    def reset(self):
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
        self.agentheading=torch.zeros(1).uniform_(0,2*pi)
        self.wrap_decision_info()
        return self.decision_info
    
    def calculate_reward(self,):
        reward=0.
        if self.reached_target():
            reward=self.reward_amount
            print('reward', reward, 'nff', self.countff())
        return reward

    def wrap_decision_info(self,):
        self.decision_info=self.s

    def reached_target(self,state=None,):
        reached=False
        for i in range(self.s.shape[0]):
            for j in range(self.s.shape[1]):
                if self.s[i,j]==3:
                    self.s[i,j]-=1
                    self.add_ff()
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
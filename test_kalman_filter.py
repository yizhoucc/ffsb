import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 

class Test(object):
    
    def __init__(self,initial_state,
    process_noise=np.asarray([0.1,0.1]),
    observation_noise=np.asarray([0.1,0.1]),
    process_gain=[100.,100.],
    observation_gain=[100.,100.],
    dt=0.1):

        super().__init__()
        self.process_noise=process_noise
        self.observation_noise=observation_noise
        self.process_gain=process_gain
        self.observation_gain=observation_gain
        self.P=np.eye(5)*1e-8
        self.dt=0.1
        self.I = np.eye(5)
        self.Q = np.zeros((5, 5))
        self.Q[-2:, -2:] = np.diag(np.square(self.process_noise*self.dt))
        self.R = np.diag(np.square(self.observation_noise*self.dt))
        self.H = np.zeros((2, 5))
        self.H[:, -2:] = np.diag(self.observation_gain)
        self.s=initial_state
    
    def state_dynamic_noiseless(self,action):

        v = action[0]  # action for velocity
        w = action[1]  # action for angular velocity

        # noise = (self.process_noise) * np.random.rand(2)
        noise = [0,0]
        vel = self.process_gain[0] * v + noise[0]
        ang_vel = self.process_gain[1] * w + noise[1]


        s=self.s.copy()
        s[0] = self.s[0] + vel * np.cos(self.s[2]) * self.dt
        s[1] = self.s[1] + vel * np.sin(self.s[2]) * self.dt
        s[2]=self.s[2] + ang_vel * self.dt
        s[3]=vel
        s[4]=ang_vel

        return s

    def state_dynamic_predict(self):

        v = self.s[3]  # action for velocity
        w = self.s[4]  # action for angular velocity

        # noise = (self.process_noise) * np.random.rand(2)

        s=self.s.copy()
        s[0] = self.s[0] + v * np.cos(self.s[2]) * self.dt
        s[1] = self.s[1] + v * np.sin(self.s[2]) * self.dt
        s[2]=self.s[2] + w * self.dt
        s[3]=vel
        s[4]=ang_vel

        return s

    def observe(self):
        
        observation=self.s[-2:].copy()
        noise = (self.observation_noise) * np.random.normal(0, 1, 2)
        observation[0] = self.observation_gain[0] * observation[0] + noise[0]
        observation[1] = self.observation_gain[1] * observation[1] + noise[1]

        return observation

    def KF(self,action,observation=None,A=None):

        observation=self.observe() if observation is None else observation
        # prediction
        s_ = (self.state_dynamic_noiseless(action)).transpose()
        A = self.A_(action) if A is None else A(action)
        P_ = A@(self.P)@(A.transpose()) + self.Q 

        # update
        error = observation - self.H@self.s
        S = self.H@(P_)@(self.H.transpose()) + self.R 
        K = P_@(self.H.transpose())@(np.linalg.inv(S)) 
        I_KH = self.I - K@(self.H)
        print(K)
        self.s = s_ + K@(error)
        self.P = I_KH@(P_)

        return 0

    def A_(self,action):
        # go v then turn w
        v=action[0]
        w=action[1]
        A = np.zeros((5, 5))
        A[:3, :3] = np.eye(3)
        # partial dev v
        A[0,3]=np.cos(self.s[2])*self.process_gain[0]* self.dt
        A[1,3]=np.sin(self.s[2])*self.process_gain[0]* self.dt
        # partial dev theta
        A[0, 2] = - v * np.sin(self.s[2]) * self.dt*self.process_gain[0]
        A[1, 2] = v * np.cos(self.s[2]) * self.dt*self.process_gain[0]
        # partial dev w
        A[2,4]=self.dt*self.process_gain[1]

        return A

    # def A_w(self,action):
    #     # turn w then go v
    #     v=action[0]
    #     w=action[1]
    #     A = np.zeros((5, 5))
    #     A[:3, :3] = np.eye(3)
    #     A[0, 2] = - v * np.sin(self.s[2]) * self.dt*self.process_gain[0]
    #     A[1, 2] = v * np.cos(self.s[2]) * self.dt*self.process_gain[0]
    #     A[2,4]=w*self.process_gain[1]

        # return A

    # def A_arc(self,action):
    #     # treat v and w as an arc, the most precise
    #     v=action[0]
    #     w=action[1]
    #     A = np.zeros((5, 5))
    #     A[:3, :3] = np.eye(3)
    #     # partial dev v
    #     A[0,3]= np.cos(self.s[2])*self.process_gain[0]
    #     A[1,3]=np.sin(self.s[2])*self.process_gain[0]
    #     # partial dev w
    #     A[0,4]= np.cos(self.s[2])*self.process_gain[0]
    #     A[1,4]=np.sin(self.s[2])*self.process_gain[0]
    #     # partial dev theta
    #     A[0, 2] = - v * np.sin(self.s[2]) * self.dt*self.process_gain[0]
    #     A[1, 2] = v * np.cos(self.s[2]) * self.dt*self.process_gain[0]
    #     A[2,4]=w*self.process_gain[1]

    #     return A


# if __name__ == "__main__":

state=np.asarray([0,0,0,0,0])
#     # goal center, relative angle 0 and agent at (0,0) facing goal
action=np.asarray([1,0])
#     # go forward, not turning
task=Test(state,
    process_noise=np.asarray([10.,10.]),
    observation_noise=np.asarray([10.,10.]))

#     x=[]
#     y=[]
#     for i in range(10):
#         task.KF(action)
#         x.append(task.s[0])
#         y.append(task.s[1])
#         print(task.s)
#         print(task.P[1,1])
    
def plot_path(task,steps,state=None,action=None):
    state=np.asarray([0,0,0,0,0]) if state is None else state
    action=np.asarray([1,0]) if action is None else action
    x=[]
    y=[]
    for i in range(steps):
        task.KF(action)
        x.append(task.s[0])
        y.append(task.s[1])
    plt.plot(x,y)

# def plot_uncertainty(task,steps,state=None,action=None):
#     state=np.asarray([0,0,0,0,0]) if state is None else state
#     action=np.asarray([1,0]) if action is None else action
#     xy_cov=[]
#     fig=plt.figure(figsize=(4,40))
#     counter=0
#     while counter < steps:
#         # ax=plt.subplot(steps,1,counter)
#         counter+=1
#         task.KF(action)
#         xy_cov.append(task.P[:2,:2])
#         ax=fig.add_subplot(steps,1,counter)
#         ax.contourf(task.P[:2,:2])

def plot_uncertainty_t(task,state=None,action=None):
    # cp and run the below code, or change it to save to dir
    task.KF(action)
    x = np.linspace(-10,10,500)
    y = np.linspace(-10,10,500)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal([0, 0], task.P[:2,:2])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Prob')
    plt.show()
    print(task.P[:2,:2])

import numpy as np
from matplotlib import pyplot as plt

# linear quadratic estimator (Kalman filter)
def lqe(A, C, V, W, X0):
    horizon = len(A)
    for t in range(horizon):
        V[t] = V[t] @ V[t].T
        W[t] = W[t] @ W[t].T
    P = [np.zeros([2,2])] # this is P_0
    S = []
    Kf = []
    for t in range(horizon):
        Kf.append(P[t] @ C[t].T @ np.linalg.pinv(C[t] @ P[t] @ C[t].T + W[t]))
        S.append(P[t] - Kf[t] @ C[t] @ P[t])
        if t < horizon - 1:
            P.append(A[t] @ S[t] @ A[t].T + V[t]) # this is P_t+1
    return Kf, P, S



# linear quadratic regulator
# def lqr(A, B, Q, R):
#     horizon = len(A)
#     Kr = [0] # a placeholder, wont be used.
#     P = [Q[horizon - 1]] # last Q
#     for t in reversed(range(horizon)):
#         # this is -Kr actually
#         Kr.append(-np.linalg.inv(R[t] + B[t].T @ P[horizon - (t + 1)] @ B[t])\
#              @ B[t].T @ P[horizon - (t + 1)] @ A[t])
#         if t > 0:
#             P.append(A[t].T@P[horizon - (t + 1)]@A[t] - \
#                     A[t].T@P[horizon - (t + 1)]@B[t]@\
#                     np.linalg.inv(B[t].T@P[horizon - (t + 1)]@B[t] + R[t])\
#                      @B[t].T@P[horizon - (t + 1)]@A[t]+Q[t])
#             # P.append(A[t].T * P[horizon - (t + 1)] * (A[t] + B[t] * Kr[horizon - t]) + Q[t])
#         # print(list(reversed(P)))
#     return list(reversed(Kr)), list(reversed(P))

def lqr(A, B, Q, R):
    horizon = len(A)
    Kr = list(range(horizon-1))
    P = list(range(horizon))
    P[-1] = [Q[-1]] # last Q
    for t in range(horizon-1):
        p=P[horizon-t-1]
        P[horizon-t-2]=A[t].T@p@A[t] - \
                    A[t].T@p@B[t]@\
                    np.linalg.inv(B[t].T@p@B[t] + R[t])@B[t].T@p@A[t]+Q[t]
        Kr[horizon-t-2]=-np.linalg.inv(R[t] + B[t].T @ p @ B[t])@ B[t].T @ p @ A[t]
    return Kr, P
# A=[np.array([[1,0.1],[0,0.8]])]*100
# B=[np.array([[0],[0.1]])]*100
# Q=[np.array([[1,0.],[0,1]])]*100
# R=[np.ones((1,1))*1]*100
# K,_=lqr(A,B,Q,R)

# K_list=[k[0][0] for k in K]
# x_list=[]
# x=np.array([[0],[0]])
# xf=np.array([[1],[0]])
# u_list=[]
# for i in range(99):
#     u=K_list[i]@(x-xf)
#     x=A[i]@x+(B[i]@u).reshape(1,-1).T
#     x_list.append(x)
#     u_list.append(u)
# pos=[x[0,0] for x in x_list]
# plt.plot(pos)
# v=[x[1,0] for x in x_list]
# plt.plot(v)

N=30
A=[np.array([[1,0.1,0,0],[0,0.7,0,0],[0,0,0,0],[0,0,-1,0]])]*N
B=[np.array([[0],[0.3],[1],[1]])]*N
Q=[np.array([[1,0,0,0],[0,0.01,0,0],[0,0,0.0000001,0],[0,0,0,0.1]])]*N
R=[np.ones((1,1))*0.1]*N

K,_=lqr(A,B,Q,R)

K_list=[k[0][0] for k in K]
x_list=[]
x=np.array([[0],[0],[0],[0]])
xf=np.array([[1],[0],[0],[0]])
u_list=[]
for i in range(N-1):
    u=K_list[i]@(x-xf)
    x=A[i]@x+(B[i]@u).reshape(1,-1).T
    x_list.append(x)
    u_list.append(u)
pos=[x[0,0] for x in x_list]
plt.plot(pos)
v=[x[1,0] for x in x_list]
plt.plot(v)
u=[u[0] for u in u_list]
plt.plot(u)


# transforms time-invariant into constant time-varying (list of) matrices
def tvar(var, horizon, idx=None):
    if not type(var) is list:
        temp = []
        if not idx:
            for t in range(horizon):
                temp.append(var)
        else: # return a 0 matrix
            for t in range(horizon):
                temp.append(np.matrix(np.zeros(np.shape(var))))
            temp[idx] = var
        var = temp
    return var


# main LQG class definition
class LQG:

    def __init__(self, horizon):
        self.horizon = horizon
        self.var = {'A': None, 'B': None, 'C': None,
                    'P': None, 'Q': None, 'R': None,
                    'V': None, 'W': None,
                    'x_p': None, 'x_e': None,'final_state':None}

    def define(self, string, val):
        for char in string:
            if char in 'ABCRVW':
                self.var[char] = tvar(val, self.horizon)
            elif char == 'Q': 
                self.var['Q'] = tvar(val, self.horizon, -1)
            elif char == 'X':
                self.var['X0'] = tvar(val, 1)

    def assign_val(self, string, val):
        for char in string:
            if char == 'A': 
                self.var['A'] = tvar(val[string.find('A')], self.horizon)
            elif char == 'B': 
                self.var['B'] = tvar(val[string.find('B')], self.horizon)
            elif char == 'C': 
                self.var['C'] = tvar(val[string.find('C')], self.horizon)
            elif char == 'R': 
                self.var['R'] = tvar(val[string.find('R')], self.horizon)
            elif char == 'V': 
                self.var['V'] = tvar(val[string.find('V')], self.horizon)
            elif char == 'W': 
                self.var['W'] = tvar(val[string.find('W')], self.horizon)
            elif char == 'Q': 
                self.var['Q'] = tvar(val[string.find('Q')], self.horizon, -1)
            elif char == 'X':
                self.var['x_p'] = tvar(val[string.find('X')], 1)
            elif char == 'F':
                self.var['final_state'] = val[string.find('F')]

    def kalman(self):
        self.var['Kf'], self.var['x_p'], self.var['S'] = \
            lqe(self.var['A'], self.var['C'], self.var['V'], self.var['W'], self.var['x_p'])

    def control(self):
        self.var['Kr'], self.var['P'] = \
            lqr(self.var['A'], self.var['B'], self.var['Q'], self.var['R'])

    def sample(self, n=1, x_p=None, x=None, u=None, v=None, w=None):
        y = None

        a = np.shape(self.var['A'][0])

        if self.var['x_p'] is None:
            self.var['x_p'] = [np.matrix(np.zeros(np.shape((a, a))))]

        if not self.var['C'] is None:
            c = np.shape(self.var['C'][0])
            self.kalman()
            obs = True
        else:
            obs = False

        if not self.var['B'] is None:
            self.control() # get Kr and P
            ctrl = True
        else:
            ctrl = False

        if x_p is None:  # default zero
            x_p = [np.matrix(np.zeros((a[0], n)))]
        elif min(np.shape(x_p)) == 1 or type(x_p) is list:  # provided mean
            x_p = np.reshape(np.matrix(x_p), (max(np.shape(x_p)), 1))
            x_p = [np.tile(x_p, (1, n))]
        assert np.shape(x_p[0]) == (a[0], n)

        if x is None:  # sample initial error
            e = np.random.randn(a[0], n)
            x = [x_p[0] + self.var['x_p'][0] * e]
        else:  # provided initial states
            assert np.shape(x) == (a[0], n)
            x = [x]
            e = np.linalg.pinv(self.var['x_p'][0]) * (x[0] - x_p[0])

        if w is None and obs:
            w = [np.random.randn(c[0], n)]
            for t in range(self.horizon - 1):
                w.append(np.random.randn(c[0], n))
        if v is None:
            v = [np.random.randn(a[0], n)]
            for t in range(self.horizon - 1):
                v.append(np.random.randn(a[0], n))

        # time 0
        if obs:
            y = [self.var['C'][0] * x[0] + self.var['W'][0] * w[0]] 
            x_e = [x_p[0] + self.var['Kf'][0] * (y[0] - self.var['C'][0] * x_p[0])]
        else:
            x_e = [x_p[0]]
        if u is None and ctrl:
            if self.var['final_state'] is None:
                u = [self.var['Kr'][0] * x_e[0]]
            else:
                u = [self.var['Kr'][0] * (x_e[0]-self.var['final_state'])]

        # time to n
        for t in range(self.horizon - 1):
            if ctrl:
                x_p.append(self.var['A'][t] * x_e[t] + self.var['B'][t] * u[t])
                x.append(self.var['A'][t] * x[t] + self.var['B'][t] * u[t] + self.var['V'][t] * v[t])
            else:
                x_p.append(self.var['A'][t] * x_e[t])
                x.append(self.var['A'][t] * x[t] + self.var['V'][t] * v[t])
            if obs:
                y.append(self.var['C'][t] * x[t] + self.var['W'][t] * w[t])
                x_e.append(x_p[t] + self.var['Kf'][t] * (y[t] - self.var['C'][t] * x_p[t]))
            else:
                x_e.append(x_p[t])
            if ctrl:
                if len(u) <= t + 1:
                    if self.var['final_state'] is None:
                        u.append(self.var['Kr'][t] * x_e[t])
                    else:
                        u.append(self.var['Kr'][t] * (x_e[t]-self.var['final_state']))

        noise = {'x': e, 'v': v, 'w': w}
        kf = {'x_p': x_p, 'x_e': x_e}
        data = {'x': x, 'y': y, 'u': u, 'kf': kf, 'noise': noise, 'cost': {}}

        return data, kf, noise


if __name__ == "__main__":
    totaltime=30
    system = LQG(totaltime)  # initialize LQG system with time horizon T = 5
    # system.define('ABCQRVWX', np.matrix(np.eye(2)))  # define matrices

    system.assign_val('ACQRVWXF', [
        np.matrix([[1,0.1],[0,0.4]]),            #a transition matrix
        # np.matrix([[0],[0.5]]),                  #b control gain
        np.matrix([[0,1]]),                     #c obs gain
        np.matrix([[10,0],[0,1]]),          #q state cost weight
        np.matrix([1e-8]),                      #r control cost weight
        np.matrix([[0],[1]])*1,                #v control noise (system noise)
        np.ones(1)*1,                       #w obs noise
        np.matrix([[0],[0]]),                  # init state uncertainty
        np.matrix([[1],[0]]),                # Final state
    ])
    data, kf, noise=system.sample(1, x_p=np.array([[0],[0]]),x=np.array([[0],[0]]) )  # simulate 10 system runs
    # print(data)



    b=[]
    for i in range (totaltime):
        b.append(data['x'][i][0,0])
    plt.plot(b,'b')
    b=[]
    for i in range (totaltime):
        b.append(kf['x_e'][i][0,0])
    plt.plot(b,'y')


# b=[]
# for i in range (50,100):
#     b.append(data['y'][i][0,0])
# plt.plot(b,'r')

# tp=list(range(totaltime))
# a1=[]
# a2=[]
# for d in range(10):
#     a1.append(sum([data['y'][i][0,0]*data['x'][i][1,0] for i in tp[:-d]]))
#     a2.append(sum([data['y'][i-d][0,0]*data['x'][i][1,0] for i in tp[:-d]]))
# plt.plot(a1,'y')
# plt.plot(a2,'r')


# b=[]
# for i in range (totaltime):
#     b.append(data['u'][i][0,0])
# plt.plot(b)


b=[]
for i in range (totaltime):
    b.append(kf['x_e'][i][0,0])
plt.plot(b,'y')

# b=[]
# for i in range (totaltime):
#     b.append(kf['x_p'][i][1,0])
# plt.plot(b)

# b=[]
# for i in range (totaltime):
#     b.append(kf['x_e'][i][0,0]-data['x'][i][0,0])
# plt.plot(b)
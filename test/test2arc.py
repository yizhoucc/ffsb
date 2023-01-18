import numpy as np
import math
from scipy.optimize import minimize

def fun(args):
    vm,wm=args
    T=lambda x: ((x[3]-x[0]*x[2]*(math.sin(1.5)-math.sin(1.5-wm*x[2]))/wm)*x[1]/wm/2/math.sin(1.5-wm*x[2]))+x[2]
    return T

def con(args):
    vmin,vmax,wmin,wmax,tsmin,tsmax,dmin,dmax=args
    cons=( 
    {'type':'ineq','fun':lambda x:x[0]-vmin},
    {'type':'ineq','fun':lambda x:-x[0]+vmax},
 
    {'type':'ineq','fun':lambda x:x[1]-wmin},
    {'type':'ineq','fun':lambda x:-x[1]+wmax},

    {'type':'ineq','fun':lambda x:x[2]-tsmin},
    {'type':'ineq','fun':lambda x:-x[2]+tsmax},
    {'type':'ineq','fun':lambda x:x[3]-dmin},
    {'type':'ineq','fun':lambda x:-x[3]+dmax}
    )
    return cons

arg1=(0.025,0.02) # vm and wm
arg2=(0,0.025,    -0.02,0.02,     20,40,     1,1.2)
# v w ts d
cons=con(arg2)

x0=np.asarray((0.0025,0.0,20,1.1))
res=minimize(fun(arg1),x0,method='SLSQP',constraints=cons)
print('best x:',res.x)


# will generate a d vs theta heatmap of T

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
class TestArc():

    def __init__(self,vm,wm,kwargs=None):


        kwargs={} if kwargs is None else kwargs
        self.vm=vm # max velocity
        self.wm=wm
        self.all_cases=None
        self.d_lables=None # distances
        self.theta_lables=None # thetas
        self.compute_case_table(**kwargs)

    def compute_case_table(self, thetarange=None, drange=None):
        thetarange=[-pi/6,pi/6] if thetarange is None else thetarange
        drange =[1, 4]  if drange is None else drange
        theta_resolution=pi/60
        d_resolution=0.1

        theta_lables=[thetarange[0]]
        d_lables=[drange[0]]
        while theta_lables[-1]+theta_resolution<=thetarange[-1]:
            theta_lables.append(theta_lables[-1]+theta_resolution)
        while d_lables[-1]+d_resolution<=drange[-1]:
            d_lables.append(d_lables[-1]+d_resolution)

        self.all_cases=(theta_lables,d_lables)
        self.d_lables=d_lables
        self.theta_lables=theta_lables


    def compute_time_table(self):
        time_table=np.zeros( (len(self.theta_lables),len(self.d_lables)) )
        action_table=np.zeros( (len(self.theta_lables),len(self.d_lables),2) )

        for i,theta in enumerate(self.theta_lables):
            for j,d in enumerate(self.d_lables):
                v,w=self.choose_action(d,theta)
                T=self.calculate_time(d,theta,w)
                time_table[i,j]=T
                action_table[i,j]=[v,w]
        self.time_table=time_table
        self.action_table=action_table
        return time_table

    def plot_cases_polar(self):
        pass

    def plot_cases_dtheta(self):
        pass


    def choose_action(self,d,theta,vm=None,wm=None):
        vm=self.vm if vm is None else vm
        wm=self.wm if wm is None else wm
        
        if d>(vm*2*np.sin(theta)/wm):
            v=vm
            w=vm*2*np.sin(theta)/d
            if d<-vm*2*np.sin(theta)/wm:
                w=-wm
                v=-wm*d/2/np.sin(theta)
        elif d==vm*2*np.sin(theta)/wm:
            v=vm
            w=wm
        elif d<vm*2*np.sin(theta)/wm:
            w=wm
            v=wm*d/2/np.sin(theta)
        else:
            print('wrong!@')
        if vm/2/wm/np.sin(theta) ==np.inf:
            v=vm
            w=0
        return v,w


    def calculate_time(self,d,theta,w):
        try:
            return 2*theta/w
        except ZeroDivisionError:
            return d/self.vm


# test=TestArc(1,1)
# test.compute_time_table()

# # plt.imshow(test.time_table)
# plt.contourf(test.time_table)
# # plt.imshow(test.action_table[:,:,0])
# # plt.imshow(test.action_table[:,:,1])

class TurnthenGo(TestArc):
    
    # everything is same, except for choosing action and measuring time
    # note, in choosing action, we keep the action for the arc part.
    # the stop and turning part is alaways wm and v=0

    def __init__(self,vm,wm,kwargs=None):
        super().__init__(vm,wm,kwargs=None)

    def choose_action(self,d,theta,vm=None,wm=None):
        vm=self.vm if vm is None else vm
        wm=self.wm if wm is None else wm
        
        if d>(vm*2*np.sin(theta)/wm):
            v=vm
            w=vm*2*np.sin(theta)/d
            if d<-vm*2*np.sin(theta)/wm:
                # one theta limiting case
                v=0
                w=0
                # not assigning values here, 
                # as the values dont need to be calculated
                # just v=0 w=wm for 1 step,
                # v=vm, w=vm for 2 step.
        elif d==vm*2*np.sin(theta)/wm:
            v=vm
            w=wm
        elif d<vm*2*np.sin(theta)/wm:
            # the other theta limiting case
            v=0
            w=0
            # assign 0 for debug use
        else:
            print('wrong!@')
        if vm/2/wm/np.sin(theta) ==np.inf:
            v=vm
            w=0
        return v,w

    def calculate_time(self,d,theta,w):
        try:
            if d<self.vm*2*np.sin(theta)/self.wm:
                return (theta+np.arcsin(d*self.wm/2/self.vm))/self.wm
            elif d<-self.vm*2*np.sin(theta)/self.wm:
                return (-theta+np.arcsin(d*self.wm/2/self.vm))/self.wm
            else:
                return 2*theta/w

        except ZeroDivisionError:
            # the stright forward casef
            return d/self.vm
 
test=TestArc(0.2,pi/4)
test.compute_time_table()
plt.contourf(test.time_table)
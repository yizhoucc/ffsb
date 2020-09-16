# will generate a d vs theta heatmap of T

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi

 
def plot_polar_contour(values, azimuths, zeniths):

    theta = azimuths
    zeniths = np.array(zeniths)
 
    values = np.array(values)
    values = values.reshape(len(azimuths), len(zeniths))
 
    r, theta = np.meshgrid(zeniths, (azimuths))
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    cax = ax.contourf(theta, r, values, 30)

    cb = fig.colorbar(cax)
    cb.set_label("Pixel reflectance")
 
    return fig, ax, cax

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


class TestRewardP(TurnthenGo):

    def __init__(self,vm,wm, dt=0.1, goal_r=0.2, nv=0.1, nw=0.1, wb='w',  kwargs=None):
        super().__init__(vm,wm,kwargs=None)
        # wabers law, assuming noise is propotional to w
        self.nv=nv
        self.nw=nw
        self.waber_law=wb
        self.dt=dt
        self.goal_r=goal_r


    def calculate_prob(self,T,v,w):
        # each step, we have:
        # cov = R@cov@RT + Q
        # weber instead of waber
        cov=np.zeros((2,2))
        if self.waber_law=='w':
            Q=np.diag([self.nv*self.vm, self.nw*abs(w)])
        elif self.waber_law=='vw':
            Q=np.diag([self.nv*v, self.nw*abs(w)])
        else:
            Q=np.diag([self.nv*self.vm, self.nw*self.wm])

        R=np.array([[np.cos(w*self.dt),-np.sin(w*self.dt)],[np.sin(w*self.dt),np.cos(w*self.dt)]])
        for i in range(int(np.ceil(T/self.dt))):
            # actually we only need cov=cov+Q. rotation dosent matter since goal is circle and we go to the center.
            # cov=R@cov@R.transpose()+Q
            cov=cov+Q
        # assuming goal is another gaussian
        # assuming goal r is the 1st std
        gaussian_cov=cov+np.diag([self.goal_r,self.goal_r])
        peak=1/2/pi/np.sqrt(np.linalg.det(gaussian_cov))
        return peak



    def compute_prob_table(self):
        prob_table=np.zeros( (len(self.theta_lables),len(self.d_lables)) )
        time_table=np.zeros( (len(self.theta_lables),len(self.d_lables)) )
        action_table=np.zeros( (len(self.theta_lables),len(self.d_lables),2) )

        for i,theta in enumerate(self.theta_lables):
            for j,d in enumerate(self.d_lables):
                v,w=self.choose_action(d,theta)
                # discrete time steps
                T=self.calculate_time(d,theta,w)
                P=self.calculate_prob(T,v,w)

                prob_table[i,j]=P
                time_table[i,j]=T
                action_table[i,j]=[v,w]

        self.time_table=time_table
        self.action_table=action_table
        self.prob_table=prob_table
        return prob_table


 
# test=TestArc(0.2,pi/4)
# test.compute_time_table()
# plt.contourf(test.time_table)

test=TestRewardP(0.3,pi/4,wb='vw',nv=0.1,nw=0.001,dt=0.01)
test.compute_prob_table()
fig1, ax2 = plt.subplots(constrained_layout=True)
CS=ax2.contourf(test.time_table)
cbar = fig1.colorbar(CS)

fig1, ax2 = plt.subplots(constrained_layout=True)
CS=ax2.contourf(test.prob_table)
cbar = fig1.colorbar(CS)

fig1, ax2 = plt.subplots(constrained_layout=True)
CS=ax2.contourf(test.prob_table*0.9**test.time_table)
cbar = fig1.colorbar(CS)

plt.jet()
plot_polar_contour(test.time_table,test.theta_lables,test.d_lables)
plot_polar_contour(test.prob_table*0.9**test.time_table,test.theta_lables,test.d_lables)

import numpy as np
import matplotlib.pyplot as plt
from plot_ult import add_colorbar

# note, actual task
'''
vmax, 200cm/s
tau, log norm,  lN(0.58, 0.4^2)
test value, tau = 1.8s
distance, 400cm
T=8.5s
'''

def a_(tau, dt=0.1):
    return np.exp(-dt/tau)

def next_v(v,a,b,control=1):
    return a*v+b*control

def s_(tau, T=7,dt=0.1):
    s=T+tau*np.log(0.5*(1+np.exp(-T/tau)))
    return s

def d_(tau, vm,T=7,dt=0.1):
    return 2*tau*vm*np.log(np.cosh(T/2/tau))

def acc(tau, vm):
    d=0
    v=0
    s=s_(tau) 
    v_list=[]
    # t=np.linspace(0,s,30)
    # for tt in t:
    #     v=vm*(1-np.exp(-tt/tau))
    #     d=d+v
    #     v_list.append(v)
    vs=vm*(1-np.exp(-s*dt/tau))
    d=vm*(s*dt+np.exp(-s*dt/tau -1)*tau-np.exp(-0/tau -1)*tau)
    return vs,d

def deacc(tau, vm, d, T=70):
    s=s_(tau) 
    vs,ds=acc(tau, vm)
    v=(-1+(vs/vm+1)*np.exp(-(T-s)/tau))*vm
    d=ds
    return v, d
    # v_list=[]
    # t=np.linspace(s, T, 30)
    # for tt in t:
    #     v=-1+(vs/vm_1)*np.exp(-(tt-s)/tau)
    #     v_list.append(v)

def actual_d_(tau, vm, T=7,dt=0.1):
    a=a_(tau)
    v=0
    v_list=[0]
    d=0
    b=vm*(1-a)
    s=s_(tau)
    s=int(np.floor(s/dt))
    for i in range(s):
        v=next_v(v,a,b)
        v_list.append(v)
        d=d+v*dt
    for i in range(int(T/dt-s)):
        v=next_v(v,a,b,control=-1)
        v_list.append(v)
        d=d+v*dt
    return d

def get_vm(tau, b):
    a=a_(tau)
    return b/(1-a)

def get_vm_(tau, x=2, T=7):
    vm=x/2/tau *(1/(np.log(np.cosh(T/2/tau))))
    return vm

def b_(tau, dt=0.1):
    return get_vm_(tau)*(1-a_(tau))

tau=0.0281
b_(tau)

def getx(a, vm=1, dt=0.1, T=7):
    b=vm*(1-a)
    totalt=T/dt
    s=int(totalt/2)
    lowerlimit=0
    upperlimit=totalt

    ve=99
    while not (ve>-0.1 and ve<0.1):
        vi=0
        for i in range(int(s)):
            vi=vi*a+b
        for i in range(int(totalt-s)):
            vi=vi*a-b
        ve=vi
        
        if ve<-0.1:
            lowerlimit=s
            s=round((s+upperlimit)/2)
        elif ve>0.1:
            upperlimit=s
            s=round((s+lowerlimit)/2)
        if upperlimit-lowerlimit<2:
            break
        # print(s)
    x=0
    vs=[0]
    for i in range(s):
        vi=vs[-1]*a+b
        vs.append(vi)
        x=x+vi*dt
    for i in range(70-s):
        vi=vs[-1]*a-b
        vs.append(vi)
        x=x+vi*dt

    param_dict={
        'v max':vm,
        'dt':dt,
        'max trial time':T,
    }

    return x, param_dict

def compute_tau_range(vm_range, d=1.5, T=7, tau=1.8):
    vm=get_vm_(tau)
    while vm > vm_range[0]:
        tau=tau-0.1
        vm=get_vm_(tau)
    lowertau=tau
    while vm < vm_range[-1]:
        tau=tau+0.1
        vm=get_vm_(tau)
    uppertau=tau
    # take 2 place after .
    lowertau=lowertau//0.1*0.1
    uppertau=uppertau//0.1*0.1
    return [lowertau, uppertau]

# ----tau to vm to tau
tau_list=np.linspace(0.1,8,40)
vm_list=[get_vm_(tau) for tau in tau_list]
vm_range=[vm_list[0],vm_list[-1]]
compute_tau_range(vm_range, d=1.5, T=7, tau=1.8)

tau=1.8
vm=get_vm_(tau)
while vm > vm_range[0]:
    tau=tau-0.1
    vm=get_vm_(tau)
lowertau=tau
while vm < vm_range[-1]:
    tau=tau+0.1
    vm=get_vm_(tau)
uppertau=tau


# ---- vm given tau
tau_list=np.linspace(0,8,40)
vm_list=[0*tau for tau in tau_list]
for i, tau in enumerate(tau_list):
    vm_list[i]=get_vm_(tau)
fig = plt.figure(figsize=[18, 9])
ax = fig.add_subplot()
ax.plot(tau_list,vm_list)
ax.set_title('vm vs tau')
ax.set_xlabel('tau')
ax.set_ylabel('vm')

# ---- use the vm to calcualte b
tau_list=np.linspace(0,8,40)
vm_list=[get_vm_(tau) for tau in tau_list]
d_list=[d_(tau, vm) for tau, vm in zip(tau_list, vm_list)]
b_list=[vm*(1-np.exp(-0.1/tau)) for tau, vm in zip(tau_list, vm_list)]
fig = plt.figure(figsize=[18, 9])
ax = fig.add_subplot()
ax.plot(tau_list, b_list)
ax.set_title('b vs tau')
ax.set_xlabel('tau')
ax.set_ylabel('b')

# ----with b, solve for tau
vm_list=[]
tau_list=np.linspace(0,3,40)
b_list=np.linspace(0.25, 1, 20)
d_array=np.zeros((len(b_list), len(tau_list)))
vm_array=np.zeros((len(b_list), len(tau_list)))
for i, b in enumerate(b_list):
    for j, tau in enumerate(tau_list):
        vm=get_vm(tau, b)
        vm_array[i,j]=vm
        d=d_(tau,vm)
        d_array[i,j]=d if d >1.5 else 0
        d_array[i,j]=d if d >1.5 and vm< 20 else 0

fig = plt.figure(figsize=[18, 9])
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

img=ax.imshow(d_array,extent=[tau_list[0],tau_list[-1],b_list[0],b_list[-1]])
add_colorbar(img)
ax.set_title('distance')
ax.set_xlabel('tau')
ax.set_ylabel('b')
img=ax2.imshow(vm_array,extent=[tau_list[0],tau_list[-1],b_list[0],b_list[-1]])
add_colorbar(img)
ax2.set_title('vm')
ax2.set_xlabel('tau')
ax2.set_ylabel('b')


# ----with vm, solve for tau
tau_list=np.linspace(0,3,80)
vm_list=np.linspace(0.25,1,50)
d_array=np.zeros((len(vm_list), len(tau_list)))
b_array=np.zeros((len(vm_list), len(tau_list)))
for i, vm in enumerate(vm_list):
    for j, tau in enumerate(tau_list):
        d=d_(tau,vm)
        d_array[i,j]=d if d >1.5 else 0
        b_array[i,j]=vm*(1-a_(tau))

fig = plt.figure(figsize=[18, 9])
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

img=ax.imshow(d_array,extent=[tau_list[0],tau_list[-1],vm_list[0],vm_list[-1]])
add_colorbar(img)
ax.set_title('distance')
ax.set_xlabel('tau')
ax.set_ylabel('vm')

img=ax2.imshow(b_array,extent=[tau_list[0],tau_list[-1],vm_list[0],vm_list[-1]])
add_colorbar(img)
ax2.set_title('b')
ax2.set_xlabel('tau')
ax2.set_ylabel('vm')

# ---------------------------------------------
# for given a, bi sect for switch control point
# return the distance travel.


# for given a, vm,  calculate max distance


a_list=np.linspace(0,1,100)
x_list=[]
for a in a_list:
    d,param_dict=getx(a)
    x_list.append(d)

fig = plt.figure(figsize=[16, 9])
ax = fig.add_subplot()
ax.plot(a_list, x_list)
ax.set_title('max distance under different alpha, fixed vm', fontsize=20)
ax.set_xlabel('alpha velocity scalar in acc control')
ax.set_ylabel('max distance agent could travel')
ax.text(0,0,'{}'.format(str(['{}:{}'.format(a, param_dict[a]) for a in param_dict])[1:-1]), fontsize=15)


# for given vm, return max a
def geta(vm, dt=0.1, T=7, target_d=3):
    lowera=0
    uppera=0.99
    a=(lowera+uppera)/2
    
    while uppera-lowera>0.02:
        d,_=getx(a, vm=vm)
        if d>target_d+0.05:
            lowera=a
            a=(lowera+uppera)/2
        elif d<target_d-0.05:
            uppera=a
            a=(lowera+uppera)/2
        else:
            return a
        print(a, lowera, uppera)
    return a



vm_list=np.linspace(0.25,1,100)
a_list=[]
for vm in vm_list:


    d,_=getx(a)
    x_list.append(d)


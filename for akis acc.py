'''
tau
a = exp(-dt/tau)
b = vm_(tau)*(1-a)
newv = a*prev+ u*gain*b

def get_vm_(tau, d=2, T=7):
    vm=d/2/tau *(1/(np.log(np.cosh(T/2/tau))))
    return vm

def get_wm_(tau, r=pi/2, T=7):
    wm= 2*r /2/tau *(1/(np.log(np.cosh(T/2/tau))))
    return wm

if there is noise:
    noise can from:
        control noise, hand cannot control very precicesly.
            it is likely to be gaussian + to the control
        process noise of system, velocity perturbation add noise to the task, making the precices control translate to noisy state update
            it is more of a paramter than an actual noise. our task do not have a random process noise even in perturbation, becase the pert is in a pattern
        observation noise, optic flow is not a perfect source of velocity.
            it could be gaussian + to the true v, with or without a bias term or scalar term. but more likely, it could be a gassuan scaled by true v and then added to the true v.
        inference noise, process noise of internal model, when belief update there could be noise, skip for now
    the main thing here is contro noise, process noise, and obs noise.
newv = a*prev+ noisy_u*gain*b + process_noise, where u_noise is noise related to joystick control, and process_noise relate to the system update.


let noise_u=control noise, 
noisy_u=(u+noise_u)
newv = a*prev+ noisy_u*gain*b + process_noise

let noise_o=observation noise,
noise_o = noise_scale_o * v *standardpdf 
noisy_o = noise_o + v


the goal of the meeting is to discuss the theoretical background: 
how the hierarchical inference the control dynamics demand works. Make a model of that.
And also see how the internal model of the control dynamics interacts with the internal model of self-motion (equivalent to kaushik's low-speed prior I guess), and how all these interact when the subject has to adapt the relative weights between internal model and sensory input, when noise is present.
'''

'''
state->obs->        next belief  -> action -> next state
    prev belief->                   prev state->
state = [x,y,h, v,w]
control trajectory, u_s, 2xnum_ts array
for each time, 
    replace the current state[3:], with the control at time t



belief update:
    input: previous belief, action, current obs
    output: cur belief

velocity control belief update:
    input: previous belief of position, current prediction of velocity(from current action), current obs of velocity(from current state)
    computation:

        # current prediction of velocity(from current action)
        prediction_velocity_t = control_gain*u_t

        # current obs of velocity(from current state)
        observed_velocity_t = noise_o_t + v_t

        # integrate
        noise_o_t, noise_p_t # they are variance
        estimation_velocity_t = prediction_velocity_t* noise_o_t/( noise_p_t+noise_o_t) + observed_velocity_t *noise_p_t/( noise_p_t+noise_o_t)

        return estimation_velocity_t

        cur belief = estimation_velocity_t*dt + previous belief of position
    output: cur belief


acc control belief update:
    params:
        tau
        a = exp(-dt/tau)
        newv = a*prev + u_t*gain*(1-a)

    input: 
        previous belief of position, 
        previous belief of velocity, 
        current prediction of velocity(from current action), current obs of velocity(from current state)

    computation:

        # current prediction of velocity(from current action)
        # velocity: newv = u_t*gain
        # acc: newv = a*prev + u_t*gain*(1-a)
!       prediction_velocity_t = a*prev + u_t*gain*(1-a)

        # current obs of velocity(from current state)
        observed_velocity_t = noise_o_t + v_t

        # integrate
        noise_o_t, noise_p_t # they are variance
        estimation_velocity_t = prediction_velocity_t* noise_o_t/( noise_p_t+noise_o_t) + observed_velocity_t *noise_p_t/( noise_p_t+noise_o_t)

        return estimation_velocity_t

        cur belief = estimation_velocity_t + previous belief of position

    output: cur belief

'''
from ossaudiodev import control_labels
import numpy as np
import torch
dt=0.1
tau_a=0.4

# state = [d, v]

A = torch.zeros(2,2)
A[0,0] = 1 # distant always integrates
# partial dev with v
A[0, 1] = dt # how much v comes into the d
A[1, 1] = tau_a # how much v presist

I = torch.eye(2)
H = torch.zeros(1,2)
H[-1,-1] = 1

# predict belief -----------------

# apply the velocity part of the state, to move the distnace
predicted_b = self.update_state(previous_b)
# apply the new control, to update the state velocity
predicted_b = self.apply_action(predicted_b,a,latent=True)
# update the ucnertainty of both d and v, by running the dynamic
predicted_P = self.A@(previous_P)@(self.A.t()) +self.Q 


# update belief ---------------------
# computet the prediction err
error = obs - H@predicted_b 
# convert the predicted uncertainty, into a diction that we can observe
# s is equvalient to (process noise var + obs noise var)
S = H@(predicted_P)@(H.t()) + self.R 
# but this R, obser noise, can be a function of velocity, it can also have bias

# compute the kalman gain, from the prediction err
# K = process noise var /  (process noise var + obs noise var)
K = predicted_P@(H.t())@(torch.inverse(S)) 
# do the update for the belief mean
b = predicted_b + K@(error)
# do the update for the belief uncertainty
# I_kH = (obs noise var /  (process noise var + obs noise var))
I_KH = I - K@(H)
# new uncertainty = (obs noise var/(process noise var + obs noise var)) * previous uncertainty
# if only one step: new uncertainty = obs noise var*process noise var/(process noise var + obs noise var))
P = I_KH@(predicted_P)

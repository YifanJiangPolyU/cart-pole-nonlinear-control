#
#                O ball: m = 1 kg
#               /
#              /
#             /  pole: R = 1 m
#            /
#     ______/_____
#    |            | Cart: M = 4 kg
#    |____________|
#      O        O
#
#

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

G =  9.8 # acceleration due to gravity, in m/s^2
R = 1.0  # length of the pole (m)
M = 4.0  # mass of the cart (kg)
m = 1  # mass of the ball at the end of the pole (kg)

#control input: u = k1*theta + k2*theta.dot
def control(q):
	#linear quadratic feedback control designed using matlab
	k1 = 140.560
	k2 = -3.162
	k3 = 41.772
	k4 = -8.314
	u = -(k1*(q[0]-pi) + k2*q[1] + k3*q[2] + k4*q[3])
	return u

#state: q = transpose([theta, x, d(theta)/dt, dx/dt])
def derivs(q, t):

	dqdt = np.zeros_like(q)

	#control input (to be added
	u = control(q)

	delta = m*sin(q[0])**2 + M

	dqdt[0] = q[2]
	dqdt[1] = q[3]

	dqdt[2] = - m*(q[2]**2)*sin(q[0])*cos(q[0])/delta  \
			  - (m+M)*G*sin(q[0])/delta/R  \
			  - u*cos(q[0])/delta/R

	dqdt[3] = m*R*(q[2]**2)*sin(q[0])/delta   \
			 + m*R*G*sin(q[0])*cos(q[0])/delta/R  \
			 + u/delta

	return dqdt

# create a time array from 0..100 sampled at 0.1 second steps
dt = 0.1
t = np.arange(0.0, 20, dt)

rad = np.pi/180

#initial conditions
theta = 180*rad 
x = 0.0
w = 1
xdot = 0.0

# initial state
state = np.array([theta, x, w, xdot])

# integrate your ODE using scipy.integrate.
y = integrate.odeint(derivs, state, t)


#frequency analysis
fs = np.fft.fft(y[:,0])
frange = np.fft.fftfreq(len(t),d = dt)

#animation generation
x1 = y[:,1]
y1 = 0.0

x2 = R*sin(y[:,0]) + x1
y2 = -R*cos(y[:,0]) + y1

fig = plt.figure()
ax = fig.add_subplot(121, autoscale_on=False, aspect='equal',\
					 xlim=(-3, 3), ylim=(-3, 3))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    thisx = [x1[i], x2[i]]
    thisy = [y1, y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template%(i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
    interval=30, blit=True, init_func=init)

#frequency domain plot
#ax = fig.add_subplot(132,xlim=(-1,1))
#ax.grid()
#ax.plot(frange, fs)

#time domain plot
ax = fig.add_subplot(122)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.grid()
ax.plot(t,y[:,1])

#ani.save('cart-pole-LQR.mp4', fps=20)

plt.show()


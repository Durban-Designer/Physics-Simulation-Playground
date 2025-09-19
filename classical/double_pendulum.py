"""
Simulation of a double pendulum, a classic example of chaotic motion.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
G = 9.81  # Acceleration due to gravity (m/s^2)
L1 = 1.0  # Length of the first pendulum arm (m)
L2 = 1.0  # Length of the second pendulum arm (m)
M1 = 1.0  # Mass of the first pendulum bob (kg)
M2 = 1.0  # Mass of the second pendulum bob (kg)

def derivs(t, y):
    """Calculate the derivatives of the system state.

    y = [theta1, omega1, theta2, omega2]
    theta1: angle of the first pendulum
    omega1: angular velocity of the first pendulum
    theta2: angle of the second pendulum
    omega2: angular velocity of the second pendulum
    """
    theta1, omega1, theta2, omega2 = y

    dydt = np.zeros_like(y)
    dydt[0] = omega1
    dydt[2] = omega2

    delta = theta2 - theta1
    den1 = (M1 + M2) * L1 - M2 * L1 * np.cos(delta) * np.cos(delta)
    dydt[1] = (M2 * L1 * omega1 * omega1 * np.sin(delta) * np.cos(delta) +
               M2 * G * np.sin(theta2) * np.cos(delta) +
               M2 * L2 * omega2 * omega2 * np.sin(delta) -
               (M1 + M2) * G * np.sin(theta1)) / den1

    den2 = (L2 / L1) * den1
    dydt[3] = (-M2 * L2 * omega2 * omega2 * np.sin(delta) * np.cos(delta) +
               (M1 + M2) * G * np.sin(theta1) * np.cos(delta) -
               (M1 + M2) * L1 * omega1 * omega1 * np.sin(delta) -
               (M1 + M2) * G * np.sin(theta2)) / den2

    return dydt

# Initial state: [theta1, omega1, theta2, omega2]
# Both pendulums hanging vertically, with a small initial velocity
y0 = [np.pi / 2, 0.0, np.pi, 0.0]

# Time array
t_max = 40.0
dt = 0.05
t = np.arange(0, t_max, dt)

# Solve the ODE
sol = solve_ivp(derivs, [0, t_max], y0, t_eval=t, dense_output=True)
y = sol.sol(t)

# Convert angles to Cartesian coordinates
x1 = L1 * np.sin(y[0])
y1 = -L1 * np.cos(y[0])
x2 = x1 + L2 * np.sin(y[2])
y2 = y1 - L2 * np.cos(y[2])

# Set up the figure and animation
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '-', lw=1, ms=0.5)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    line.set_data(thisx, thisy)
    trace.set_data(x2[:i], y2[:i])
    time_text.set_text(time_template % (i * dt))
    return line, trace, time_text

ani = animation.FuncAnimation(
    fig, animate, len(y[0]), interval=dt*1000, blit=True)

print("Starting double pendulum animation. Close the plot window to continue.")
plt.show()

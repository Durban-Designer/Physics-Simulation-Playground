"""
Simulation of the 1D wave equation, showing wave propagation and reflection.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
N = 200  # Number of spatial points
c = 1.0  # Wave speed
dx = 1.0  # Spatial step
dt = 0.5  # Time step (Courant condition: c * dt / dx <= 1)

# Initialize grid
u = np.zeros(N)
u_prev = np.zeros(N)
u_next = np.zeros(N)

# Initial condition (a Gaussian pulse)
x = np.arange(N)
u = np.exp(-0.01 * (x - N/4)**2)
u_prev = u.copy()

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot(u)
ax.set_ylim(-1.1, 1.1)

def animate(i):
    global u, u_prev, u_next
    # Update using the finite difference formula for the wave equation
    for j in range(1, N - 1):
        u_next[j] = 2 * u[j] - u_prev[j] + (c * dt / dx)**2 * (u[j+1] - 2*u[j] + u[j-1])

    # Update history
    u_prev[:] = u[:]
    u[:] = u_next[:]

    # Boundary conditions (fixed ends)
    u[0] = 0
    u[-1] = 0

    line.set_ydata(u)
    return line,

print("Creating 1D wave equation animation...")
ani = animation.FuncAnimation(fig, animate, frames=500, interval=20, blit=True)

try:
    ani.save('wave_equation.gif', writer='pillow', fps=30)
    print("Animation saved to wave_equation.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

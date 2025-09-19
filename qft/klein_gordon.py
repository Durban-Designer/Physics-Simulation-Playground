"""
Simulation of the 1D Klein-Gordon equation for a scalar quantum field.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
N = 200  # Number of spatial points
m = 0.5  # Mass of the field quanta
dx = 1.0
dt = 0.5

# Initialize fields (phi, and its time derivative pi)
phi = np.zeros(N)
pi = np.zeros(N)
phi_prev = np.zeros(N)

# Initial condition (a Gaussian pulse)
x = np.arange(N)
phi = np.exp(-0.01 * (x - N/2)**2)
phi_prev = phi.copy()

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot(phi)
ax.set_ylim(-1.1, 1.1)
ax.set_title('Klein-Gordon Scalar Field')

def animate(i):
    global phi, phi_prev, pi
    # Update using the finite difference formula
    laplacian = np.roll(phi, 1) + np.roll(phi, -1) - 2 * phi
    phi_next = 2 * phi - phi_prev + (dt/dx)**2 * laplacian - dt**2 * m**2 * phi
    
    # Update history
    phi_prev = phi.copy()
    phi = phi_next.copy()

    line.set_ydata(phi)
    return line,

print("Creating Klein-Gordon animation...")
ani = animation.FuncAnimation(fig, animate, frames=500, interval=20, blit=True)

try:
    ani.save('klein_gordon.gif', writer='pillow', fps=30)
    print("Animation saved to klein_gordon.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

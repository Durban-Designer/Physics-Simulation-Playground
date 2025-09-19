"""
Simulation of a 1D quantum wave packet tunneling through a potential barrier.
This uses the Finite-Difference Time-Domain (FDTD) method.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
N = 1000  # Number of spatial points
L = 1.0  # Length of the spatial domain
dx = L / N  # Spatial step
dt = 0.000005  # Time step

# Wave packet parameters
x0 = L / 4  # Initial position
sigma = L / 20  # Width of the wave packet
k0 = 300 * np.pi  # Wavenumber

# Potential barrier parameters
V0 = 1.5e5  # Height of the barrier
barrier_start = L / 2 - L/40
barrier_end = L / 2 + L/40

# Create the spatial grid and potential
x = np.linspace(0, L, N)
V = np.zeros(N)
V[int(barrier_start/dx):int(barrier_end/dx)] = V0

# Initialize the wavefunction (real and imaginary parts)
psi_r = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.cos(k0 * x)
psi_i = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.sin(k0 * x)

# FDTD coefficients (assuming hbar=1, m=1 for simplicity)
C1 = dt / (2 * dx**2)
C2 = dt * V

# Set up the plot
fig, ax = plt.subplots()
line, = ax.plot(x, psi_r**2 + psi_i**2)
ax.plot(x, V / (4 * V0), 'r--', label='Potential Barrier')
ax.set_ylim(0, 1.2)
ax.set_xlabel("Position")
ax.set_ylabel("Probability Density")
ax.legend()

def animate(i):
    for _ in range(10):  # Evolve multiple steps per frame for speed
        # Update imaginary part
        psi_i[1:-1] += C1 * (psi_r[:-2] - 2 * psi_r[1:-1] + psi_r[2:]) - C2[1:-1] * psi_r[1:-1]
        # Update real part
        psi_r[1:-1] -= C1 * (psi_i[:-2] - 2 * psi_i[1:-1] + psi_i[2:]) + C2[1:-1] * psi_i[1:-1]

    prob_density = psi_r**2 + psi_i**2
    line.set_ydata(prob_density)
    return line,

print("Creating quantum tunneling animation. This may take a moment...")

try:
    ani = animation.FuncAnimation(fig, animate, frames=200, blit=True, interval=20)
    ani.save('quantum_tunneling.gif', writer='pillow', fps=30)
    print("Animation saved to quantum_tunneling.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Saving a static plot instead.")
    plt.savefig("quantum_tunneling_static.png")
    print("Static plot saved to quantum_tunneling_static.png")

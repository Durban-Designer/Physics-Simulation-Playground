"""
Visualization of quantum foam with a single photon propagating through it.
This is a conceptual visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

try:
    import cupy as cp
    GPU_ENABLED = True
    print("CuPy found, running on GPU.")
except ImportError:
    GPU_ENABLED = False
    print("CuPy not found, running on CPU.")

# Parameters
GRID_SIZE = 150

# Grid
x = np.linspace(-2, 2, GRID_SIZE)
y = np.linspace(-2, 2, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# Photon parameters
photon_energy = 0.8
photon_width = 0.2
photon_speed = 0.04

if GPU_ENABLED:
    X_gpu = cp.asarray(X)
    Y_gpu = cp.asarray(Y)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-0.5, 1.5)
ax.set_title('Quantum Foam with Photon')
ax.axis('off')

# Initial surface
if GPU_ENABLED:
    Z_gpu = cp.zeros_like(X_gpu)
    Z = cp.asnumpy(Z_gpu)
else:
    Z = np.zeros_like(X)

surface = [ax.plot_surface(X, Y, Z, cmap='viridis')]

def animate(i):
    global surface
    
    # 1. Generate quantum foam
    if GPU_ENABLED:
        foam = (cp.random.rand(GRID_SIZE, GRID_SIZE) - 0.5) * 0.1
    else:
        foam = (np.random.rand(GRID_SIZE, GRID_SIZE) - 0.5) * 0.1

    # 2. Generate and position the photon wave packet
    photon_x_pos = -2 + 4 * ((i * photon_speed) % 1.0)
    
    if GPU_ENABLED:
        photon = photon_energy * cp.exp(-((X_gpu - photon_x_pos)**2 + Y_gpu**2) / photon_width**2)
        Z_gpu = foam + photon
        Z = cp.asnumpy(Z_gpu)
    else:
        photon = photon_energy * np.exp(-((X - photon_x_pos)**2 + Y**2) / photon_width**2)
        Z = foam + photon

    # 3. Update plot
    surface[0].remove()
    surface[0] = ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, antialiased=False)
    return surface

print("Creating quantum foam with photon animation...")
ani = animation.FuncAnimation(fig, animate, frames=int(1/photon_speed), interval=50, blit=False)

try:
    ani.save('quantum_foam_with_photon.gif', writer='pillow', fps=15)
    print("Animation saved to quantum_foam_with_photon.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

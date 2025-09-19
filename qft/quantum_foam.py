"""
Visualization of quantum foam using GPU acceleration with cupy.
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
GRID_SIZE = 100

# Create grid
x = np.linspace(-1, 1, GRID_SIZE)
y = np.linspace(-1, 1, GRID_SIZE)
X, Y = np.meshgrid(x, y)

if GPU_ENABLED:
    X_gpu = cp.asarray(X)
    Y_gpu = cp.asarray(Y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(-1, 1)
ax.set_title('Quantum Foam Visualization')

# Initial surface
if GPU_ENABLED:
    Z_gpu = cp.zeros_like(X_gpu)
    Z = cp.asnumpy(Z_gpu)
else:
    Z = np.zeros_like(X)

surface = [ax.plot_surface(X, Y, Z, cmap='viridis')]

def animate(i):
    global surface
    if GPU_ENABLED:
        # Generate random fluctuations on the GPU
        fluctuations = (cp.random.rand(GRID_SIZE, GRID_SIZE) - 0.5) * 0.1
        Z_gpu = fluctuations
        Z = cp.asnumpy(Z_gpu)
    else:
        # Generate random fluctuations on the CPU
        fluctuations = (np.random.rand(GRID_SIZE, GRID_SIZE) - 0.5) * 0.1
        Z = fluctuations

    # Update the plot
    surface[0].remove()
    surface[0] = ax.plot_surface(X, Y, Z, cmap='viridis')
    return surface

print("Creating quantum foam animation...")
ani = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=False)

try:
    ani.save('quantum_foam.gif', writer='pillow', fps=15)
    print("Animation saved to quantum_foam.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

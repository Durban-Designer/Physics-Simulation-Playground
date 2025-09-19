"""
Animation of a simple plane electromagnetic wave propagating through space.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Parameters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Wave parameters
omega = 1
k = 1

# Create grid
z = np.linspace(0, 20, 100)

# Set up the plot
ax.set_xlim(0, 20)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_xlabel('Z (propagation direction)')
ax.set_ylabel('E field (x)')
ax.set_zlabel('B field (y)')
ax.set_title('Electromagnetic Wave')

E_line, = ax.plot([], [], [], 'r-', label='E field')
B_line, = ax.plot([], [], [], 'b-', label='B field')
ax.legend()

def animate(t):
    # E field (in x direction)
    Ex = np.cos(k * z - omega * t)
    Ey = np.zeros_like(z)
    
    # B field (in y direction, in phase with E)
    Bx = np.zeros_like(z)
    By = np.cos(k * z - omega * t)

    E_line.set_data(z, Ex)
    E_line.set_3d_properties(Ey)
    
    B_line.set_data(z, Bx)
    B_line.set_3d_properties(By)
    
    return E_line, B_line

print("Creating electromagnetic wave animation...")
ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, 20, 0.2), interval=50, blit=True)

try:
    ani.save('em_wave.gif', writer='pillow', fps=15)
    print("Animation saved to em_wave.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

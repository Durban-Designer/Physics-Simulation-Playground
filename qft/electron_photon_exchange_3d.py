"""
A more realistic 3D animation of two electrons interacting via a virtual photon.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Set up the plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 10)
ax.set_title('3D Virtual Photon Exchange')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Time)')
ax.view_init(elev=20., azim=-135)

# Define the electron paths (hyperbolic-like repulsion)
def get_electron_paths():
    z = np.linspace(0, 10, 100)
    # Path 1
    x1 = -2 / np.cosh(z - 5)
    y1 = 2 + 0 * z
    # Path 2
    x2 = 2 / np.cosh(z - 5)
    y2 = -2 + 0 * z
    return z, np.vstack((x1, y1)).T, np.vstack((x2, y2)).T

z_path, path1, path2 = get_electron_paths()

# Draw the full paths for context
ax.plot(path1[:, 0], path1[:, 1], z_path, 'c--', alpha=0.5)
ax.plot(path2[:, 0], path2[:, 1], z_path, 'c--', alpha=0.5)

# Artists for the animation
electron1 = ax.scatter([], [], [], s=150, c='blue', alpha=0.8)
electron2 = ax.scatter([], [], [], s=150, c='blue', alpha=0.8)
photon, = ax.plot([], [], [], 'r--', lw=2)

INTERACTION_FRAME = 50

def animate(i):
    # Update electron positions
    e1_pos = path1[i]
    e2_pos = path2[i]
    z_pos = z_path[i]
    
    electron1._offsets3d = ([e1_pos[0]], [e1_pos[1]], [z_pos])
    electron2._offsets3d = ([e2_pos[0]], [e2_pos[1]], [z_pos])

    # Animate the photon exchange around the interaction point
    if abs(i - INTERACTION_FRAME) < 10:
        # Make electrons flash
        flash_size = 300 * (1 - abs(i - INTERACTION_FRAME) / 10)
        electron1.set_sizes([150 + flash_size])
        electron2.set_sizes([150 + flash_size])
        
        # Draw the photon
        photon.set_data([e1_pos[0], e2_pos[0]], [e1_pos[1], e2_pos[1]])
        photon.set_3d_properties([z_pos, z_pos])
        photon.set_visible(True)
    else:
        electron1.set_sizes([150])
        electron2.set_sizes([150])
        photon.set_visible(False)

    return electron1, electron2, photon

print("Creating improved 3D electron-photon exchange animation...")
ani = animation.FuncAnimation(fig, animate, frames=len(z_path), interval=50, blit=False)

try:
    ani.save('electron_photon_exchange_3d.gif', writer='pillow', fps=15)
    print("Animation saved to electron_photon_exchange_3d.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

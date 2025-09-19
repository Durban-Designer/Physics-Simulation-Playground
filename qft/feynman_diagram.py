"""
A conceptual animation of a Feynman diagram.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title('Feynman Diagram: Electron-Electron Scattering')
ax.set_xticks([])
ax.set_yticks([])

# Particle paths
ax.plot([1, 4], [1, 4], 'b-', label='Electron 1')
ax.plot([1, 4], [9, 6], 'b-', label='Electron 2')
ax.plot([6, 9], [4, 1], 'b-')
ax.plot([6, 9], [6, 9], 'b-')

# Virtual photon path
virtual_photon, = ax.plot([], [], 'r--', label='Virtual Photon (γ)')

def animate(i):
    if i < 50:
        # Move particles towards interaction point
        pass # Static for this simple animation
    elif i < 100:
        # Exchange virtual photon
        end_x = 4 + (i - 50) * (2/50)
        end_y = 4 + (i - 50) * (2/50)
        virtual_photon.set_data([4, end_x], [4, end_y])
    else:
        # Particles move away
        pass # Static
    return virtual_photon,

print("Creating Feynman diagram animation...")
ani = animation.FuncAnimation(fig, animate, frames=150, interval=50, blit=True, repeat=False)

try:
    ani.save('feynman_diagram.gif', writer='pillow', fps=15)
    print("Animation saved to feynman_diagram.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

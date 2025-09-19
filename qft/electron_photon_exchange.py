"""
A conceptual animation of two electrons interacting via the exchange of a virtual photon.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_title('Virtual Photon Exchange (Electron-Electron Repulsion)')
ax.set_xticks([])
ax.set_yticks([])

# Initial positions
e1_start_pos = np.array([1.0, 7.0])
e2_start_pos = np.array([1.0, 3.0])

# Artists for the animation
electron1, = ax.plot([], [], 'o', ms=15, color='blue', label='Electron 1')
electron2, = ax.plot([], [], 'o', ms=15, color='blue', label='Electron 2')
photon, = ax.plot([], [], 'r--', lw=2, label='Virtual Photon')

ax.legend()

def animate(i):
    total_frames = 200
    
    # Phase 1: Initial state (frames 0-30)
    if i < 30:
        electron1.set_data([e1_start_pos[0]], [e1_start_pos[1]])
        electron2.set_data([e2_start_pos[0]], [e2_start_pos[1]])

    # Phase 2: Emission (frames 30-50)
    elif i < 50:
        progress = (i - 30) / 20
        # Electron 1 recoils and emits photon
        e1_pos = e1_start_pos + np.array([progress * 2, -progress * 0.5])
        electron1.set_data([e1_pos[0]], [e1_pos[1]])
        photon.set_data([e1_pos[0]], [e1_pos[1]])

    # Phase 3: Photon travel (frames 50-150)
    elif i < 150:
        progress_emission = 1.0
        e1_pos = e1_start_pos + np.array([progress_emission * 2, -progress_emission * 0.5])
        
        progress_travel = (i - 50) / 100
        photon_end_pos = e2_start_pos + np.array([progress_travel * 2, progress_travel * 0.5])
        
        photon.set_data([e1_pos[0], photon_end_pos[0]], [e1_pos[1], photon_end_pos[1]])

    # Phase 4: Absorption (frames 150-170)
    elif i < 170:
        photon.set_data([], []) # Photon is absorbed
        progress = (i - 150) / 20
        # Electron 2 recoils
        e2_pos = e2_start_pos + np.array([2, 0.5]) + np.array([progress * 2, progress * 0.5])
        electron2.set_data([e2_pos[0]], [e2_pos[1]])

    # Phase 5: Final state (frames 170-200)
    else:
        # Keep particles in their final positions
        e1_final_pos = e1_start_pos + np.array([2, -0.5])
        e2_final_pos = e2_start_pos + np.array([2, 0.5]) + np.array([2, 0.5])
        electron1.set_data([e1_final_pos[0]], [e1_final_pos[1]])
        electron2.set_data([e2_final_pos[0]], [e2_final_pos[1]])

    return electron1, electron2, photon

print("Creating electron-photon exchange animation...")
ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True, repeat=False)

try:
    ani.save('electron_photon_exchange.gif', writer='pillow', fps=15)
    print("Animation saved to electron_photon_exchange.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

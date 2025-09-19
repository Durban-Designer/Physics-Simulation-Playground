"""
1D simulation of the expansion of the universe (Hubble's Law).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
N_GALAXIES = 10
H0 = 1  # Hubble constant (simplified)

# Initial positions of galaxies
galaxies = np.linspace(1, 10, N_GALAXIES)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 2)
ax.set_yticks([])
ax.set_title('Expansion of the Universe')

points, = ax.plot(galaxies, np.ones(N_GALAXIES), 'bo', ms=10)
ax.plot(0, 1, 'ro', ms=15, label='Observer (Us)')
ax.legend()

def animate(t):
    # Expansion according to Hubble's Law: v = H0 * d
    # New distance = old distance + v * dt
    # For this animation, let's use an exponential expansion factor a(t)
    a_t = np.exp(t * 0.02)
    new_positions = galaxies * a_t
    points.set_data(new_positions, np.ones(N_GALAXIES))
    return points,

print("Creating universe expansion animation...")
ani = animation.FuncAnimation(fig, animate, frames=100, interval=100, blit=True)

try:
    ani.save('universe_expansion.gif', writer='pillow', fps=10)
    print("Animation saved to universe_expansion.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

"""
Simulation of 2D Brownian motion.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
N_PARTICLES = 50
N_STEPS = 1000
STEP_SIZE = 0.1

# Initialize particles at the center
positions = np.zeros((N_PARTICLES, 2))

# Store history for animation
positions_history = np.zeros((N_STEPS, N_PARTICLES, 2))

for i in range(N_STEPS):
    positions_history[i] = positions
    # Generate random steps for each particle
    steps = np.random.randn(N_PARTICLES, 2) * STEP_SIZE
    positions += steps

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.grid()

points, = ax.plot([], [], 'o')

def animate(i):
    points.set_data(positions_history[i, :, 0], positions_history[i, :, 1])
    return points,

print("Creating Brownian motion animation. This may take a moment...")
ani = animation.FuncAnimation(fig, animate, frames=N_STEPS, interval=30, blit=True)

try:
    ani.save('brownian_motion.gif', writer='pillow', fps=30)
    print("Animation saved to brownian_motion.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

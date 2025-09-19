"""
Simulation of an ideal gas in a 2D box, demonstrating the kinetic theory of gases.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
N_PARTICLES = 100
BOX_SIZE = 100
PARTICLE_RADIUS = 2
MAX_VELOCITY = 2

# Initialize particles
positions = np.random.rand(N_PARTICLES, 2) * BOX_SIZE
velocities = (np.random.rand(N_PARTICLES, 2) - 0.5) * MAX_VELOCITY

def update(dt):
    global positions, velocities

    # Update positions
    positions += velocities * dt

    # Handle wall collisions
    for i in range(2):
        hit_wall = (positions[:, i] < 0) | (positions[:, i] > BOX_SIZE)
        velocities[hit_wall, i] *= -1

    # Handle particle-particle collisions (simplified)
    for i in range(N_PARTICLES):
        for j in range(i + 1, N_PARTICLES):
            dist_sq = np.sum((positions[i] - positions[j])**2)
            if dist_sq < (2 * PARTICLE_RADIUS)**2:
                # Elastic collision
                v1, v2 = velocities[i], velocities[j]
                x1, x2 = positions[i], positions[j]
                velocities[i] = v1 - np.dot(v1 - v2, x1 - x2) / dist_sq * (x1 - x2)
                velocities[j] = v2 - np.dot(v2 - v1, x2 - x1) / dist_sq * (x2 - x1)

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, BOX_SIZE)
ax.set_ylim(0, BOX_SIZE)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

points, = ax.plot([], [], 'o', ms=PARTICLE_RADIUS*2)

def animate(i):
    update(1.0)
    points.set_data(positions[:, 0], positions[:, 1])
    return points,

print("Creating ideal gas animation...")
ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)

try:
    ani.save('ideal_gas.gif', writer='pillow', fps=30)
    print("Animation saved to ideal_gas.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

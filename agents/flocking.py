"""
Simulation of flocking behavior, based on a simplified Boids model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
N_BOIDS = 50
WIDTH, HEIGHT = 640, 480
MAX_SPEED = 5
PERCEPTION_RADIUS = 50
SEPARATION_DISTANCE = 20

# Initialize boids
positions = np.random.rand(N_BOIDS, 2) * [WIDTH, HEIGHT]
velocities = (np.random.rand(N_BOIDS, 2) - 0.5) * MAX_SPEED

def update_boids(positions, velocities):
    new_velocities = velocities.copy()
    for i in range(N_BOIDS):
        # Find neighbors
        distances = np.linalg.norm(positions - positions[i], axis=1)
        neighbors = (distances < PERCEPTION_RADIUS) & (distances > 0)

        if not np.any(neighbors):
            continue

        # Rule 1: Separation
        separation_neighbors = distances < SEPARATION_DISTANCE
        separation_vector = np.sum(positions[i] - positions[separation_neighbors], axis=0)

        # Rule 2: Alignment
        alignment_vector = np.mean(velocities[neighbors], axis=0)

        # Rule 3: Cohesion
        cohesion_vector = np.mean(positions[neighbors], axis=0) - positions[i]

        # Update velocity
        new_velocities[i] += separation_vector * 0.05 + (alignment_vector - velocities[i]) * 0.05 + cohesion_vector * 0.001

    # Limit speed
    speed = np.linalg.norm(new_velocities, axis=1)
    too_fast = speed > MAX_SPEED
    new_velocities[too_fast] = new_velocities[too_fast] / speed[too_fast, np.newaxis] * MAX_SPEED

    # Update positions
    positions += new_velocities

    # Boundary conditions (wrap around)
    positions[:, 0] %= WIDTH
    positions[:, 1] %= HEIGHT

    return positions, new_velocities

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 7.5))
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_aspect('equal')
ax.axis('off')

points, = ax.plot([], [], 'o', ms=4)

def animate(i):
    global positions, velocities
    positions, velocities = update_boids(positions, velocities)
    points.set_data(positions[:, 0], positions[:, 1])
    return points,

print("Creating flocking animation. This may take a moment...")
ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)

try:
    ani.save('flocking.gif', writer='pillow', fps=30)
    print("Animation saved to flocking.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

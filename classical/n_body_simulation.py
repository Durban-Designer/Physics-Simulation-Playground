"""
N-body gravitational simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Gravitational constant
G = 6.67430e-11

# Simulation parameters
N_BODIES = 4
TIME_STEP = 0.001
N_STEPS = 5000

# Initial conditions (mass, position, velocity)
# A simple system of 4 bodies
masses = np.array([1e12, 1e10, 1e10, 1e10])
positions = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
velocities = np.array([[0.0, 0.0], [0.0, 15.0], [0.0, -15.0], [-15.0, 0.0]])

def calculate_forces(masses, positions):
    n = len(masses)
    forces = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = positions[j] - positions[i]
                r_mag = np.linalg.norm(r_vec)
                force_mag = G * masses[i] * masses[j] / r_mag**2
                forces[i] += force_mag * r_vec / r_mag
    return forces

# Store history for animation
positions_history = np.zeros((N_STEPS, N_BODIES, 2))

for i in range(N_STEPS):
    positions_history[i] = positions
    forces = calculate_forces(masses, positions)
    accelerations = forces / masses[:, np.newaxis]
    velocities += accelerations * TIME_STEP
    positions += velocities * TIME_STEP

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid()

points, = ax.plot([], [], 'o')

def animate(i):
    points.set_data(positions_history[i, :, 0], positions_history[i, :, 1])
    return points,

print("Creating N-body simulation animation. This may take a moment...")
ani = animation.FuncAnimation(fig, animate, frames=N_STEPS, interval=20, blit=True)

try:
    ani.save('n_body_simulation.gif', writer='pillow', fps=30)
    print("Animation saved to n_body_simulation.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

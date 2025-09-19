"""
Simulation of a 2D random walk.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N_STEPS = 5000

# Generate steps
steps = np.random.randn(N_STEPS, 2)

# Calculate positions
positions = np.cumsum(steps, axis=0)

# Plot the walk
plt.figure(figsize=(8, 8))
plt.plot(positions[:, 0], positions[:, 1], lw=0.5)
plt.plot(positions[0, 0], positions[0, 1], 'go', label='Start')
plt.plot(positions[-1, 0], positions[-1, 1], 'ro', label='End')
plt.title("2D Random Walk")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.savefig("random_walk.png")
print("Random walk plot saved to random_walk.png")

"""
Simulation of Diffusion-Limited Aggregation (DLA).
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 200  # Size of the grid
MAX_PARTICLES = 500

# Initialize grid and place a seed at the center
grid = np.zeros((N, N), dtype=bool)
grid[N//2, N//2] = True

particle_count = 0
while particle_count < MAX_PARTICLES:
    # Start a particle at a random position on a circle
    angle = 2 * np.pi * np.random.rand()
    r = N // 2 - 5
    x, y = N//2 + int(r * np.cos(angle)), N//2 + int(r * np.sin(angle))

    # Random walk until it sticks or goes too far
    while True:
        # Move in a random direction
        dx, dy = np.random.choice([-1, 0, 1], 2)
        x, y = x + dx, y + dy

        # Check for out of bounds
        if not (0 < x < N-1 and 0 < y < N-1):
            break  # Start a new particle

        # Check for sticking
        if (grid[x-1, y] or grid[x+1, y] or grid[x, y-1] or grid[x, y+1]):
            grid[x, y] = True
            particle_count += 1
            if particle_count % 50 == 0:
                print(f"{particle_count}/{MAX_PARTICLES} particles attached.")
            break

print("DLA simulation finished.")

# Plot the result
plt.figure(figsize=(10, 10))
plt.imshow(grid, cmap='gray', interpolation='nearest')
plt.title("Diffusion-Limited Aggregation")
plt.axis('off')
plt.savefig("dla.png")
print("DLA plot saved to dla.png")

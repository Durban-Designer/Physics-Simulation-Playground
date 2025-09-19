"""
Simulation of a forest fire, a classic percolation model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# States
EMPTY = 0
TREE = 1
FIRE = 2

# Parameters
N = 100  # Grid size
p_tree = 0.6  # Probability of a cell being a tree
p_fire = 0.001 # Probability of a tree catching fire spontaneously

# Initialize grid
grid = np.random.choice([EMPTY, TREE], N*N, p=[1-p_tree, p_tree]).reshape(N, N)
grid[N//2, N//2] = FIRE  # Start a fire in the middle

cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])

def update_grid(grid):
    new_grid = grid.copy()
    for i in range(N):
        for j in range(N):
            if grid[i, j] == TREE:
                # Check for burning neighbors
                is_burning = False
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if grid[(i+di)%N, (j+dj)%N] == FIRE:
                            is_burning = True
                            break
                    if is_burning:
                        break
                if is_burning or np.random.rand() < p_fire:
                    new_grid[i, j] = FIRE
            elif grid[i, j] == FIRE:
                new_grid[i, j] = EMPTY
    return new_grid

# Set up the plot
fig, ax = plt.subplots()
img = ax.imshow(grid, cmap=cmap, interpolation='nearest')
ax.axis('off')

def animate(i):
    global grid
    grid = update_grid(grid)
    img.set_data(grid)
    return [img]

print("Creating forest fire animation. This may take a moment...")
ani = animation.FuncAnimation(fig, animate, frames=200, interval=100, blit=True)

try:
    ani.save('forest_fire.gif', writer='pillow', fps=10)
    print("Animation saved to forest_fire.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

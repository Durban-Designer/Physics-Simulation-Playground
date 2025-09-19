"""
Conway's Game of Life, a classic cellular automaton.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_grid(grid):
    """Update the grid for one time step of the Game of Life."""
    new_grid = grid.copy()
    N = grid.shape[0]
    for i in range(N):
        for j in range(N):
            # Count live neighbors
            total = int((grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, j] + grid[(i-1)%N, (j+1)%N] +
                         grid[i, (j-1)%N] + grid[i, (j+1)%N] +
                         grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, j] + grid[(i+1)%N, (j+1)%N]))

            # Apply rules
            if grid[i, j] == 1:
                if (total < 2) or (total > 3):
                    new_grid[i, j] = 0
            else:
                if total == 3:
                    new_grid[i, j] = 1
    return new_grid

# Initialize grid
N = 100
grid = np.random.choice([0, 1], N*N, p=[0.8, 0.2]).reshape(N, N)

# Set up the plot
fig, ax = plt.subplots()
img = ax.imshow(grid, interpolation='nearest', cmap='gray')
ax.axis('off')

def animate(i):
    global grid
    grid = update_grid(grid)
    img.set_data(grid)
    return [img]

print("Creating Game of Life animation. This may take a moment...")
ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)

try:
    ani.save('game_of_life.gif', writer='pillow', fps=15)
    print("Animation saved to game_of_life.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

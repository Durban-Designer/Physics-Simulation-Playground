"""
Simulation of the Gray-Scott reaction-diffusion model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
N = 200  # Grid size
Du, Dv = 0.16, 0.08  # Diffusion rates
F, k = 0.055, 0.062  # Feed and kill rates
dt = 1.0

# Initialize grids
U = np.ones((N, N))
V = np.zeros((N, N))

# Initial perturbation
r = 10
U[N//2 - r:N//2 + r, N//2 - r:N//2 + r] = 0.5
V[N//2 - r:N//2 + r, N//2 - r:N//2 + r] = 0.25

U += 0.02 * np.random.rand(N, N)
V += 0.02 * np.random.rand(N, N)

def laplacian(grid):
    return (np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
            np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) - 4 * grid)

# Set up the plot
fig, ax = plt.subplots()
im = ax.imshow(V, cmap='viridis', animated=True)
ax.axis('off')

def animate(i):
    global U, V
    for _ in range(10):
        laplace_U = laplacian(U)
        laplace_V = laplacian(V)
        reaction = U * V**2
        U_new = U + (Du * laplace_U - reaction + F * (1 - U)) * dt
        V_new = V + (Dv * laplace_V + reaction - (F + k) * V) * dt
        U, V = U_new, V_new
    im.set_array(V)
    if i % 10 == 0:
        print(f"Frame {i}")
    return [im]

print("Creating Gray-Scott reaction-diffusion animation. This may take a moment...")
ani = animation.FuncAnimation(fig, animate, frames=200, interval=20, blit=True)

try:
    ani.save('gray_scott.gif', writer='pillow', fps=30)
    print("Animation saved to gray_scott.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

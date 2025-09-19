"""
Visualization of gravitational lensing and spacetime curvature.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
GRID_SIZE = 100

# Mass position and properties
mass_x, mass_y = GRID_SIZE // 2, GRID_SIZE // 2
SCHWARZSCHILD_RADIUS = 10

# Create grid
x = np.linspace(0, GRID_SIZE, GRID_SIZE)
y = np.linspace(0, GRID_SIZE, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# Spacetime curvature (simplified 2D potential well)
dist = np.sqrt((X - mass_x)**2 + (Y - mass_y)**2)
Z = -SCHWARZSCHILD_RADIUS / (dist + 1e-6)

# Light rays
N_RAYS = 20
rays_y = np.linspace(0, GRID_SIZE, N_RAYS)
rays_x = np.zeros_like(rays_y)

# Deflect light rays
deflected_rays_x = []
deflected_rays_y = []

for start_y in rays_y:
    ray_x = np.arange(GRID_SIZE)
    ray_y = np.full_like(ray_x, start_y, dtype=float)
    for i in range(1, GRID_SIZE):
        dist_to_mass = np.sqrt((ray_x[i-1] - mass_x)**2 + (ray_y[i-1] - mass_y)**2)
        if dist_to_mass > 0:
            # Simplified deflection angle calculation
            deflection = 4 * SCHWARZSCHILD_RADIUS / dist_to_mass
            # Apply deflection towards the mass
            angle_to_mass = np.arctan2(mass_y - ray_y[i-1], mass_x - ray_x[i-1])
            ray_y[i] = ray_y[i-1] + deflection * np.sin(angle_to_mass)
    deflected_rays_x.append(ray_x)
    deflected_rays_y.append(ray_y)

# Plotting
fig = plt.figure(figsize=(12, 6))

# Spacetime curvature plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('Spacetime Curvature')

# Gravitational lensing plot
ax2 = fig.add_subplot(122)
ax2.set_aspect('equal')
ax2.set_title('Gravitational Lensing')
ax2.add_patch(plt.Circle((mass_x, mass_y), SCHWARZSCHILD_RADIUS, color='k', label='Mass'))
for rx, ry in zip(deflected_rays_x, deflected_rays_y):
    ax2.plot(rx, ry, 'y-')
ax2.legend()

plt.tight_layout()
plt.savefig("gravitational_lensing.png")
print("Gravitational lensing plot saved to gravitational_lensing.png")

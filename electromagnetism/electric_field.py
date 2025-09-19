"""
Visualization of the electric field from a collection of point charges.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
GRID_SIZE = 50

# Charges (charge, x, y)
charges = [(1, 20, 25), (-1, 30, 25)]

# Create grid
x = np.linspace(0, GRID_SIZE, GRID_SIZE)
y = np.linspace(0, GRID_SIZE, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# Calculate electric field
Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)

for q, qx, qy in charges:
    dist_sq = (X - qx)**2 + (Y - qy)**2
    # Avoid division by zero at the charge location
    dist_sq[dist_sq == 0] = 1e-6
    dist = np.sqrt(dist_sq)
    Ex += q * (X - qx) / dist**3
    Ey += q * (Y - qy) / dist**3

# Plot the field
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_title("Electric Field of Point Charges")

# Plot charges
for q, qx, qy in charges:
    color = 'r' if q > 0 else 'b'
    ax.plot(qx, qy, color + 'o', ms=10)

# Plot field lines using streamplot
ax.streamplot(X, Y, Ex, Ey, color='k', linewidth=1, density=2)

plt.savefig("electric_field.png")
print("Electric field plot saved to electric_field.png")

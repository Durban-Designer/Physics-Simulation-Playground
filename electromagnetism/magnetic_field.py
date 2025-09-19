"""
Visualization of the magnetic field from a current-carrying wire using the Biot-Savart Law.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
GRID_SIZE = 50
I = 1.0  # Current

# Wire position (perpendicular to the xy-plane)
wire_x, wire_y = 25, 25

# Create grid
x = np.linspace(0, GRID_SIZE, GRID_SIZE)
y = np.linspace(0, GRID_SIZE, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# Calculate magnetic field (Biot-Savart Law for an infinite wire)
# B = (mu_0 * I) / (2 * pi * r)
dx = X - wire_x
dy = Y - wire_y
r_sq = dx**2 + dy**2
r_sq[r_sq == 0] = 1e-6

# Field components (B is tangential, so Bx = -dy, By = dx)
Bx = -dy / r_sq
By = dx / r_sq

# Plot the field
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_title("Magnetic Field of a Current-Carrying Wire")

# Plot wire
ax.plot(wire_x, wire_y, 'ro', ms=10, label='Wire (current out of page)')

# Plot field lines using streamplot
ax.streamplot(X, Y, Bx, By, color='k', linewidth=1, density=2)
ax.legend()

plt.savefig("magnetic_field.png")
print("Magnetic field plot saved to magnetic_field.png")

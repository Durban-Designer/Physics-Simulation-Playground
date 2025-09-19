"""
Visualization of the 3D probability distributions (orbitals) of the hydrogen atom.
This script is computationally intensive and may take some time to run.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm, genlaguerre, factorial

# Quantum numbers (n, l, m)
ORBITALS_TO_PLOT = [(2, 1, 0), (3, 2, 1)]

# Create a 3D grid
GRID_SIZE = 100
MAX_R = 40
x = np.linspace(-MAX_R, MAX_R, GRID_SIZE)
y = np.linspace(-MAX_R, MAX_R, GRID_SIZE)
z = np.linspace(-MAX_R, MAX_R, GRID_SIZE)
X, Y, Z = np.meshgrid(x, y, z)

# Convert to spherical coordinates
R = np.sqrt(X**2 + Y**2 + Z**2)
THETA = np.arccos(Z / R)
PHI = np.arctan2(Y, X)

fig = plt.figure(figsize=(12, 6))

for i, (n, l, m) in enumerate(ORBITALS_TO_PLOT):
    ax = fig.add_subplot(1, len(ORBITALS_TO_PLOT), i + 1, projection='3d')

    # Radial part of the wavefunction
    a0 = 1  # Bohr radius
    rho = 2 * R / (n * a0)
    L_nl = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    R_nl = np.sqrt((2 / (n * a0))**3 * factorial(n - l - 1) / (2 * n * factorial(n + l))) * np.exp(-rho / 2) * rho**l * L_nl

    # Angular part (spherical harmonics)
    Y_lm = sph_harm(m, l, PHI, THETA)

    # Total wavefunction (probability density)
    psi_sq = np.abs(R_nl * Y_lm)**2

    # Plotting the isosurface
    # This is a simplified way to visualize. A proper 3D plot would use a library like mayavi.
    # We plot points where the probability density is above a certain threshold.
    threshold = np.max(psi_sq) * 0.1
    points = np.where(psi_sq > threshold)
    ax.scatter(X[points], Y[points], Z[points], c=np.abs(Y_lm[points]), cmap='hsv', s=1)

    ax.set_title(f'n={n}, l={l}, m={m}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.savefig("hydrogen_orbitals.png")
print("Hydrogen orbitals plot saved to hydrogen_orbitals.png")



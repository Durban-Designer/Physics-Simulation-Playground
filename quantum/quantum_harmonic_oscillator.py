"""
Visualization of the eigenstates of the quantum harmonic oscillator.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite, factorial

# Parameters
N_LEVELS = 5
x = np.linspace(-5, 5, 1000)

# Potential
potential = 0.5 * x**2

def qho_wavefunction(n, x):
    """Calculates the wavefunction for the nth energy level."""
    prefactor = 1 / np.sqrt(2**n * factorial(n)) * (1/np.pi)**0.25
    hermite_poly = hermite(n)(x)
    return prefactor * np.exp(-x**2 / 2) * hermite_poly

# Plot the results
plt.figure(figsize=(10, 8))
plt.plot(x, potential, 'k--', label='Potential')

for n in range(N_LEVELS):
    energy = n + 0.5
    psi = qho_wavefunction(n, x)
    plt.plot(x, energy + psi**2, label=f'n={n}')

plt.title("Quantum Harmonic Oscillator Eigenstates")
plt.xlabel("Position")
plt.ylabel("Energy / Probability Density")
plt.legend()
plt.grid(True)
plt.savefig("quantum_harmonic_oscillator.png")
print("Quantum harmonic oscillator plot saved to quantum_harmonic_oscillator.png")

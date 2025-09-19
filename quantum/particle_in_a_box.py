"""
Simulation of a quantum particle in a 1D infinite potential well.
"""

import numpy as np
import matplotlib.pyplot as plt

def solve_particle_in_a_box(length=1.0, num_points=1000, n_levels=3):
    """
    Calculates the wavefunctions and energy levels for a particle in a 1D box.

    Args:
        length (float): The length of the box.
        num_points (int): The number of points to use for the x-axis.
        n_levels (int): The number of energy levels to calculate.

    Returns:
        A tuple containing: x-coordinates, wavefunctions, and energy levels.
    """
    x = np.linspace(0, length, num_points)
    wavefunctions = []
    energies = []

    for n in range(1, n_levels + 1):
        psi = np.sqrt(2 / length) * np.sin(n * np.pi * x / length)
        wavefunctions.append(psi)

        # Energy levels (using arbitrary units for hbar and mass)
        energy = (n**2 * np.pi**2) / (2 * length**2)
        energies.append(energy)

    return x, np.array(wavefunctions), np.array(energies)

def plot_particle_in_a_box(x, wavefunctions, energies):
    """
    Plots the wavefunctions and energy levels.

    Args:
        x (np.ndarray): The x-coordinates.
        wavefunctions (np.ndarray): The wavefunctions.
        energies (np.ndarray): The energy levels.
    """
    plt.figure(figsize=(10, 6))
    for i, psi in enumerate(wavefunctions):
        plt.plot(x, psi + energies[i], label=f"n={i+1}, E={energies[i]:.2f}")

    plt.title("Particle in a 1D Box")
    plt.xlabel("Position")
    plt.ylabel("Wavefunction (offset by energy)")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    x, wavefunctions, energies = solve_particle_in_a_box()
    plot_particle_in_a_box(x, wavefunctions, energies)

"""
2D Ising model simulation using the Metropolis-Hastings algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt

def ising_model(N=50, temp=2.269, n_steps=100000):
    """
    Simulates the 2D Ising model.

    Args:
        N (int): The size of the N x N lattice.
        temp (float): The temperature.
        n_steps (int): The number of Monte Carlo steps.

    Returns:
        The final state of the lattice.
    """
    lattice = np.random.choice([-1, 1], size=(N, N))

    for step in range(n_steps):
        # Choose a random spin to flip
        i, j = np.random.randint(0, N, 2)

        # Calculate the change in energy if the spin is flipped
        # Periodic boundary conditions are used
        delta_E = 2 * lattice[i, j] * (lattice[(i + 1) % N, j] + lattice[(i - 1) % N, j] +
                                       lattice[i, (j + 1) % N] + lattice[i, (j - 1) % N])

        # Metropolis-Hastings condition
        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / temp):
            lattice[i, j] *= -1

    return lattice

def plot_lattice(lattice, temp):
    """
    Plots the lattice and saves it to a file.

    Args:
        lattice (np.ndarray): The final lattice state.
        temp (float): The temperature of the simulation.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(lattice, cmap='binary', interpolation='nearest')
    plt.title(f"Ising Model at T={temp:.3f}, Magnetization={np.mean(lattice):.3f}")
    plt.xlabel("Spin")
    plt.ylabel("Spin")
    plt.savefig("ising_model.png")
    print("Ising model plot saved to ising_model.png")

if __name__ == '__main__':
    # Run at the critical temperature
    final_lattice = ising_model(temp=2.269)
    plot_lattice(final_lattice, temp=2.269)

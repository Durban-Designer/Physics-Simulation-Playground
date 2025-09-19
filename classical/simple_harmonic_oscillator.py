"""
A simple simulation of a classical harmonic oscillator.
"""

import numpy as np
import matplotlib.pyplot as plt

def simulate_harmonic_oscillator(mass=1.0, k=1.0, initial_position=1.0, initial_velocity=0.0, total_time=10.0, dt=0.01):
    """
    Simulates the motion of a simple harmonic oscillator.

    Args:
        mass (float): The mass of the object.
        k (float): The spring constant.
        initial_position (float): The initial position of the object.
        initial_velocity (float): The initial velocity of the object.
        total_time (float): The total time to simulate.
        dt (float): The time step for the simulation.

    Returns:
        A tuple containing two numpy arrays: time and position.
    """
    num_steps = int(total_time / dt)
    time = np.linspace(0, total_time, num_steps)
    position = np.zeros(num_steps)
    velocity = np.zeros(num_steps)

    position[0] = initial_position
    velocity[0] = initial_velocity

    for i in range(1, num_steps):
        acceleration = -k / mass * position[i-1]
        velocity[i] = velocity[i-1] + acceleration * dt
        position[i] = position[i-1] + velocity[i] * dt

    return time, position

def plot_oscillator(time, position):
    """
    Plots the position of the harmonic oscillator over time.

    Args:
        time (np.ndarray): The time array.
        position (np.ndarray): The position array.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, position)
    plt.title("Simple Harmonic Oscillator")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    time, position = simulate_harmonic_oscillator()
    plot_oscillator(time, position)

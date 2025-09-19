"""
Simulation of radioactive decay.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N0 = 1000  # Initial number of atoms
TAU = 5.0  # Mean lifetime
TOTAL_TIME = 30.0
DT = 0.1

# Time array
time = np.arange(0, TOTAL_TIME, DT)

# Analytical solution
N_analytical = N0 * np.exp(-time / TAU)

# Monte Carlo simulation
N_mc = []
N_current = N0
for t in time:
    N_mc.append(N_current)
    decayed = np.sum(np.random.rand(N_current) < (1 - np.exp(-DT / TAU)))
    N_current -= decayed

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time, N_analytical, label='Analytical Solution')
plt.step(time, N_mc, where='post', label='Monte Carlo Simulation')
plt.title("Radioactive Decay")
plt.xlabel("Time")
plt.ylabel("Number of Atoms")
plt.legend()
plt.grid(True)
plt.savefig("radioactive_decay.png")
print("Radioactive decay plot saved to radioactive_decay.png")

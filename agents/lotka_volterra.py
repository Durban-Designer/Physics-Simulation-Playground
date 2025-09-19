"""
Simulation of the Lotka-Volterra (predator-prey) equations.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
alpha = 1.1  # Prey growth rate
beta = 0.4   # Predation rate
delta = 0.4  # Predator death rate
gamma = 0.1  # Predator growth rate from predation

# Initial conditions (prey, predators)
xy0 = [10, 5]

# The Lotka-Volterra model differential equations
def lotka_volterra(t, y, alpha, beta, delta, gamma):
    x, y = y
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return dxdt, dydt

# Time grid
t_span = [0, 70]
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Integrate the equations over the time grid
sol = solve_ivp(lotka_volterra, t_span, xy0, args=(alpha, beta, delta, gamma), t_eval=t_eval)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], 'b', label='Prey')
plt.plot(sol.t, sol.y[1], 'r', label='Predators')
plt.title("Lotka-Volterra (Predator-Prey) Model")
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.savefig("lotka_volterra.png")
print("Lotka-Volterra plot saved to lotka_volterra.png")

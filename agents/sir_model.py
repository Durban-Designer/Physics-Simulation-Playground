"""
Simulation of the SIR (Susceptible-Infected-Recovered) epidemic model.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
N = 1000  # Total population
I0, R0 = 1, 0  # Initial number of infected and recovered individuals
S0 = N - I0 - R0  # Initial number of susceptible individuals
beta = 0.2  # Contact rate
gamma = 1/10  # Recovery rate

# The SIR model differential equations
def sir_model(t, y, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0

# Time grid
t_span = [0, 160]
t_eval = np.linspace(t_span[0], t_span[1], 160)

# Integrate the SIR equations over the time grid
sol = solve_ivp(sir_model, t_span, y0, args=(N, beta, gamma), t_eval=t_eval)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], 'b', label='Susceptible')
plt.plot(sol.t, sol.y[1], 'r', label='Infected')
plt.plot(sol.t, sol.y[2], 'g', label='Recovered')
plt.title("SIR Epidemic Model")
plt.xlabel('Time / days')
plt.ylabel('Number of People')
plt.legend()
plt.grid(True)
plt.savefig("sir_model.png")
print("SIR model plot saved to sir_model.png")

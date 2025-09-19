"""
Simulation of the Lorenz attractor, a system of ordinary differential equations
known for its chaotic behavior.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def lorenz(t, xyz, sigma=10, rho=28, beta=8/3):
    """
    Calculates the derivatives of the Lorenz system.

    Args:
        t: Time (not used, but required by solve_ivp).
        xyz (list): A list or array [x, y, z] representing the state.
        sigma (float): The sigma parameter.
        rho (float): The rho parameter.
        beta (float): The beta parameter.

    Returns:
        A list of the derivatives [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Initial condition
xyz0 = [0.1, 0.0, 0.0]

# Time span
t_span = [0, 50]
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Solve the ODEs
sol = solve_ivp(lorenz, t_span, xyz0, t_eval=t_eval)

# Plot the result
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.5)
ax.set_title("Lorenz Attractor")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")

plt.savefig("lorenz_attractor.png")
print("Lorenz attractor plot saved to lorenz_attractor.png")

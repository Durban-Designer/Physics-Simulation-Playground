"""
2D Lattice Boltzmann simulation of fluid flow around a cylinder.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
NX = 400  # x-dimension
NY = 100  # y-dimension
MAX_STEPS = 10000
REYNOLDS_NUMBER = 100

# Lattice parameters
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
WEIGHTS = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Flow parameters
U_IN = 0.04  # Inlet velocity
NU = U_IN * (NY / 4) / REYNOLDS_NUMBER  # Kinematic viscosity
TAU = 3 * NU + 0.5  # Relaxation time

# Initialize distribution functions
F = np.ones((NY, NX, 9)) * (1/9)
F_EQ = np.zeros_like(F)

# Obstacle (cylinder)
obstacle = np.fromfunction(lambda y, x: (x - NX/5)**2 + (y - NY/2)**2 < (NY/4)**2, (NY, NX))

# Main loop
fig, ax = plt.subplots(figsize=(10, 4))

def run_simulation(max_steps):
    u = np.zeros((NY, NX, 2))
    rho = np.ones((NY, NX))
    F = np.ones((NY, NX, 9)) * (1/9)

    for step in range(max_steps):
        # Streaming
        for i in range(9):
            F[:, :, i] = np.roll(np.roll(F[:, :, i], CX[i], axis=1), CY[i], axis=0)

        # Boundary conditions (bounce-back on obstacle)
        bounced = F[obstacle, ::-1]
        F[obstacle] = bounced

        # Macroscopic properties
        rho = np.sum(F, axis=2)
        u = np.dot(F, np.array([CX, CY]).T) / rho[..., np.newaxis]

        # Inlet (Zou-He)
        u[:, 0, :] = [U_IN, 0]
        rho[:, 0] = 1 / (1 - U_IN) * (np.sum(F[:, 0, [0, 2, 4]], axis=1) + 2 * np.sum(F[:, 0, [3, 6, 7]], axis=1))
        F[:, 0, 1] = F[:, 0, 3] + 2/3 * rho[:, 0] * U_IN
        F[:, 0, 5] = F[:, 0, 7] - 0.5 * (F[:, 0, 2] - F[:, 0, 4]) + 1/6 * rho[:, 0] * U_IN
        F[:, 0, 8] = F[:, 0, 6] + 0.5 * (F[:, 0, 2] - F[:, 0, 4]) + 1/6 * rho[:, 0] * U_IN

        # Equilibrium distribution function
        u_sq = u[..., 0]**2 + u[..., 1]**2
        for i in range(9):
            cu = CX[i] * u[..., 0] + CY[i] * u[..., 1]
            F_EQ[:, :, i] = rho * WEIGHTS[i] * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)

        # Collision
        F += -(1/TAU) * (F - F_EQ)

        if step % 100 == 0:
            print(f"Step {step}/{max_steps}")
            yield np.sqrt(u[..., 0]**2 + u[..., 1]**2)


im = ax.imshow(np.zeros((NY, NX)), cmap='viridis', animated=True)
ax.set_title("Lattice Boltzmann Flow")
ax.axis('off')

def animate(fluid_velocity):
    im.set_array(fluid_velocity)
    return im,

print("Starting Lattice Boltzmann simulation. This will take a while...")
ani = animation.FuncAnimation(fig, animate, frames=run_simulation(MAX_STEPS), interval=50, blit=True, repeat=False)

try:
    ani.save('lbm_flow.gif', writer='pillow', fps=15)
    print("Animation saved to lbm_flow.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

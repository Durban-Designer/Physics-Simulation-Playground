"""
Simulation of basic kinematics: constant velocity vs. constant acceleration.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
TOTAL_TIME = 10
DT = 0.1

# Object 1: Constant Velocity
v1 = 10  # m/s

# Object 2: Constant Acceleration
a2 = 2  # m/s^2

# Time array
t = np.arange(0, TOTAL_TIME, DT)

# Calculate positions
x1 = v1 * t
x2 = 0.5 * a2 * t**2

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, max(np.max(x1), np.max(x2)) * 1.1)
ax.set_ylim(0, 3)
ax.set_yticks([])
ax.set_xlabel("Position (m)")
ax.set_title("Kinematics: Constant Velocity vs. Constant Acceleration")

point1, = ax.plot([], [], 'bo', ms=10, label='Constant Velocity')
point2, = ax.plot([], [], 'ro', ms=10, label='Constant Acceleration')
ax.legend()

def animate(i):
    point1.set_data([x1[i]], [1])
    point2.set_data([x2[i]], [2])
    return point1, point2

print("Creating kinematics animation...")
ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=100, blit=True)

try:
    ani.save('kinematics.gif', writer='pillow', fps=10)
    print("Animation saved to kinematics.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

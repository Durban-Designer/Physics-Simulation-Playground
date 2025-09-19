"""
Visualization of Lorentz contraction and time dilation in special relativity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
C = 1.0  # Speed of light

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Ax 1: Lorentz Contraction
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_aspect('equal')
ax1.set_title('Lorentz Contraction')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')

contracting_square = plt.Rectangle((-0.5, -0.5), 1, 1, fc='b')
ax1.add_patch(contracting_square)

# Ax 2: Time Dilation
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 10)
ax2.set_title('Time Dilation')
ax2.set_xlabel('Velocity (v/c)')
ax2.set_ylabel('Time Dilation Factor (gamma)')

v_range = np.linspace(0, 0.999, 100)
gamma_values = 1 / np.sqrt(1 - v_range**2)
ax2.plot(v_range, gamma_values, 'r-')
dilation_point, = ax2.plot([], [], 'ro', ms=10)

def animate(v):
    # Lorentz Contraction
    gamma = 1 / np.sqrt(1 - v**2)
    contracted_width = 1 / gamma
    contracting_square.set_width(contracted_width)
    contracting_square.set_x(-contracted_width / 2)
    
    # Time Dilation
    dilation_point.set_data([v], [gamma])
    
    fig.suptitle(f'Velocity = {v:.3f}c')
    
    return contracting_square, dilation_point

print("Creating special relativity animation...")
velocities = np.concatenate([np.linspace(0, 0.99, 100), np.linspace(0.99, 0, 100)])
ani = animation.FuncAnimation(fig, animate, frames=velocities, interval=50, blit=True, repeat=False)

try:
    ani.save('special_relativity.gif', writer='pillow', fps=15)
    print("Animation saved to special_relativity.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

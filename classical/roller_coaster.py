"""
Simulation of a roller coaster, demonstrating the conservation of energy.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the track
x = np.linspace(0, 100, 500)
y = 10 * np.sin(x / 10) + 5 * np.sin(x / 5) + 15

# Parameters
m = 1.0  # mass
g = 9.81 # gravity

# Initial conditions
E_total = m * g * y[0]
v = 0

# Simulation
positions = []
velocities = []
KE = []
PE = []

current_pos_index = 0

for _ in range(1000):
    positions.append(current_pos_index)
    velocities.append(v)
    
    pe = m * g * y[current_pos_index]
    ke = E_total - pe
    if ke < 0:
        ke = 0 # Can't have negative kinetic energy
    
    KE.append(ke)
    PE.append(pe)
    
    v = np.sqrt(2 * ke / m)
    
    # Find next position based on velocity
    if v > 0:
        distance_to_move = v * 0.1 # dt = 0.1
        path_lengths = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumulative_path = np.cumsum(path_lengths)
        
        current_path_pos = cumulative_path[current_pos_index-1] if current_pos_index > 0 else 0
        target_path_pos = current_path_pos + distance_to_move
        
        # Find the index corresponding to the new position
        new_pos_index = np.searchsorted(cumulative_path, target_path_pos)
        if new_pos_index >= len(x) -1:
            current_pos_index = 0 # restart
        else:
            current_pos_index = new_pos_index
    

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

ax1.plot(x, y, 'k-')
ax1.set_title("Roller Coaster")
ax1.set_xlabel("Position")
ax1.set_ylabel("Height")

cart, = ax1.plot([], [], 'ro', ms=10)

ax2.set_xlim(0, len(positions))
ax2.set_ylim(0, E_total * 1.1)
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Energy")
ke_line, = ax2.plot([], [], 'r-', label='Kinetic Energy')
pe_line, = ax2.plot([], [], 'b-', label='Potential Energy')
ax2.axhline(E_total, color='k', ls='--', label='Total Energy')
ax2.legend()

def animate(i):
    cart.set_data([x[positions[i]]], [y[positions[i]]])
    ke_line.set_data(range(i), KE[:i])
    pe_line.set_data(range(i), PE[:i])
    return cart, ke_line, pe_line

print("Creating roller coaster animation...")
ani = animation.FuncAnimation(fig, animate, frames=len(positions), interval=30, blit=True)

try:
    ani.save('roller_coaster.gif', writer='pillow', fps=30)
    print("Animation saved to roller_coaster.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

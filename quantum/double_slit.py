"""
Simulation of the double-slit experiment, showing the build-up of an interference pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
N_PARTICLES = 2000
SCREEN_WIDTH = 200
SLIT_SEPARATION = 40
SLIT_WIDTH = 5
WAVELENGTH = 10

# Create screen
screen = np.zeros(SCREEN_WIDTH)

# Probability distribution on the screen
x = np.arange(SCREEN_WIDTH)
# Path length difference from each slit
d1 = np.abs(x - (SCREEN_WIDTH/2 - SLIT_SEPARATION/2))
d2 = np.abs(x - (SCREEN_WIDTH/2 + SLIT_SEPARATION/2))
# This is a simplified model for the intensity
intensity = (np.cos(np.pi * (d1 - d2) / WAVELENGTH))**2

# Single-slit diffraction envelope
envelope = (np.sinc((x - SCREEN_WIDTH/2) / (WAVELENGTH * 10)))**2

probability = intensity * envelope
probability /= np.sum(probability)

# Simulate particle hits
particle_hits = np.random.choice(x, size=N_PARTICLES, p=probability)

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 3]})

ax1.plot(x, probability, 'r-')
ax1.set_title("Probability Distribution")
ax1.set_yticks([])

ax2.set_xlim(0, SCREEN_WIDTH)
ax2.set_ylim(0, N_PARTICLES / 10)
ax2.set_title("Interference Pattern")
ax2.set_xlabel("Screen Position")
ax2.set_ylabel("Number of Hits")

hist, bins = np.histogram(particle_hits, bins=SCREEN_WIDTH, range=(0, SCREEN_WIDTH))

bar_container = ax2.bar(bins[:-1], np.zeros_like(bins[:-1]), width=1)

def animate(i):
    hist, _ = np.histogram(particle_hits[:i+1], bins=SCREEN_WIDTH, range=(0, SCREEN_WIDTH))
    for count, rect in zip(hist, bar_container.patches):
        rect.set_height(count)
    return bar_container.patches

print("Creating double-slit experiment animation...")
ani = animation.FuncAnimation(fig, animate, frames=N_PARTICLES, interval=10, blit=True, repeat=False)

try:
    ani.save('double_slit.gif', writer='pillow', fps=30)
    print("Animation saved to double_slit.gif")
except Exception as e:
    print(f"Could not save animation due to error: {e}")
    print("Aborting.")

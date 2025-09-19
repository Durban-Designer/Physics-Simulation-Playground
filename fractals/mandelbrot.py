"""
Generation of the Mandelbrot set, a classic fractal.
"""

import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(width=800, height=800, x_center=-0.75, y_center=0, zoom=1, max_iter=256):
    """
    Generates the Mandelbrot set.

    Args:
        width (int): The width of the image in pixels.
        height (int): The height of the image in pixels.
        x_center (float): The x-coordinate of the center of the view.
        y_center (float): The y-coordinate of the center of the view.
        zoom (float): The zoom level.
        max_iter (int): The maximum number of iterations.

    Returns:
        A 2D numpy array representing the Mandelbrot set image.
    """
    x_min = x_center - 1.5 / zoom
    x_max = x_center + 1.5 / zoom
    y_min = y_center - 1.5 / zoom
    y_max = y_center + 1.5 / zoom

    x, y = np.meshgrid(np.linspace(x_min, x_max, width), np.linspace(y_min, y_max, height))
    c = x + 1j * y
    z = np.zeros_like(c)
    divergence_time = np.full(c.shape, max_iter, dtype=int)

    for i in range(max_iter):
        z = z*z + c
        diverged = np.abs(z) > 2
        diverged_now = diverged & (divergence_time == max_iter)
        divergence_time[diverged_now] = i
        z[diverged] = 2  # Avoid overflow

    return divergence_time

print("Generating Mandelbrot set. This may take a moment...")
mandelbrot_image = mandelbrot()

plt.figure(figsize=(10, 10))
plt.imshow(mandelbrot_image, cmap='twilight_shifted', interpolation='nearest')
plt.title("Mandelbrot Set")
plt.axis('off')
plt.savefig("mandelbrot_set.png")
print("Mandelbrot set image saved to mandelbrot_set.png")

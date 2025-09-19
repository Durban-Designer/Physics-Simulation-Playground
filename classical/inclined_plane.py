"""
Visualization of the forces acting on a block on an inclined plane.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

# Parameters
mass = 10  # kg
g = 9.81   # m/s^2
angle_deg = 30
mu_s = 0.5  # Coefficient of static friction
mu_k = 0.3  # Coefficient of kinetic friction

angle_rad = np.deg2rad(angle_deg)

# Forces
F_gravity = mass * g
F_normal = F_gravity * np.cos(angle_rad)
F_parallel = F_gravity * np.sin(angle_rad)
F_friction_static_max = mu_s * F_normal

if F_parallel > F_friction_static_max:
    F_friction = mu_k * F_normal
    acceleration = (F_parallel - F_friction) / mass
    status = f"Block is sliding with a = {acceleration:.2f} m/s^2"
else:
    F_friction = -F_parallel
    acceleration = 0
    status = "Block is held by static friction"

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
ax.set_xlim(-5, 15)
ax.set_ylim(-2, 12)
ax.set_title("Forces on a Block on an Inclined Plane")
ax.grid(True)

# Draw the plane
plane_x = np.array([-5, 15])
plane_y = plane_x * np.tan(angle_rad)
ax.plot(plane_x, plane_y, 'k', lw=2)

# Draw the block
block_center_x = 5
block_center_y = block_center_x * np.tan(angle_rad)
rect = plt.Rectangle((block_center_x - 1, block_center_y - 1), 2, 2, color='blue')
rot = transforms.Affine2D().rotate_deg(angle_deg)
t = transforms.Affine2D().translate(block_center_x, block_center_y)
rect.set_transform(rot + t + ax.transData)
ax.add_patch(rect)

# Draw force vectors
# Gravity
ax.arrow(block_center_x, block_center_y, 0, -F_gravity/10, head_width=0.5, head_length=0.5, fc='r', ec='r', label=f'Gravity ({F_gravity:.1f} N)')
# Normal
ax.arrow(block_center_x, block_center_y, -F_normal/10 * np.sin(angle_rad), F_normal/10 * np.cos(angle_rad), head_width=0.5, head_length=0.5, fc='g', ec='g', label=f'Normal ({F_normal:.1f} N)')
# Friction
ax.arrow(block_center_x, block_center_y, F_friction/10 * np.cos(angle_rad), F_friction/10 * np.sin(angle_rad), head_width=0.5, head_length=0.5, fc='orange', ec='orange', label=f'Friction ({np.abs(F_friction):.1f} N)')

ax.text(0, 10, status, fontsize=12)
ax.legend()

plt.savefig("inclined_plane.png")
print("Inclined plane plot saved to inclined_plane.png")
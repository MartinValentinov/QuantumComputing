import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_box_aspect([1,1,1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='c', alpha=0.1)

vec = [0, 0, 1]
arrow = ax.quiver(0,0,0, vec[0], vec[1], vec[2], color='r', linewidth=2, arrow_length_ratio=0.1)

def update(frame):
    global arrow
    arrow.remove()
    theta = np.radians(frame)
    phi = np.radians(frame*0.5)
    x_vec = np.sin(phi)*np.cos(theta)
    y_vec = np.sin(phi)*np.sin(theta)
    z_vec = np.cos(phi)
    arrow = ax.quiver(0,0,0, x_vec, y_vec, z_vec, color='r', linewidth=2, arrow_length_ratio=0.1)
    return arrow,

ani = FuncAnimation(fig, update, frames=np.arange(0,360,2), interval=50, blit=False)

plt.show()
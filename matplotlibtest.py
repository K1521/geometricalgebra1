import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

# Define the implicit surface equation
def f(x, y, z):
    return ((x**2 + y**2 + z**2)**2 - 2.5*(x**2 + y**2) + 1.5*z**2 + 0.5625)

# Generate the mesh using the marching cubes algorithm
x, y, z = np.mgrid[-2:2:100j, -2:2:100j, -2:2:100j]
vol = f(x, y, z)
verts, faces, _, _ = measure.marching_cubes(vol, 0)

# Plot the mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='jet', lw=1)
plt.show()
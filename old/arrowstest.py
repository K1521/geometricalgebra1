import numpy as np

import pyvista as pv

sphere = pv.Sphere(radius=3.14)

# make cool swirly pattern
vectors = np.vstack(
    (
        np.sin(sphere.points[:, 0]),
        np.cos(sphere.points[:, 1]),
        np.cos(sphere.points[:, 2]),
    )
).T
print(sphere.points)
print(vectors)
# add and scale
sphere["vectors"] = vectors * 0.3
sphere.set_active_vectors("vectors")

# plot just the arrows
sphere.arrows.plot()
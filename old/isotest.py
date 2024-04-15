from numpy import cos, pi, mgrid
import pyvista as pv
pv.set_plot_theme('dark')

#%% Data
x, y, z = pi*mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
vol = cos(x) + cos(y) + cos(z)
grid = pv.StructuredGrid(x, y, z)
grid["vol"] = vol.flatten()
contours = grid.contour([0])
#grid.plot()
#%% Visualization


p = pv.Plotter()
p.add_axes()
p.add_mesh(contours, scalars=contours.points[:, 2], show_scalar_bar=False)
p.show_grid()
p.show()
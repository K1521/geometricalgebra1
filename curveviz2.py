from intervallarethmetic.derivativexyz import xyzderiv


import numpy as np
import pyvista as pv

# Define the functions
def f1(x, y, z):
    return x**2 + y**2 + z**2 - 1

def f2(x, y, z):
    return 0.3*x + 1.5*y + 0.7*z - 0.8

# Create a grid of points in the domain
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
z = np.linspace(-2, 2, 100)
x, y, z = np.meshgrid(x, y, z)

# Evaluate the functions on the grid
values1 = f1(x, y, z)
values2 = f2(x, y, z)

# Create a PyVista structured grid
grid = pv.StructuredGrid(x, y, z)

# Add the function values as point data
grid['f1'] = values1.ravel(order='F')
grid['f2'] = values2.ravel(order='F')

# Contour the functions at level 0
contours1 = grid.contour([0], scalars='f1')
contours2 = grid.contour([0], scalars='f2')

# Create a PyVista plotter object with a dark background
plotter = pv.Plotter(theme=pv.themes.DarkTheme())

# Add the contours to the plotter
plotter.add_mesh(contours1, color='blue', label='f1(x, y, z) = 0')
plotter.add_mesh(contours2, color='red', label='f2(x, y, z) = 0')

# Add grid and axes
plotter.show_grid(color='white')
plotter.add_axes()

# Add legend and display the plot
plotter.add_legend()
plotter.show()

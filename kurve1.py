import numpy as np
import pyvista as pv

# Definieren Sie Ihre Funktion f(x, y, z)
def f(x, y, z):
    return np.sin(x) + np.cos(y) - z

# Erstellen Sie ein Gitter von Punkten
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
z = np.linspace(-5, 5, 100)
x, y, z = np.meshgrid(x, y, z)

# Berechnen Sie den Wert der Funktion an jedem Punkt
values = f(x, y, z)

# Erstellen Sie ein PyVista-Gitter
grid = pv.UniformGrid(x.shape)
grid["values"] = values.flatten(order="F")

# Erzeugen Sie die Kontur bei f(x, y, z) = 0
contours = grid.contour([0])

# Visualisieren Sie die Kontur
plotter = pv.Plotter()
plotter.add_mesh(contours, color="red", line_width=5)
plotter.show()
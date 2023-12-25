import numpy as np
import pyvista as pv


pv.set_plot_theme('dark')
plotter = pv.Plotter()
plotter.add_axes()





cubemesh=pv.PolyData()

cubemesh.shallow_copy(pv.Arrow())
plotter.add_mesh(cubemesh)
plotter.show(interactive_update=True)


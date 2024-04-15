import numpy as np
import pyvista as pv


pv.set_plot_theme('dark')
plotter = pv.Plotter()
plotter.add_axes()





cubemesh=pv.Cube()





while True:
    for i in range(1700):
        cubemesh.rotate_vector((1, 0, 0),360/17, inplace=True)
    #cubemeshc.translate((0.001, 0, 0), inplace=True)
    print()
from blademul5 import *
import numpy as np
import pyvista as pv
from time import time


from random import random

pv.set_plot_theme('dark')
p = pv.Plotter()
p.add_axes()
#p.show_grid()




lines = np.hstack([[2, 0, 1], [2, 1, 2]])


strips = np.hstack([[4, 0, 1, 3, 2]])


#vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [0, 0.5, 0]])

vertices = np.array([[np.sin(i), np.cos(i), random()] for i in np.mgrid[0:2*np.pi:100*1J]] )


#mesh = pv.PolyData(vertices, strips=strips)
#faces = np.hstack([[3, 0, 1, 2], [3, 0, 3, 2]])
faces = np.array([100]+list(range(100)))

#mesh = pv.PolyData(vertices, lines=lines)
#mesh = pv.PolyData(vertices,faces)
#mesh = pv.PolyData(vertices, strips=faces)


vertices=[]
faces=[]

faces.extend([3,len(vertices),len(vertices)+1,len(vertices)+2])
vertices.extend([[1,1,1],[1,1,0],[1,0,0]])
faces.extend([3,len(vertices),len(vertices)+1,len(vertices)+2])
vertices.extend([[2,2,2],[2,2,1],[2,1,1]])
faces.extend([3,len(vertices),len(vertices)+1,len(vertices)+2])
vertices.extend([[3,3,3],[3,3,2],[3,2,2]])


mesh=pv.PolyData(np.array(vertices).ravel(), strips=np.array(faces).ravel())
p.add_mesh(mesh,show_edges=1)
p.show()
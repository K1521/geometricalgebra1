import pyvista as pv
import numpy as np
from voxel2 import Voxels

print("hi")

pv.set_plot_theme('dark')
plt = pv.Plotter()
plt.add_axes()
plt.show_grid()


for order in (1,2,4):
    print([i&order==0 for i in range(8)])


voxels=Voxels(1)
voxels.voxels=np.array([
    [0,0,0],
    [2,2,2]
])
#voxels.subdivide()
#vdel=[ True , True , True , True , True , True , True , True]
#vdel[0]=False
#vdel=[False]*8
#vdel[0]=True
#vdel[1]=True
#vdel[5]=True

"""c_n0 = np.stack((intervallx.min, intervally.min, intervallz.min), axis=1)
c_n1 = np.stack((intervallx.max, intervally.min, intervallz.min), axis=1)
c_n2 = np.stack((intervallx.min, intervally.max, intervallz.min), axis=1)
c_n3 = np.stack((intervallx.max, intervally.max, intervallz.min), axis=1)
# - Top
c_n4 = np.stack((intervallx.min, intervally.min, intervallz.max), axis=1)
c_n5 = np.stack((intervallx.max, intervally.min, intervallz.max), axis=1)
c_n6 = np.stack((intervallx.min, intervally.max, intervallz.max), axis=1)
c_n7 = np.stack((intervallx.max, intervally.max, intervallz.max), axis=1)"""
print(Voxels.subvox)
#voxels.removecells(np.array(vdel))

plt.add_mesh(voxels.gridify(),opacity=0.5)
plt.show()
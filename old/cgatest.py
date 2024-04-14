from blademul5 import *

cga=algebra(4,1)
cga.bladenames="123pm"
scalar,e1,e2,e3,ep,em,*_=cga.allblades()
e0=(em-ep)*.5
einf=(em+ep)

def point(x,y,z):
    return e1*x+e2*y+e3*z+einf*(.5*(x*x+y*y+z*z))+e0

def sphere(p,r):
    return p-einf*(0.5*r*r)

s=sphere(point(0.5,0,0),1)


print(s.inner(point(1,1,1)))
#ep=0.5*einf-e0,em=0.5*einf+e0





#from numpy import mgrid
import numpy as np
import pyvista as pv
pv.set_plot_theme('dark')

#%% Data
x, y, z = 2*np.mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
print(x.shape)
vol = np.zeros(x.shape)
for i in range(31):
    for j in range(31):
        for k in range(31):
            vol[i,j,k]=s.inner(point(x[i,j,k],y[i,j,k],z[i,j,k])).lst[0].magnitude
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



row=cga.allblades()
import pandas as pd
print("imported")
table = [[str(x.inner(y))for y in row]for x in row]
df = pd.DataFrame(table, columns = row, index=row)
print(df)

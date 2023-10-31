

from blademul5 import *
dcga=algebra(8,2)

#e1=1,e2=1,e3=1,e4=1,e5=-1,e6=1,e7=1,e8=1,e9=1,e10=-1
#skalar,e1,e2,e3,e4,e6,e7,e8,e9,e5,e10,*_=dcga.allblades()
#multivec=sortgeo(dcga)
multivec=dictgeo(dcga)
e1,e2,e3,e4,e6,e7,e8,e9,e5,e10=multivec.monoblades()

dcga.bladenames="1,2,3,4,6,7,8,9,5,10".split(",")
print(e1,e2,e3,e4,e6,e7,e8,e9,e5,e10)
eo1=0.5*e5-0.5*e4
eo2=0.5*e10-0.5*e9
ei1=e4+e5
ei2=e9+e10


def point1(x,y,z):
    return e1*x+e2*y+e3*z+ei1*(.5*(x*x+y*y+z*z))+eo1
def point2(x,y,z):
    return e6*x+e7*y+e8*z+ei2*(.5*(x*x+y*y+z*z))+eo2

def point(x,y,z):
    return point1(x,y,z).outer(point2(x,y,z))



eo=eo1.outer(eo2)
ei=ei1.outer(ei2)
Txx=e6.outer(e1)
Tyy=e7.outer(e2)

T1=-ei
Tt2=eo2.outer(ei1)+ei2.outer(eo1)
Tt4=-4*eo

def toroid(R,r):
    dSq=R*R-r*r
    return Tt4+2*Tt2*dSq+T1*dSq*dSq-4*R*R*(Txx+Tyy)


t=toroid(1,0.5)

print(t.inner(point(1,1,1)))
print(point(7,8,9))
print(t)

#from numpy import mgrid
import numpy as np
import pyvista as pv
from time import time

pv.set_plot_theme('dark')

#%% Data
x, y, z = 2*np.mgrid[-1:1:31j, -1:1:31j, -1:1:31j]
print(x.shape)
vol = np.zeros(x.shape)
t0=time()
for i in range(31):
    for j in range(31):
        for k in range(31):
            vol[i,j,k]=t.inner(point(x[i,j,k],y[i,j,k],z[i,j,k])).toscalar()
print(time()-t0)
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

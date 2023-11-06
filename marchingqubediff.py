

from blademul5 import *
import numpy as np
import pyvista as pv
from time import time
dcga=algebra(8,2)

#e1=1,e2=1,e3=1,e4=1,e5=-1,e6=1,e7=1,e8=1,e9=1,e10=-1
#skalar,e1,e2,e3,e4,e6,e7,e8,e9,e5,e10,*_=dcga.allblades()
#multivec=sortgeo(dcga)
multivec=sortgeotf(dcga,[])
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

def CGA1_Plane(x,y,z,h):
    vec=(x*e1+y*e2+z*e3)
    return vec*(1/np.sqrt(vec.inner(vec).toscalar()))+h*ei1
def CGA2_Plane(x,y,z,h):
    vec=(x*e6+y*e7+z*e8)
    return vec*(1/np.sqrt(vec.inner(vec).toscalar()))+h*ei2
#CGA2_Plane = { Normalize(_P(1)*e6 + _P(2)*e7 + _P(3)*e8) + _P(4)*ei2 }
def Plane(x,y,z,h):
    return CGA1_Plane(x,y,z,h)^CGA2_Plane(x,y,z,h)
#Plane = {
# CGA1_Plane(_P(1),_P(2),_P(3),_P(4))^CGA2_Plane(_P(1),_P(2),_P(3),_P(4))
#}

T1=-ei
Tt2=eo2.outer(ei1)+ei2.outer(eo1)
Tt4=-4*eo

def toroid(R,r):
    dSq=R*R-r*r
    return Tt4+2*Tt2*dSq+T1*dSq*dSq-4*R*R*(Txx+Tyy)




t=toroid(1,0.5)
p=Plane(2,0,0,0)

for x in np.mgrid[-1:1:5j]:
    print(p.inner(point(x,0,0)))






tovisualise=[t,p]
usecuda=False

#def marching

pv.set_plot_theme('dark')
p = pv.Plotter()
p.add_axes()
p.show_grid()

t0=time()
step=111
#x, y, z = 2*np.mgrid[-1:1:100j, -1:1:100j, -1:1:100j]
x, y, z = 2*np.mgrid[-1:1:step*1J, -1:1:step*1J, -1:1:step*1J]
grid = pv.StructuredGrid(x, y, z)
contours=[]
if usecuda:
    import cupy as cp
    points=point(cp.array(x.flatten()),cp.array(y.flatten()),cp.array(z.flatten()))
    for shape in tovisualise:
        grid["vol"]=cp.asnumpy(points.inner(shape).toscalar())
        contours.append(grid.contour([0]))
else:
    points=point(x.flatten(),y.flatten(),z.flatten())
    for shape in tovisualise:
        grid["vol"]=points.inner(shape).toscalar()
        #print(grid["vol"])
        contours.append(grid.contour([0]))
print(time()-t0)
for c in contours:
    #print(c.points[:2])
    #point(c.points[:0],c.points[:1],c.points[:2]).inner(shape).toscalar()
    p.add_mesh(c, scalars=np.sqrt(1+point(c.points[:,2],c.points[:,1],c.points[:,0]).inner(t).toscalar()))
p.show()
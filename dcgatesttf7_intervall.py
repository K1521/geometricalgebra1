
import tensorflow as tf
from blademul6 import *
import numpy as np
import pyvista as pv
import time
import itertools
dcga=algebra(8,2)

#e1=1,e2=1,e3=1,e4=1,e5=-1,e6=1,e7=1,e8=1,e9=1,e10=-1
#skalar,e1,e2,e3,e4,e6,e7,e8,e9,e5,e10,*_=dcga.allblades()
#multivec=sortgeo(dcga)

#algebra declaration
multivec=sortgeo(dcga,[])
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



#constructing objects to plot
t=toroid(1,0.5)
p=Plane(.1,0,0,.1)



#for x in np.mgrid[-1:1:5j]:
#    print(p.inner(point(x,0,0)))



vis=t^p
#pyvista setup
pv.set_plot_theme('dark')
p = pv.Plotter()
p.add_axes()
p.show_grid()

t0=time.time() 
step=101
#x, y, z = 2*np.mgrid[-1:1:100j, -1:1:100j, -1:1:100j]
x, y, z = 2*np.mgrid[-1:1:step*1J, -1:1:step*1J, -1:1:step*1J]
x, y, z = 2*np.mgrid[-1:1:100J, -1:1:100J, -1:1:100J]
#x, y, z = 2*np.mgrid[0:1:step*1J, 0:1:step*1J, 0:1:step*1J]
grid = pv.StructuredGrid(x, y, z)

##simpleviz
#points=point(x,y,z)
#iprod=points.inner(vis)
#subprod=sum(np.abs(blade.magnitude) for blade in iprod.lst)
#pointsgrid=np.stack([x,y,z], axis=-1)
#dotgrid=pv.StructuredGrid(x,y,z)
#dotgrid["vol"]=subprod.ravel()
#p.add_mesh(dotgrid.threshold((-0.2,0.2)),opacity=0.5)


class intervallareth:
    def __init__(self,min,max) -> None:
        self.min=min
        self.max=max

    def __mul__(self,other):
        #todo optimise case if self==other
        if other==self:
            return self**2
        if isinstance(other,intervallareth):
            combis=[self.min*other.min,self.min*other.max,self.max*other.min,self.max*other.max]
            return intervallareth(np.min(combis, axis=0),np.max(combis, axis=0))
        if other>0:
            return intervallareth(self.min*other,self.max*other)
        else:
            return intervallareth(self.max*other,self.min*other)
    def __rmul__(self,other):
        return self*other
    
    def __add__(self,other):
        if isinstance(other,intervallareth):
            return intervallareth(self.min+other.min,self.max+other.max)
        return intervallareth(self.min+other,self.max+other)
    def __radd__(self,other):
        return self+other
    
    def __sub__(self,other):
        if isinstance(other,intervallareth):
            return intervallareth(self.min-other.max,self.max-other.min)
        return intervallareth(self.min-other,self.max-other)
    def __rsub__(self,other):
        return intervallareth(other-self.max,other-self.min)
    def __neg__(self):
        return intervallareth(-self.max,-self.min)

    def mid(self):
        return (self.min+self.max)/2
    def haszerro(self,maxdelta=0):
        return (self.min<=maxdelta)&(self.max>=-maxdelta)
    def __abs__(self):
        combis=[abs(self.min),abs(self.max)]
        return intervallareth(np.where((self.min<=0)&(0<=self.max), 0, np.min(combis, axis=0)),np.max(combis, axis=0))

    def __pow__(self,exponent):
        if exponent < 0:
            raise ValueError("Power must be non-negative")
        elif exponent == 0:
            #return 1
            return intervallareth(1)
        
        
        if exponent % 2 == 0:
            combis=[self.min**exponent,self.max**exponent]
            return intervallareth(np.where((self.min<=0)&(0<=self.max), 0, np.min(combis, axis=0)),np.max(combis, axis=0))
        else:
            return intervallareth(self.min**exponent,self.max**exponent)
        


t0=time.time()

import sympy
symbols=sympy.symbols('x y z')
#print(point(*symbols).inner(vis).toscalar())
print(sum(abs(blade.magnitude) for blade in (point(*symbols).inner(vis).lst)))
print()
print((sum(abs(blade.magnitude.simplify()) for blade in (point(*symbols).inner(vis).lst))))
print()
print((sum((blade.magnitude)**2 for blade in (point(*symbols).inner(vis).lst))).simplify())
def simpleeval(f):
    s=str(f).replace("Abs","abs")
    return lambda x,y,z:eval(s,locals())


intervallx,intervally,intervallz=[intervallareth(arr[:-1,:-1,:-1],arr[1:,1:,1:]) for arr in (x,y,z)]
points=point(intervallx,intervally,intervallz)
iprod=points.inner(vis)
#subprod=iprod.toscalar()#sum(np.abs(blade.magnitude) for blade in iprod.lst)
#subprod=sum(abs(blade.magnitude) for blade in iprod.lst)
#subprod=sum(abs(blade.magnitude) for blade in iprod.lst)
print(time.time()-t0)
#subprod=simpleeval((sum((blade.magnitude)**2 for blade in (point(*symbols).inner(vis).lst))).simplify())(intervallx,intervally,intervallz)
#subprod=simpleeval((sum(abs(blade.magnitude) for blade in (point(*symbols).inner(vis).lst))).simplify())(intervallx,intervally,intervallz)
#visiblecells=subprod.haszerro().ravel(order='F')
visiblecells=np.all([(blade.magnitude.haszerro()) for blade in iprod.lst],axis=0)
p.add_mesh(grid.hide_cells(np.logical_not(visiblecells.ravel(order='F'))),opacity=0.5)
#p.add_mesh(grid.hide_cells(np.logical_not(subprod.haszerro())),opacity=0.5)
print(time.time()-t0)

p.show()
            
#print(grid.contour([0]))
#print(mesh)








        
#def abs(mima):
#    combis=[abs(mima.min),abs(mima.max)]
#    if mima.min<=0<=mima.max:
#        return intervallareth(0,max(combis))
#    else:
#        return intervallareth(min(combis),max(combis))
    


#todo trinary logic
# TODO  
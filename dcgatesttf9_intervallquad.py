
import tensorflow as tf
from blademul5 import *
import numpy as np
import pyvista as pv
import time
import itertools
import PVGeo
dcga=algebra(8,2)

#e1=1,e2=1,e3=1,e4=1,e5=-1,e6=1,e7=1,e8=1,e9=1,e10=-1
#skalar,e1,e2,e3,e4,e6,e7,e8,e9,e5,e10,*_=dcga.allblades()
#multivec=sortgeo(dcga)

#algebra declaration
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



#constructing objects to plot
t=toroid(1,0.5)
p=Plane(.1,0,0,.1)



#for x in np.mgrid[-1:1:5j]:
#    print(p.inner(point(x,0,0)))



vis=p
#pyvista setup
pv.set_plot_theme('dark')
p = pv.Plotter()
p.add_axes()
p.show_grid()

t0=time.time() 
step=101





class intervallareth:
    def __init__(self,min,max) -> None:
        self.min=min
        self.max=max

    def __mul__(self,other):
        #todo optimise case if self==other
        if other==self:return self**2
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
        #return (self.min+self.max)/2
        return np.average([self.min,self.max], axis=0)
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



lowerbound=-1
upperbound=1
depth=16
maxvoxelnum=100000

intervallx,intervally,intervallz=[intervallareth(np.array([lowerbound]),np.array([upperbound])) for _ in range(3)]



lastvoxnum=1
for j in range(1,depth+1):
    #subdivide cells
    #midx=intervallx.mid()
    #midy=intervally.mid()
    #midz=intervallz.mid()
    #intervallxdown=intervallareth(intervallx.min,midx)#lower half
    #intervallxup  =intervallareth(midx,intervallx.max)#upper half
    #intervallydown=intervallareth(intervally.min,midy)
    #intervallyup  =intervallareth(midy,intervally.max)
    #intervallzdown=intervallareth(intervallz.min,midz)
    #intervallzup  =intervallareth(midz,intervallz.max)
    #intervallx,intervally,intervallz
    #intervallx=intervallareth(np.concatenate([[intervallxdown.min,intervallxup.min][i&1==0] for i in range(8)]),np.concatenate([[intervallxdown.max,intervallxup.max][i&1==0] for i in range(8)]))
    #intervally=intervallareth(np.concatenate([[intervallydown.min,intervallyup.min][i&2==0] for i in range(8)]),np.concatenate([[intervallydown.max,intervallyup.max][i&2==0] for i in range(8)]))
    #intervallz=intervallareth(np.concatenate([[intervallzdown.min,intervallzup.min][i&4==0] for i in range(8)]),np.concatenate([[intervallzdown.max,intervallzup.max][i&4==0] for i in range(8)]))
    #intervallx,intervally,intervallz=[intervallareth(np.concatenate([[min,mid][i&1==0] for i in range(8)]),np.concatenate([[mid,intervall.max][i&1==0] for i in range(8)])) for j,min,mid,max in zip((1,2,4),)]
    intervalls=[]
    for intervall,order in zip((intervallx,intervally,intervallz),(1,2,4)):
        mid=intervall.mid()
        intervalls.append(
            intervallareth(
                np.concatenate([[intervall.min,mid][i&order==0] for i in range(8)]),
                np.concatenate([[mid,intervall.max][i&order==0] for i in range(8)])
                #TODO intervall concatenate method
            )
        )
    intervallx,intervally,intervallz=intervalls

    #print(np.stack([intervallx.min,intervallx.max,intervally.min,intervally.max,intervallz.min,intervallz.max]).T)
    #delete cells without 0
    voxelsintervall=point(intervallx,intervally,intervallz).inner(vis)#calculate 
    voxelswithzerro=np.all([blade.magnitude.haszerro() for blade in voxelsintervall.lst],axis=0)
    intervallx,intervally,intervallz=[intervallareth(intervall.min[voxelswithzerro],intervall.max[voxelswithzerro]) for intervall in (intervallx,intervally,intervallz)]
    print(len(voxelswithzerro))
    if len(voxelswithzerro)>maxvoxelnum:
        depth=j
        break

#print(np.stack([intervallx.min,intervally.min,intervallz.min]).T)

grid=pv.PolyData(np.stack([intervallx.mid(),intervally.mid(),intervallz.mid()]).T)
length=(upperbound-lowerbound)/2**depth
print(time.time()-t0)
#p.add_mesh(PVGeo.filters.VoxelizePoints(dx=length,dy=length,dz=length).apply(grid),opacity=0.5)




PVGeo.filters.VoxelizePoints(dx=length,dy=length,dz="qwqe").apply(grid)


#todo: convert to pyvista code
import vtk
grid = vtk.vtkUnstructuredGrid()#np.arange(-64,64,2**depth),np.arange(-64,64,2**depth),np.arange(-64,64,2**depth)
x, y, z =[intervallx.mid(),intervally.mid(),intervallz.mid()]

#print(x,len(x))
dx=dy=dz = length

n_cells = len(x)



#TODO replace with good intervalls
# Generate cell nodes for all points in data set
# - Bottom
#c_n1 = np.stack(((x - dx / 2), (y - dy / 2), (z - dz / 2)), axis=1)
#c_n2 = np.stack(((x + dx / 2), (y - dy / 2), (z - dz / 2)), axis=1)
#c_n3 = np.stack(((x - dx / 2), (y + dy / 2), (z - dz / 2)), axis=1)
#c_n4 = np.stack(((x + dx / 2), (y + dy / 2), (z - dz / 2)), axis=1)
# - Top
#c_n5 = np.stack(((x - dx / 2), (y - dy / 2), (z + dz / 2)), axis=1)
#c_n6 = np.stack(((x + dx / 2), (y - dy / 2), (z + dz / 2)), axis=1)
#c_n7 = np.stack(((x - dx / 2), (y + dy / 2), (z + dz / 2)), axis=1)
#c_n8 = np.stack(((x + dx / 2), (y + dy / 2), (z + dz / 2)), axis=1)


c_n1 = np.stack((intervallx.min, intervally.min, intervallz.min), axis=1)
c_n2 = np.stack((intervallx.max, intervally.min, intervallz.min), axis=1)
c_n3 = np.stack((intervallx.min, intervally.max, intervallz.min), axis=1)
c_n4 = np.stack((intervallx.max, intervally.max, intervallz.min), axis=1)
# - Top
c_n5 = np.stack((intervallx.min, intervally.min, intervallz.max), axis=1)
c_n6 = np.stack((intervallx.max, intervally.min, intervallz.max), axis=1)
c_n7 = np.stack((intervallx.min, intervally.max, intervallz.max), axis=1)
c_n8 = np.stack((intervallx.max, intervally.max, intervallz.max), axis=1)
# - Concatenate
all_nodes = np.concatenate(
    (c_n1, c_n2, c_n3, c_n4, c_n5, c_n6, c_n7, c_n8), axis=0
)
#all_nodes, ind_nodes = np.unique(all_nodes , return_inverse=True, axis=0)
#if 1:
#    TOLERANCE = length / 2.0
#    all_nodes, ind_nodes = np.unique(np.around(all_nodes / TOLERANCE)*TOLERANCE , return_inverse=True, axis=0)
#    #all_nodes*=TOLERANCE
#else:
#    ind_nodes = np.arange(len(all_nodes))


from vtk.util import numpy_support as nps
cells = vtk.vtkCellArray()
pts = vtk.vtkPoints()



print()

#can be replaced with all_nodes, ind_nodes = np.unique(all_nodes, return_inverse=True, axis=0) if every cube fitts perfectly
#TOLERANCE = length / 2.0
#txy = np.around(all_nodes[:, 0:2] / TOLERANCE)
#all_nodes[:, 0:2] = txy
#unique_nodes, ind_nodes = np.unique(all_nodes, return_inverse=True, axis=0)
#unique_nodes[:, 0:2] *= TOLERANCE
#all_nodes = unique_nodes

#if 1:
#    TOLERANCE = length / 2.0
#    all_nodes, ind_nodes = np.unique(np.around(all_nodes / TOLERANCE)*TOLERANCE , return_inverse=True, axis=0)
#    #all_nodes*=TOLERANCE
#else:
#    ind_nodes = np.arange(len(all_nodes))

# Add unique nodes as points in output
#pts.SetData(interface.convert_array(all_nodes))
pts.SetData(nps.numpy_to_vtk(all_nodes))
#pts.SetData(pv.vtk_points(all_nodes))
#pts=pv.vtk_points(all_nodes)
#pts=pv.PointSet(all_nodes)
print(pts)
# Add cell vertices
j = np.tile(np.arange(8), n_cells)* n_cells
#print(j)
arridx = np.add(j, np.repeat(np.arange(n_cells), 8))
print(arridx)
ids = ind_nodes[arridx].reshape((n_cells, 8))
#ids=np.arange(n_cells*8).reshape((n_cells, 8),order='F')
#print(ids)

cells_mat = np.concatenate(
    (np.ones((n_cells, 1), dtype=np.int_) * 8, ids), axis=1
)
#print(cells_mat)
pv.StructuredGrid()
cells = vtk.vtkCellArray()
cells.SetNumberOfCells(n_cells)
cells.SetCells(
    n_cells, nps.numpy_to_vtk(cells_mat.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE)
)
# Set the output
grid.SetPoints(pts)
grid.SetCells(vtk.VTK_VOXEL, cells)



print(time.time()-t0)

p.add_mesh(grid,opacity=0.5)
p.show()
            





#todo trinary logic
# TODO  
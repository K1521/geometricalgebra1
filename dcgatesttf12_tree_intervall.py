
#import tensorflow as tf
from blademul6 import *
import numpy as np
import pyvista as pv
import time
#import itertools
import functools
import operator
#import PVGeo
from collections import namedtuple

dcga=algebra(8,2)
import sympy

#e1=1,e2=1,e3=1,e4=1,e5=-1,e6=1,e7=1,e8=1,e9=1,e10=-1
#skalar,e1,e2,e3,e4,e6,e7,e8,e9,e5,e10,*_=dcga.allblades()
#multivec=sortgeo(dcga)

#algebra declaration
multivec=sortgeo(dcga)
e1,e2,e3,e4,e6,e7,e8,e9,e5,e10=multivec.monoblades()

dcga.bladenames="1,2,3,4,6,7,8,9,5,10".split(",")
print(e1,e2,e3,e4,e6,e7,e8,e9,e5,e10)
#eo1=0.5*e5-0.5*e4
#eo2=0.5*e10-0.5*e9
eo1=(e5-e4)/2
eo2=(e10-e9)/2
ei1=e4+e5
ei2=e9+e10


def point1(x,y,z):
    return e1*x+e2*y+e3*z+ei1*((x*x+y*y+z*z)/2)+eo1
def point2(x,y,z):
    return e6*x+e7*y+e8*z+ei2*((x*x+y*y+z*z)/2)+eo2

def point(x,y,z):
    return point1(x,y,z).outer(point2(x,y,z))

eo=eo1.outer(eo2)
ei=ei1.outer(ei2)
Txx=e6.outer(e1)
Tyy=e7.outer(e2)

def CGA1_Plane(x,y,z,h):
    vec=(x*e1+y*e2+z*e3)
    #return vec/sympy.sqrt(vec.inner(vec).toscalar())+h*ei1
    return vec+h*sympy.sqrt(vec.inner(vec).toscalar())*ei1
    #return vec+ei1
def CGA2_Plane(x,y,z,h):
    vec=(x*e6+y*e7+z*e8)
    #return vec/sympy.sqrt(vec.inner(vec).toscalar())+h*ei2
    return vec+h*sympy.sqrt(vec.inner(vec).toscalar())*ei2
    #return vec+ei2
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
t=toroid(1,.5)
p=Plane(0.1,0.1,0.1,0.5)



#for x in np.mgrid[-1:1:5j]:
#    print(p.inner(point(x,0,0)))



vis=t^Plane(0.1,0.1,0.001,0.5)^Plane(0.001,0.001,0.1,0.1)
#pyvista setup
pv.set_plot_theme('dark')
p = pv.Plotter()
p.add_axes()
p.show_grid()

t0=time.time() 
step=101


x,y,z=multivec.skalar(sympy.symbols('x y z'))
expr=point(x,y,z).inner(vis)
equations=[sympy.simplify(blade.magnitude*2**10) for blade in expr.lst]
equationsnew=[]
for e in equations:
    if e.is_number:
        #print(sympy.factor(e,deep=True))
        continue
    #equationsnew.extend([f for f,p in sympy.factor_list(e)[1]])
    equationsnew.append(sympy.Mul(*[f for f,p in sympy.factor_list(e)[1]]))
    #print(sympy.factor(e,deep=True).args)
    #equationsnew.append(sympy.factor(e))
    #equationsnew.append(sympy.factor(e,deep=True))
equations=equationsnew

#equations="""
#(64*x**4 + 128*x**2*y**2 + 128*x**2*z**2 + 96*x**2 + 64*y**4 + 128*y**2*z**2 - 160*y**2 + 64*z**4 + 96*z**2 + 35)
#y*(16*x - 1)
#(4*x**4 - 16*x**3 + 8*x**2*y**2 + 8*x**2*z**2 + 7*x**2 - 16*x*y**2 - 16*x*z**2 - 13*x + 4*y**4 + 8*y**2*z**2 - 9*y**2 + 4*z**4 + 7*z**2 + 3)
#(16*x - 1)
#(x**4 - 8*x**3 + 2*x**2*y**2 + 2*x**2*z**2 + 14*x**2 - 8*x*y**2 - 8*x*z**2 - 8*x + y**4 + 2*y**2*z**2 - 2*y**2 + z**4 + 2*z**2 + 1)
#(4*x**4 + 112*x**3 + 8*x**2*y**2 + 8*x**2*z**2 - x**2 + 112*x*y**2 + 112*x*z**2 + 83*x + 4*y**4 + 8*y**2*z**2 - 17*y**2 + 4*z**4 - z**2 - 3)
#(x**4 + 24*x**3 + 2*x**2*y**2 + 2*x**2*z**2 - 116*x**2 + 24*x*y**2 + 24*x*z**2 + 32*x + y**4 + 2*y**2*z**2 - 4*y**2 + z**4 - 1)
#(x**4 + 56*x**3 + 2*x**2*y**2 + 2*x**2*z**2 + 778*x**2 + 56*x*y**2 + 56*x*z**2 - 56*x + y**4 + 2*y**2*z**2 - 6*y**2 + z**4 - 2*z**2 + 1)
#""".split("\n")
#equations.remove("")
#equations.remove("")
#equations=[sympy.parse_expr(e) for e in equations]

print(equations)


equations=list({sympy.sstr(e,full_prec=True):e for e in equations}.values())
print("------------------------------------------------------")
print(*equations,sep="\n")
#print(sorted(equations,key=str))

#equations=sorted(equations,key=str)
#print(sympy.python(equations[0]))
#exit()
#print(sympy.factor_list(equations[0]))


def evaluatefun(equation,cache):
    def evaluate(equation):
        eqs=sympy.sstr(equation,full_prec=True)
        if ret:=cache.get(eqs,None):
            return ret
        elif equation.is_number:
            ret=float(equation)
        elif equation.func==sympy.Add:
            ret=functools.reduce(operator.add,map(evaluate,equation.args))
        elif equation.func==sympy.Mul:
            ret=functools.reduce(operator.mul,map(evaluate,equation.args))
        elif equation.func==sympy.Pow:
            a,e=equation.args
            ret=evaluate(a)**evaluate(e)
            #print(equation.args)
        else:
            print(equation.func)
            raise Exception(f"type {equation.func} not supported")
        cache[eqs]=ret
        return ret

    
    return evaluate(equation)
print(evaluatefun(equations[0],cache={"x":1,"y":3,"z":4}))

#exit()
#print(print(sympy.simplify(sum(abs(blade.magnitude) for blade in expr.lst))))

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
        #if other==0:return 0
        #if other==1:return self
        if other>0:
            return intervallareth(self.min*other,self.max*other)
        else:
            return intervallareth(self.max*other,self.min*other)
    def __rmul__(self,other):
        return self*other
    
    def __add__(self,other):
        if isinstance(other,intervallareth):
            return intervallareth(self.min+other.min,self.max+other.max)
        #if other==0:return self
        return intervallareth(self.min+other,self.max+other)
    def __radd__(self,other):
        return self+other
    
    def __sub__(self,other):
        if isinstance(other,intervallareth):
            return intervallareth(self.min-other.max,self.max-other.min)
        #if other==0:return self
        return intervallareth(self.min-other,self.max-other)
    def __rsub__(self,other):
        return intervallareth(other-self.max,other-self.min)
    def __neg__(self):
        return intervallareth(-self.max,-self.min)

    def mid(self):
        #return (self.min+self.max)/2
        #print(self.min,self.max)
        return np.average([self.min,self.max], axis=0)
    def containsnum(self,num=0,maxdelta=0):
        return (self.min<=num+maxdelta)&(self.max>=num-maxdelta)
    def __abs__(self):
        combis=[abs(self.min),abs(self.max)]
        return intervallareth(np.where((self.min<=0)&(0<=self.max), 0, np.min(combis, axis=0)),np.max(combis, axis=0))

    def __pow__(self,exponent):
        if exponent < 0:
            raise ValueError("Power must be non-negative")
        elif exponent == 0:
            #return 1
            return intervallareth(1,1)
        
        
        if exponent % 2 == 0:
            combis=[self.min**exponent,self.max**exponent]
            return intervallareth(np.where((self.min<=0)&(0<=self.max), 0, np.min(combis, axis=0)),np.max(combis, axis=0))
        else:
            return intervallareth(self.min**exponent,self.max**exponent)
        


t0=time.time()



lowerbound=-64
upperbound=64
depth=16
maxvoxelnum=100000

intervallx,intervally,intervallz=[intervallareth(np.array([lowerbound]),np.array([upperbound])) for _ in range(3)]



lastvoxnum=1
for j in range(1,depth+1):
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
    #voxelsintervall=point(intervallx,intervally,intervallz).inner(vis)#calculate #TODO
    cache={"x":intervallx,"y":intervally,"z":intervallz}
    #print(evaluatefun(equations[0],cache))
    #voxelswithzerro=sum(abs(evaluatefun(e,cache)) for e in equations).containsnum()#TODO
    voxelswithzerro=np.all([evaluatefun(e,cache).containsnum() for e in equations],axis=0)#TODO

    #TODO wenn ableitung sum(abs(evaluatefun(e,cache)) for e in equations).containsnum() nach x,y,z contains 0,0,0 

    intervallx,intervally,intervallz=[intervallareth(intervall.min[voxelswithzerro],intervall.max[voxelswithzerro]) for intervall in (intervallx,intervally,intervallz)]
    print(len(voxelswithzerro))
    if len(voxelswithzerro)>maxvoxelnum:
        depth=j
        break


print(time.time()-t0)






#todo: convert to pyvista code
import vtk
grid = vtk.vtkUnstructuredGrid()


n_cells = len(intervallx.min)


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
#all_nodes = np.concatenate(
#    (c_n1, c_n2, c_n3, c_n4, c_n5, c_n6, c_n7, c_n8), axis=0
#)
all_nodes=np.empty((n_cells*8,3))
all_nodes[0::8] = c_n1
all_nodes[1::8] = c_n2
all_nodes[2::8] = c_n3
all_nodes[3::8] = c_n4
all_nodes[4::8] = c_n5
all_nodes[5::8] = c_n6
all_nodes[6::8] = c_n7
all_nodes[7::8] = c_n8

#print(all_nodes[0::8,(0,1)])
all_nodes, ind_nodes = np.unique(all_nodes , return_inverse=True, axis=0)
#ind_nodes=np.arange(n_cells*8)
from vtk.util import numpy_support as nps
cells = vtk.vtkCellArray()
pts = vtk.vtkPoints()



print()


# Add unique nodes as points in output
#pts.SetData(interface.convert_array(all_nodes))
pts.SetData(nps.numpy_to_vtk(all_nodes))
#pts.SetData(pv.vtk_points(all_nodes))
#pts=pv.vtk_points(all_nodes)
#pts=pv.PointSet(all_nodes)
#print(pts)
# Add cell vertices
#j = np.tile(np.arange(8), n_cells)* n_cells
#print(j)
#arridx = np.add(j, np.repeat(np.arange(n_cells), 8))
arridx=np.arange(8*n_cells)
#print(arridx)
ids = ind_nodes[arridx].reshape((n_cells, 8))
#ids=np.arange(n_cells*8).reshape((n_cells, 8),order='F')
#print(ids)

#cells_mat = np.concatenate(
#    (np.ones((n_cells, 1), dtype=np.int_) * 8, ids), axis=1
#)
cells_mat = np.concatenate(
    (np.full((n_cells, 1),8) , ids), axis=1
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
#grid = pv.UnstructuredGrid(cells_mat.ravel(), np.array([pv.CellType.VOXEL]*len(cells_mat), np.int8), all_nodes.ravel())
p.add_mesh(grid,opacity=0.5)
p.show()


#entweder f'==0 oder vorzeichenwechsel in den punkten

import tensorflow as tf
from blademul6 import *
import numpy as np
import pyvista as pv
import time
import itertools
import PVGeo
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



vis=t
#pyvista setup
pv.set_plot_theme('dark')
p = pv.Plotter()
p.add_axes()
p.show_grid()

t0=time.time() 
step=101


from sympy.parsing.sympy_parser import parse_expr
class dagexpr:
    def __init__(self,expr,parents=None):
        self.expr=expr
        if parents:
            self.parents=[p for p in parents if isinstance(p,dagexpr)]
            self.expr=expr.format(*["{}" if isinstance(p,dagexpr) else str(p) for p in parents ])
        else:
            self.parents=[]
        #self.parents=parents or []
        self.children=[]
        for p in self.parents:
            p.children.append(self)
    
    def __add__(self,other):
        if other==0:
            return self
        return dagexpr("({}+{})",[self,other])
    def __radd__(self,other):
        return self+other#dagexpr(f"({{}}+{other})",[self])
    def __mul__(self,other):
        if other==1:
            return self
        elif other==0:
            return 0
        return dagexpr("({}*{})",[self,other])
    def __rmul__(self,other):
        return self*other#dagexpr(f"({{}}*{other})",[self])
    def __sub__(self,other):
        if other==0:
            return 0
        return dagexpr("({}-{})",[self,other])
    def __rsub__(self,other):
        return dagexpr(f"({other}-{{}})",[self])
    def __neg__(self):
        return dagexpr("-({})",[self])
    def __abs__(self):
        return dagexpr("abs({})",[self])
    
    def asexpr(self,collapse=True):
        #Topological sorting
        #passed=set()
        #stack=[self]
       # sorting=[]
        #while stack:
          #  act=stack.pop()
            #if act in passed:
            #    continue
            #sorting.append(act)
            #passed.add(act)
           #stack.extend(act.parents)
        #sorting=sorting[::-1]
        #print( [x.expr for x in sorting])

        passed=set()
        stack=[self]
        sorting=[]
        while stack:
            act=stack[-1]
            for p in act.parents:
                if p not in passed:
                    stack.append(p)
                    break
            else:
                passed.add(act)
                sorting.append(act)
                stack.pop()


            
            #if passed.issuperset(act.parents):
            #    sorting.append(stack.pop())
            #else:


        #sorting=sorting[::-1]
        #print( [x.expr for x in sorting])

        
        Lineinfo = namedtuple('Lineinfo', 'name expr collapsible')
        lines=dict()#dagexpr=>[name,expr,collapsible]
        for i,node in enumerate(sorting):
            #expr=parse_expr(node.expr.format(*[lines[p].name for p in node.parents]))
            expr=node.expr.format(*[lines[p].name for p in node.parents])
            collapsible=len(node.children)==1 or len(node.parents)==0
            if collapse and collapsible:#all(len(p.children)==1 for p in node.parents):
                name=expr
            else:
                name=f"L{i}"

            lines[node]=Lineinfo(name,expr,collapsible)



        #return [x.expr for x in sorting]
        return [f"{name}={expr}"for name,expr,collapsible in lines.values()if not(collapsible and collapse)]
    


        




x,y,z=multivec.skalar([dagexpr("X"),dagexpr("Y"),dagexpr("Z")])


expr=point(x,y,z).outer(vis)

#print(sum(abs(blade.magnitude) for blade in x.lst).asexpr())
print(sum(abs(blade.magnitude) for blade in expr.lst).asexpr())

"""class term:
    def __init__(self,expr) -> None:
        self.expr=expr
    def __mul__(self,othe):
        if othe==1:
            return self
        elif othe==0:
            return 0
        return term(f"({self})*({othe})")
    def __rmul__(self,othe):
        if othe==1:
            return self
        elif othe==0:
            return 0
        return term(f"({othe})*({self})")
    def __add__(self,othe):
        if othe==0:
            return self
        return term(f"({self})+({othe})")
    def __radd__(self,othe):
        if othe==0:
            return self
        return term(f"({othe})+({self})")
    def __neg__(self):
        return term(f"-({self})")
    def __repr__(self):
        return self.expr
    def __format__(self,form):
        return self.expr
t=toroid(1,0.5).inner(point(term("X"),term("Y"),term("Z")))
print(toroid(1,0.5).outer(point(term("X"),term("Y"),term("Z"))))"""
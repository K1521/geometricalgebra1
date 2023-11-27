
import numpy as np

class polynom3d:


    def __init__(self,d=None):
        self.d=d or dict()
    
    def __add__(a,b):
        if isinstance(b,polynom3d):
            return polynom3d({exeyez:value for exeyez in a.d.keys()|b.d.keys() if (value:=a.d.get(exeyez,0)+b.d.get(exeyez,0))})
        d=a.d.copy()
        value=d.get((0,0,0), 0)+b
        if value:
            d[(0,0,0)]=value
        return polynom3d(d)
    def __radd__(a,b):
        return a+b
    def __sub__(a,b):
        return a+(-b)
    def __rsub__(a,b):
        return (-a)+b
        #return polynom3d({exeyez:value for exeyez in a.d.keys() if (value:=b-a.d.get(exeyez,0))})
    def __neg__(self):
        return polynom3d({exeyez:-value for exeyez,value in self.d.items()})
    def __mul__(a,b):
        if isinstance(b,polynom3d):
            d=dict()
            for (x1,y1,z1),v1 in a.d.items():
                for (x2,y2,z2),v2 in b.d.items():
                    k=(x1+x2,y1+y2,z1+z2)
                    d[k]=d.get(k,0)+v1*v2
            return polynom3d(d)
        return polynom3d({exeyez:value*b for exeyez,value in a.d.items()})
    def __rmul__(a,b):
        return a*b
    def __truediv__(a,b):
        if isinstance(b,polynom3d):
            raise Exception("cant divide polys")
        return a*(1/b)
    def __str__(self):
        s=[]
        for xyz,v in self.d.items():
            if v==0:
                continue
            s.append("+" if v>0 else "-")
            v=abs(v)
            mul=[]
            if v!=1:
                mul.append(str(v))
            for var,exp in zip("xyz",xyz):
                if exp==1:
                    mul.append(var)
                elif exp!=0:
                    mul.append(f"{var}**{exp}")
            if mul:
                s.append("*".join(mul))
        return " ".join(s).lstrip("+ ")

from dcgasym1 import *
import pyvista as pv       

x,y,z=multivec.skalar([polynom3d({(1,0,0):1}),polynom3d({(0,1,0):1}),polynom3d({(0,0,1):1})])
print((x*x*y+z))



t=toroid(1,.5)
p=Plane(0.1,0.1,0.1,0.5)

vis=t^Plane(0.1,0.1,0.001,0.5)^Plane(0.001,0.001,0.1,0.1)

polys=[b.magnitude for b in (point(x,y,z).inner(vis)).lst]
#x,y,z=[polynom3d({(1,0,0):1}),polynom3d({(0,1,0):1}),polynom3d({(0,0,1):1})]
#polys=[x*x+1,x*z,z+1,x*z+2*z+1]
powers=sorted(set.union(*[set(p.d)for p in polys]),key=lambda p:(-p.count(0),sum(p),p))
if powers[0]==(0,0,0):
    powers.pop(0)
    powers.append((0,0,0))
arrays=np.array([[p.d.get(power,0)for power in powers]for p in polys])
#print(arrays)


#arrays=np.array([[1,3,-2,5],[3,5,6,7],[2,4,3,8]])
import sympy
print(np.min(arrays))
#print(powers)
m,r=sympy.Matrix(arrays).rref()
print(m)
#print(m[r])
print(len(arrays))
import time

wort="hallo"
for i in range(10):
    print(end=("\r"+"["+wort[:i].ljust(len(wort),"_")+"]"))
    time.sleep(0.1)

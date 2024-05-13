import numpy as np
import sympy as sy

#ps=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1]])
#ns=
def norm(x):
    return sy.sqrt(sum(x**2))
a=np.array(sy.symbols("ax,ay,az"))
p=np.array(sy.symbols("px,py,pz"))
d=np.array(sy.symbols("dx,dy,dz"))
print(a)
print(p)
dist=(norm(np.cross(a-p,d))/norm(d)).simplify()
#print(dist)
print()
print(dist.diff(p[0]).simplify())
print()
#print(sum(np.cross(a-p,d)**2).diff(p[0]).simplify())
print()
#print(np.cross(a-p,d)[0].simplify())
#print(np.cross(a-p,d)[1].simplify())
#print(np.cross(a-p,d)[2].simplify())
#print((np.cross(np.cross(a-p,d),d)/dist)[0].simplify())

print((np.cross(np.cross(a-p,d),d)/(norm(np.cross(a-p,d))*norm(d)))[0].simplify())
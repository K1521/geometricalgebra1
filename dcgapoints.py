
import sympy as sy
import algebra.dcga as dcga
from algebra.blademul import blade


p=dcga.point(1,2,3)
#point.compress()
#print(point)


#point=dcga.point(*sy.symbols("x,y,z"))
#for i in range(len(point.lst)):
#    point.lst[i]=blade(point.lst[i].basis,point.lst[i].magnitude.simplify())
#print(point)

t=dcga.toroid(1,.5)
print(t^p)
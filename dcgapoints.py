
import sympy as sy
import algebra.dcga as dcga
from algebra.blademul import blade

print("hi")
p=dcga.point(1,2,3)
#point.compress()
#print(point)



def printeach(o):
   for i in range(len(o.lst)):
      print("+",type(o)(o.algebra,[o.lst[i]]))


point=dcga.point(*sy.symbols("x,y,z"))
for i in range(len(point.lst)):
   #point.lst[i]=blade(point.lst[i].basis,point.lst[i].magnitude.simplify())#.expand()
   #point.lst[i]=blade(point.lst[i].basis,point.lst[i].magnitude.expand())#.expand()
   point.lst[i]=blade(point.lst[i].basis,point.lst[i].magnitude)#.expand()

print(len(point.lst))
t=dcga.toroid(1,.5)^dcga.Plane(1,1,1,0)

result=point#.inner(t)

for i in range(len(result.lst)):
   #point.lst[i]=blade(point.lst[i].basis,point.lst[i].magnitude.simplify())#.expand()
   #point.lst[i]=blade(point.lst[i].basis,point.lst[i].magnitude.expand())#.expand()
   result.lst[i]=blade(result.lst[i].basis,result.lst[i].magnitude.simplify())#.expand()
printeach(result)
print(len(result.lst))
#print(t.inner(point).toscalar().simplify())

# p=dcga.Plane(0.1,0.1,0.1,0)
# p2=dcga.Plane(0.2,0.3,0.4,0.5)
# # o=point.inner(t)


# # printeach(point)

# from intervallarethmetic.intervallareth3d1 import inter3d
# x=inter3d({(1,0,0):1})
# y=inter3d({(0,1,0):1})
# z=inter3d({(0,0,1):1})
# point=dcga.point(x,y,z)
# printeach(point)

# o=point.inner(t^p^p2)
# allkeys=set()
# for i in range(len(o.lst)):
#    allkeys=allkeys.union([k for k,v in o.lst[i].magnitude.coeffs.items() if v!=0])
# print(sorted(allkeys))
# print(len(allkeys),len(o.lst))



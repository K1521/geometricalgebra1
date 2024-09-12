



import numpy as np

a=np.arange(16)
a=np.array([0,1,4,3])
a=np.array([0,1,2,4,3,5,6,7])
b=a[:,None]
print(a)

#print((a&b)!=0)

#(((~blade1.basis)&blade2.basis) and (blade1.basis&(~blade2.basis))):

#print(np.logical_and(((~a)&b)!=0 , ((~b)&a)!=0))
i=np.logical_and(((~a)&b)!=0 , ((~b)&a)!=0)
o=(a&b)!=0

print(i==o)



def geos(x):
    geos=np.empty(x.shape,dtype=object)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            s=str(x[i,j])#.replace(".0*","*").replace("-1*","-").replace("+1*","+").replace("+1","1").replace("+0","0")
            n,_,b=s.partition("*")
            n=float(n)
            if n==0:
                s="0"
            else:
                if n==-1:
                    s="-"+(b or "1")
                if n==1:
                    s=(b or "1")

            
            geos[i,j]=s.rjust(5," ")
    return geos


from algebra.blademul import sortgeo,algebra
dcga=algebra(3,0)
#algebra declaration
multivec=sortgeo(dcga)
dcga.bladenames="1,2,3".split(",")

e1,e2,e3=multivec.monoblades()

print(e1^e2)
blades=[multivec+1,e1,e2,e3,e1^e2,e1^e3,e2^e3,e1^e2^e3]
geo=np.empty([8, 8],dtype=object)
#print(geo)
for i in range(8):
    for j in range(8):
        geo[i,j]=blades[i]*blades[j]


print(geos(geo))

inner=np.empty([8, 8],dtype=object)
#print(geo)
for i in range(8):
    for j in range(8):
        inner[i,j]=blades[i].inner(blades[j])-(blades[i]*blades[j]+blades[j]*blades[i])/2
        inner[i,j]=(blades[i]*blades[j]+blades[j]*blades[i])/2

print(geos(inner))

outer=np.empty([8, 8],dtype=object)
#print(geo)
for i in range(8):
    for j in range(8):
        outer[i,j]=blades[i].outer(blades[j])-(blades[i]*blades[j]-blades[j]*blades[i])/2
        outer[i,j]=(blades[i]*blades[j]-blades[j]*blades[i])/2

print(geos(outer))
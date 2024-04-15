from intervallarethmetic1 import intervallareth

import math
import sympy
import numpy as np


def interpow(intervall,offset,n):
    #intervall**n
    #=(intervall-offset+offset)**n
    #=((intervall-offset)+offset)**n
    #=(intervalloff+offset)**n
    intervalloff=intervall-offset
    s=0
    for k in range(0,n+1):
        s+=math.comb(n,k)*offset**k*intervalloff**(n-k)
    return s

#for i in range(-10,10):
#    print(i,interpow(intervallareth(5,6),i,3))

x,a=sympy.symbols("x a")
f=x+0.05*x**2
#f=sympy.simplify(f.subs(x,x-30))
fa=sympy.Poly(f.subs(x,x+a),x).as_expr()
f4=sympy.expand(f)

zp=intervallareth(0,1)
fastr=str(fa).replace("a","i").replace("x","zp")

fastr4=str(f4).replace("x","ix")



print(fastr)
print(fastr4)

xlower=-10
xupper=10
#xpoints=np.linspace(-100,100,1000)
xpoints=np.arange(xlower,xupper,dtype=float)

i=-10


mima=eval(fastr)
print(mima)


ix=intervallareth(i,i+1)
mima=eval(fastr4)
print(mima)

print()


intervall=zp
offset=i
print((intervall+offset)+0.05*(intervall**2+2*offset*intervall+offset**2))

print(0.05*offset**2+offset+intervall+0.05*(2*offset*intervall)+0.05*(intervall**2))
print(0.05*offset**2+offset+intervall+0.1*offset*intervall+0.05*intervall**2)
print(0.05*offset**2+offset+(1+0.1*offset)*intervall+0.05*intervall**2)


print((1+0.1*offset)*intervall)
print((1-1)*intervall)
print(intervall-intervall)
"""
for j in range(-200,200):
    j/=10
    intervall=intervallareth(j,j+1)
    intervallout=intervallareth(-float("inf"),float("inf"))
    
    #print(intervallout)
    for i in range(-200,200):
        i/=10
        intervallout2=interpow(intervall,i,3)-interpow(intervall,i,2)+interpow(intervall,i,4)
        intervallout.min=max(intervallout.min,intervallout2.min)
        intervallout.max=min(intervallout.max,intervallout2.max)
    #print(intervallout)
    intervallout2=interpow(intervall,0,3)-interpow(intervall,0,2)+interpow(intervall,i,4)
    if abs(intervallout.min-intervallout2.min)>0.0001 or abs(intervallout.max-intervallout2.max)>0.0001:
        print(intervallout)
        print(intervallout2)
        print(j)
        break
"""
"""
for j in range(-200,200):
    j/=10
    intervall=intervallareth(j,j+1)
    #f=-x**5-0.01*3*x**6+0.001*x**7

    intervallout=-interpow(intervall,0,3)-0.01*interpow(intervall,0,2)+0.001*interpow(intervall,0,4)
    for i in range(-200,200):
        i/=10
        intervallout2=-interpow(intervall,i,3)-0.01*interpow(intervall,i,2)+0.001*interpow(intervall,i,4)

        if intervallout.min+0.001<intervallout2.min or intervallout.max-0.001>intervallout2.max:
            print(intervallout)
            print(intervallout2)
            print(j)
            break

"""

#(intervall+offset)+0.05*(intervall+offset)**2
#=(intervall+offset)+0.05*(intervall**2+2*offset*intervall+offset**2)


#TODO Domain intervall arethmetic

#(intervall**2+2*offset*intervall+offset**2)


#f.subs(x,i+o)
#f.subs(x,-i+1+o)
#f.subs(x,(i+1)*0.5+o)
#f.subs(x,(i-i+1)*0.5+o)
#generalisiert f.subs(x,lerp(i,-i+1)+o)

#zp=intervallareth(0,1)
#zn=intervallareth(-1,0)
#fastr2=str(fa2).replace("a","i").replace("x","zn")#zn=intervall zero negative 

#inp=intervallareth(-1,1)=zp*2-1
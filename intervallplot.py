from intervallarethmetic.intervallareth3d1 import inter3d
from intervallarethmetic.intervallarethmetic1 import intervallareth
import numpy as np
from matplotlib import pyplot as plt
import math
import sympy
Ipm=intervallareth(-1,1)
Ip=intervallareth(0,1)
Im=intervallareth(-1,0)


def mkintervallfromnp(arr):
    return intervallareth(arr.min(axis=-1),arr.max(axis=-1))
def mkintervalltonp(intervall):
    return np.linspace(intervall.min,intervall.max,10,True)[None]

def plotintervall(x,y):
    print(y.min)
    plt.plot(x,y.min,label="min")
    plt.plot(x,y.max,label="max")
    plt.show()

#f1=lambda x:-x**5-0.01*3*x**6+0.001*x**7
#f=lambda x:f1(x-100)
#f=lambda x:x**3-2*x**2+x
#f=lambda x:(x+1)**2-(x)**2
#f=lambda x:x - x**3/6 + x**5/120 - x**7/5040
#f=lambda x:1*x**1/1-1*x**3/6+1*x**5/120-1*x**7/5040+1*x**9/362880
#f=lambda x:-5.302721495*10**(-8)*x**(10)+0.000006386725359*x**(9)-0.0003226963201*x**(8)+0.009013570906*x**(7)-0.1535425783*x**(6)+1.652938279*x**(5)-11.23137846*x**(4)+46.50445519*x**(3)-108.1093811*x**(2)+118.6680891*x-39.98491616
f=lambda x:x**(2)
for i in range(10):
    n=2*i+1
    print(f"{(-1)**i}*x**{n}/{math.factorial(n)}")
x,I=sympy.symbols("xx I")
s=str(sympy.Poly(f(x).subs(x,I+x),I).as_expr())
print(s)
I=Ipm
x=np.linspace(-10,10,200)*3
xnp=mkintervalltonp(Ipm)+x[:,None]
xi=I+x
#print(xi)
print()
#plotintervall(x,f(xi))
#plotintervall(x,mkintervallfromnp(f(xnp)))
y=f(xi)
plt.plot(x,y.min,label="min intervall",c="r")
plt.plot(x,y.max,label="max intervall",c="r")
y=mkintervallfromnp(f(xnp))

plt.plot(x,y.min,label="min real",c="b")
plt.plot(x,y.max,label="max real",c="b")

plt.plot(x,f(x),label="real",c="g")

#f=lambda x:I**9/362880 - I**7/5040 + I**5/120 - I**3/6 + I*x**8/40320 + I + x**9/362880 + x**7*(I**2/10080 - 1/5040) + x**6*(I**3/4320 - I/720) + x**5*(I**4/2880 - I**2/240 + 1/120) + x**4*(I**5/2880 - I**3/144 + I/24) + x**3*(I**6/4320 - I**4/144 + I**2/12 - 1/6) + x**2*(I**7/10080 - I**5/240 + I**3/12 - I/2) + x*(I**8/40320 - I**6/720 + I**4/24 - I**2/2 + 1)
y=[eval(s) for xx in x]
#print(y)
plt.plot(x,[i.min for i in y],label="min intervall moved",c="purple")
plt.plot(x,[i.max for i in y],label="max intervall moved",c="purple")


#ix=inter3d({(1,0,0):1})
#iy=inter3d({(0,1,0):1})
#iz=inter3d({(0,0,1):1})
#y=f(ix+x).intervallnp()
#plt.plot(x,y.min,label="min intervall moved",c="purple")
#plt.plot(x,y.max,label="max intervall moved",c="purple")
#plt.plot(x,(x-1)**2,label="max intervall moved",c="y")
#I**2 + 2*I*xx + xx**2
#0+2*(x-1)+(x-1)**2
#plt.plot(x,0-2*x+(x)**2,label="max intervall moved",c="y")
plt.show()


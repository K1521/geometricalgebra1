from intervallarethmetic.intervallareth3d1 import inter3d
from intervallarethmetic.intervallareth3d2 import poly3d
from intervallarethmetic.intervallarethmetic1 import intervallareth
import numpy as np
from matplotlib import pyplot as plt



f=lambda x:x*x
f=lambda x:2*x**3-5*x+3*x-7
f=lambda x:2*(x+3)**2-x*x
f=lambda x:x*x-x*x
x=np.linspace(-10,10,100)
fx=f(x)

I=intervallareth(-1,1)
ix=intervallareth(x)


# y=(I+ix)**2

# plt.plot(x,y.min,color='red',linewidth=4)
# plt.plot(x,y.max,color='red',linewidth=4)

# y=ix**2+I**2+2*I*ix

# plt.plot(x,y.min,color='blue',linewidth=4)
# plt.plot(x,y.max,color='blue',linewidth=4)


y=f(I+x)

plt.plot(x,y.min-fx,color='black')
plt.plot(x,y.max-fx,color='black')




expr=f(poly3d.ix)
y=expr.intervallnp(x=x,delta=1)
print(expr)
plt.plot(x,y.min-fx,color='orange')
plt.plot(x,y.max-fx,color='orange')




expr=f(inter3d.ix*1+x)
y=expr.intervallnp()
plt.plot(x,y.min-fx,color='green')
plt.plot(x,y.max-fx,color='green')

plt.show()


#y=ix**2+I**2+2*I*ix=ix**2+I*(ix+I*2)
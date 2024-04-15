import matplotlib.pyplot as plt
import numpy as np

x = np.mgrid[-0.5:1.5:100*1J]


y1=1
y1_=-1
y2=1
y2_=0.5


c=y1_
d=y1
#y2=a+b+c+d
#y2_=3*a+2*b+c

#y2_-2*y2=3*a+2*b+c-2*(a+b+c+d)
a=y2_-2*y2+c+2*d
b=-2*c-3*d+3*y2-y2_

f=a*x**3+b*x**2+c*x+d
f_=3*a*x**2+2*b*x+c



print(sum(1 for i in range(2**12) if 3<=i.bit_count()<=4))

plt.plot(x, f)
plt.show()

"""
print("Hello World")

c=1
print(1,2)
for i in range(3,1000,2):
    
    for j in range(3,int(i**0.5+2),2):
        if i%j==0:
            break
    else:
        c+=1
        print(c,i)
        
"""

import numpy as np

arrays=np.array([[1,3,-2,5],[3,5,6,7],[2,4,3,8]])[:,::-1]
import sympy

print(arrays[:,:3]@np.array([-15,8,2]))

print(np.min(arrays))

m,r=sympy.Matrix(arrays).rref()
print(m)
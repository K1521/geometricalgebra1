import numpy as np

a=np.array([1,2,3,4,5,6]).reshape((-1,3))

b=np.array([7,8])
print(a)
print(b)
print(np.hstack((a,b)))



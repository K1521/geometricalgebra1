import itertools
import numpy as np

v=np.array(list(v[::-1] for v in itertools.product((0,1),repeat=3)))
print(np.array([[0,0,0],[5,5,5]])+v[:,None,:])
print(np.array([[0,0,0],[5,5,5]])[:,None,:]+v)
print(np.array([[0,0,0],[5,5,5]])[:,None,:]+v)

print(np.array([[1,2],[3,4]])+np.array([[[4,5]],[[5,6]],[[6,7]]]))
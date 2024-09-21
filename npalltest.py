
import numpy as np


dat=[np.array([True,False,True]),np.array([True,True,True]),True]

np.all([x for x in dat if not x is True],axis=0)
print(np.atleast_1d(*dat))
#np.all(np.atleast_1d(*dat),axis=0)

#np.all()


print(np.array([True,True,True])*np.array([True]))
print(np.broadcast_arrays(*dat))
print(np.all(np.broadcast_arrays(*dat),axis=0))

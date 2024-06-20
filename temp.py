from algebra.dcga import *
#algebra.dcga.mode="numpy"
import pyvista as pv
import time
#import functools
#import operator
import numpy as np
#import tensorflow as tf
#import matplotlib.pyplot as mplt

t=toroid(1,.5)
p=Plane(0.1,0.1,0.1,0.5)
pt=p^t
x=point(1,2,3)
print(pt^x)
#print(x^x)
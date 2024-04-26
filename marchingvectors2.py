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

vis=t#^p#^Plane(0.1,0.1,0.001,0.5)#^Plane(0.001,0.001,0.1,0.1)



print(p)


#pyvista setup
pv.set_plot_theme('dark')
plt = pv.Plotter()
plt.add_axes()
plt.show_grid()

t0=time.time() 
step=101







from intervallarethmetic.intervallareth3d1 import inter3d
from intervallarethmetic.intervallarethmetic1 import intervallareth
from intervallarethmetic.voxels import Voxels
t0=time.time()




depth=16
maxvoxelnum=10000

voxels=Voxels(64)
#intervallx,intervally,intervallz=intervallx+17.1225253,intervally+13.127876,intervallz+32.135670
zerrovec=np.zeros(3)

lastvoxnum=1
ix=inter3d({(1,0,0):1})
iy=inter3d({(0,1,0):1})
iz=inter3d({(0,0,1):1})
for j in range(1,depth+1):


    intervallx,intervally,intervallz=voxels.intervallarethpoints()
    x,y,z=ix*voxels.delta/2+intervallx.mid(),iy*voxels.delta/2+intervally.mid(),iz*voxels.delta/2+intervallz.mid()
    
    p=point(ix*voxels.delta/2+intervallx.mid(),
            iy*voxels.delta/2+intervally.mid(),
            iz*voxels.delta/2+intervallz.mid())
    #print("p")
    
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile(builtins=False)
    pr.enable()
    expr=p.inner(vis)
    
    #plt.add_mesh(voxels.gridify(),opacity=0.5)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(10)

   
    voxelswithzerro=np.all([blade.magnitude.intervallnp().containsnum(0) for blade in expr.lst],axis=0)
    voxels.removecells(voxelswithzerro)
    
    
    #print(voxelswithzerro)
    #print(expr.lst)
    print(len(voxelswithzerro),j,len(voxelswithzerro)/8**j)
    if len(voxelswithzerro)>maxvoxelnum:
        depth=j
        break
    voxels.subdivide()

intervallx,intervally,intervallz=voxels.intervallarethpoints()
c_n0 = (intervallx.min, intervally.min, intervallz.min)#000
c_n1 = (intervallx.max, intervally.min, intervallz.min)#100
c_n2 = (intervallx.min, intervally.max, intervallz.min)#010
c_n3 = (intervallx.max, intervally.max, intervallz.min)#110
c_n4 = (intervallx.min, intervally.min, intervallz.max)#001
c_n5 = (intervallx.max, intervally.min, intervallz.max)#101
c_n6 = (intervallx.min, intervally.max, intervallz.max)#011
c_n7 = (intervallx.max, intervally.max, intervallz.max)#111

#[(i+(ecke&1),j+((ecke>>1)&1),k+((ecke>>2)&1))]
#ecke=100->0,0,1->c_n4->combined_array[1]
#[0,7,3,4,1,6,2,5]

#graycode=np.array([[0,0,0,0,1,1,1,1],
#                   [0,0,1,1,1,1,0,0],
#                   [0,1,1,0,0,1,1,0]])

#combined_array=np.stack((c_n0, c_n1, c_n2, c_n3, c_n4, c_n5, c_n6, c_n7), axis=0)
combined_array=np.stack((c_n0, c_n4, c_n6, c_n2, c_n3, c_n7, c_n5, c_n1), axis=0)
combined_array=combined_array.swapaxes( 1, 2).swapaxes( 0, 1)
print(combined_array.shape)
allpoints=combined_array.reshape(-1, 3)


#u[indices]

def calcderiv(allpoints,vis):
    #calculate the derivative of point(x,y,z).inner(vis) with respect to x,y,z
    from intervallarethmetic.derivativexyz import xyzderiv
    u, rindices = np.unique(allpoints, return_inverse=True,axis=0)#reduce to unique to reduce redundant computation
    xtf,ytf,ztf=(xyzderiv(var,d)for var,d in zip(u.T,[[1,0,0],[0,1,0],[0,0,1]]))
    iprod=point(xtf,ytf,ztf).inner(vis)
    voltf=sum(abs(blade.magnitude) for blade in iprod.lst)
    delta=np.stack(voltf.df,axis=-1)#make the derivatives to a array of vectors
    return voltf.f[rindices],delta[rindices]#f,df

lot=[[], [(0, 3, 8)], [(0, 9, 1)], [(3, 8, 1), (1, 8, 9)], [(2, 11, 3)], [(8, 0, 11), (11, 0, 2)], [(3, 2, 11), (1, 0, 9)], [(11, 1, 2), (11, 9, 1), (11, 8, 9)], [(1, 10, 2)], [(0, 3, 8), (2, 1, 10)], [(10, 2, 9), (9, 2, 0)], [(8, 2, 3), (8, 10, 2), (8, 9, 10)], [(11, 3, 10), (10, 3, 1)], [(10, 0, 1), (10, 8, 0), (10, 11, 8)], [(9, 3, 0), (9, 11, 3), (9, 10, 11)], [(8, 9, 11), (11, 9, 10)], [(4, 8, 7)], [(7, 4, 3), (3, 4, 0)], [(4, 8, 7), (0, 9, 1)], [(1, 4, 9), (1, 7, 4), (1, 3, 7)], [(8, 7, 4), (11, 3, 2)], [(4, 11, 7), (4, 2, 11), (4, 0, 2)], [(0, 9, 1), (8, 7, 4), (11, 3, 2)], [(7, 4, 11), (11, 4, 2), (2, 4, 9), (2, 9, 1)], [(4, 8, 7), (2, 1, 10)], [(7, 4, 3), (3, 4, 0), (10, 2, 1)], [(10, 2, 9), (9, 2, 0), (7, 4, 8)], [(10, 2, 3), (10, 3, 4), (3, 7, 4), (9, 10, 4)], [(1, 10, 3), (3, 10, 11), (4, 8, 7)], [(10, 11, 1), (11, 7, 4), (1, 11, 4), (1, 4, 0)], [(7, 4, 8), (9, 3, 0), (9, 11, 3), (9, 10, 11)], [(7, 4, 11), (4, 9, 11), (9, 10, 11)], [(9, 4, 5)], [(9, 4, 5), (8, 0, 3)], [(4, 5, 0), (0, 5, 1)], [(5, 8, 4), (5, 3, 8), (5, 1, 3)], [(9, 4, 5), (11, 3, 2)], [(2, 11, 0), (0, 11, 8), (5, 9, 4)], [(4, 5, 0), (0, 5, 1), (11, 3, 2)], [(5, 1, 4), (1, 2, 11), (4, 1, 11), (4, 11, 8)], [(1, 10, 2), (5, 9, 4)], [(9, 4, 5), (0, 3, 8), (2, 1, 10)], [(2, 5, 10), (2, 4, 5), (2, 0, 4)], [(10, 2, 5), (5, 2, 4), (4, 2, 3), (4, 3, 8)], [(11, 3, 10), (10, 3, 1), (4, 5, 9)], [(4, 5, 9), (10, 0, 1), (10, 8, 0), (10, 11, 8)], [(11, 3, 0), (11, 0, 5), (0, 4, 5), (10, 11, 5)], [(4, 5, 8), (5, 10, 8), (10, 11, 8)], [(8, 7, 9), (9, 7, 5)], [(3, 9, 0), (3, 5, 9), (3, 7, 5)], [(7, 0, 8), (7, 1, 0), (7, 5, 1)], [(7, 5, 3), (3, 5, 1)], [(5, 9, 7), (7, 9, 8), (2, 11, 3)], [(2, 11, 7), (2, 7, 9), (7, 5, 9), (0, 2, 9)], [(2, 11, 3), (7, 0, 8), (7, 1, 0), (7, 5, 1)], [(2, 11, 1), (11, 7, 1), (7, 5, 1)], [(8, 7, 9), (9, 7, 5), (2, 1, 10)], [(10, 2, 1), (3, 9, 0), (3, 5, 9), (3, 7, 5)], [(7, 5, 8), (5, 10, 2), (8, 5, 2), (8, 2, 0)], [(10, 2, 5), (2, 3, 5), (3, 7, 5)], [(8, 7, 5), (8, 5, 9), (11, 3, 10), (3, 1, 10)], [(5, 11, 7), (10, 11, 5), (1, 9, 0)], [(11, 5, 10), (7, 5, 11), (8, 3, 0)], [(5, 11, 7), (10, 11, 5)], [(6, 7, 11)], [(7, 11, 6), (3, 8, 0)], [(6, 7, 11), (0, 9, 1)], [(9, 1, 8), (8, 1, 3), (6, 7, 11)], [(3, 2, 7), (7, 2, 6)], [(0, 7, 8), (0, 6, 7), (0, 2, 6)], [(6, 7, 2), (2, 7, 3), (9, 1, 0)], [(6, 7, 8), (6, 8, 1), (8, 9, 1), (2, 6, 1)], [(11, 6, 7), (10, 2, 1)], [(3, 8, 0), (11, 6, 7), (10, 2, 1)], [(0, 9, 2), (2, 9, 10), (7, 11, 6)], [(6, 7, 11), (8, 2, 3), (8, 10, 2), (8, 9, 10)], [(7, 10, 6), (7, 1, 10), (7, 3, 1)], [(8, 0, 7), (7, 0, 6), (6, 0, 1), (6, 1, 10)], [(7, 3, 6), (3, 0, 9), (6, 3, 9), (6, 9, 10)], [(6, 7, 10), (7, 8, 10), (8, 9, 10)], [(11, 6, 8), (8, 6, 4)], [(6, 3, 11), (6, 0, 3), (6, 4, 0)], [(11, 6, 8), (8, 6, 4), (1, 0, 9)], [(1, 3, 9), (3, 11, 6), (9, 3, 6), (9, 6, 4)], [(2, 8, 3), (2, 4, 8), (2, 6, 4)], [(4, 0, 6), (6, 0, 2)], [(9, 1, 0), (2, 8, 3), (2, 4, 8), (2, 6, 4)], [(9, 1, 4), (1, 2, 4), (2, 6, 4)], [(4, 8, 6), (6, 8, 11), (1, 10, 2)], [(1, 10, 2), (6, 3, 11), (6, 0, 3), (6, 4, 0)], [(11, 6, 4), (11, 4, 8), (10, 2, 9), (2, 0, 9)], [(10, 4, 9), (6, 4, 10), (11, 2, 3)], [(4, 8, 3), (4, 3, 10), (3, 1, 10), (6, 4, 10)], [(1, 10, 0), (10, 6, 0), (6, 4, 0)], [(4, 10, 6), (9, 10, 4), (0, 8, 3)], [(4, 10, 6), (9, 10, 4)], [(6, 7, 11), (4, 5, 9)], [(4, 5, 9), (7, 11, 6), (3, 8, 0)], [(1, 0, 5), (5, 0, 4), (11, 6, 7)], [(11, 6, 7), (5, 8, 4), (5, 3, 8), (5, 1, 3)], [(3, 2, 7), (7, 2, 6), (9, 4, 5)], [(5, 9, 4), (0, 7, 8), (0, 6, 7), (0, 2, 6)], [(3, 2, 6), (3, 6, 7), (1, 0, 5), (0, 4, 5)], [(6, 1, 2), (5, 1, 6), (4, 7, 8)], [(10, 2, 1), (6, 7, 11), (4, 5, 9)], [(0, 3, 8), (4, 5, 9), (11, 6, 7), (10, 2, 1)], [(7, 11, 6), (2, 5, 10), (2, 4, 5), (2, 0, 4)], [(8, 4, 7), (5, 10, 6), (3, 11, 2)], [(9, 4, 5), (7, 10, 6), (7, 1, 10), (7, 3, 1)], [(10, 6, 5), (7, 8, 4), (1, 9, 0)], [(4, 3, 0), (7, 3, 4), (6, 5, 10)], [(10, 6, 5), (8, 4, 7)], [(9, 6, 5), (9, 11, 6), (9, 8, 11)], [(11, 6, 3), (3, 6, 0), (0, 6, 5), (0, 5, 9)], [(11, 6, 5), (11, 5, 0), (5, 1, 0), (8, 11, 0)], [(11, 6, 3), (6, 5, 3), (5, 1, 3)], [(9, 8, 5), (8, 3, 2), (5, 8, 2), (5, 2, 6)], [(5, 9, 6), (9, 0, 6), (0, 2, 6)], [(1, 6, 5), (2, 6, 1), (3, 0, 8)], [(1, 6, 5), (2, 6, 1)], [(2, 1, 10), (9, 6, 5), (9, 11, 6), (9, 8, 11)], [(9, 0, 1), (3, 11, 2), (5, 10, 6)], [(11, 0, 8), (2, 0, 11), (10, 6, 5)], [(3, 11, 2), (5, 10, 6)], [(1, 8, 3), (9, 8, 1), (5, 10, 6)], [(6, 5, 10), (0, 1, 9)], [(8, 3, 0), (5, 10, 6)], [(6, 5, 10)], [(10, 5, 6)], [(0, 3, 8), (6, 10, 5)], [(10, 5, 6), (9, 1, 0)], [(3, 8, 1), (1, 8, 9), (6, 10, 5)], [(2, 11, 3), (6, 10, 5)], [(8, 0, 11), (11, 0, 2), (5, 6, 10)], [(1, 0, 9), (2, 11, 3), (6, 10, 5)], [(5, 6, 10), (11, 1, 2), (11, 9, 1), (11, 8, 9)], [(5, 6, 1), (1, 6, 2)], [(5, 6, 1), (1, 6, 2), (8, 0, 3)], [(6, 9, 5), (6, 0, 9), (6, 2, 0)], [(6, 2, 5), (2, 3, 8), (5, 2, 8), (5, 8, 9)], [(3, 6, 11), (3, 5, 6), (3, 1, 5)], [(8, 0, 1), (8, 1, 6), (1, 5, 6), (11, 8, 6)], [(11, 3, 6), (6, 3, 5), (5, 3, 0), (5, 0, 9)], [(5, 6, 9), (6, 11, 9), (11, 8, 9)], [(5, 6, 10), (7, 4, 8)], [(0, 3, 4), (4, 3, 7), (10, 5, 6)], [(5, 6, 10), (4, 8, 7), (0, 9, 1)], [(6, 10, 5), (1, 4, 9), (1, 7, 4), (1, 3, 7)], [(7, 4, 8), (6, 10, 5), (2, 11, 3)], [(10, 5, 6), (4, 11, 7), (4, 2, 11), (4, 0, 2)], [(4, 8, 7), (6, 10, 5), (3, 2, 11), (1, 0, 9)], [(1, 2, 10), (11, 7, 6), (9, 5, 4)], [(2, 1, 6), (6, 1, 5), (8, 7, 4)], [(0, 3, 7), (0, 7, 4), (2, 1, 6), (1, 5, 6)], [(8, 7, 4), (6, 9, 5), (6, 0, 9), (6, 2, 0)], [(7, 2, 3), (6, 2, 7), (5, 4, 9)], [(4, 8, 7), (3, 6, 11), (3, 5, 6), (3, 1, 5)], [(5, 0, 1), (4, 0, 5), (7, 6, 11)], [(9, 5, 4), (6, 11, 7), (0, 8, 3)], [(11, 7, 6), (9, 5, 4)], [(6, 10, 4), (4, 10, 9)], [(6, 10, 4), (4, 10, 9), (3, 8, 0)], [(0, 10, 1), (0, 6, 10), (0, 4, 6)], [(6, 10, 1), (6, 1, 8), (1, 3, 8), (4, 6, 8)], [(9, 4, 10), (10, 4, 6), (3, 2, 11)], [(2, 11, 8), (2, 8, 0), (6, 10, 4), (10, 9, 4)], [(11, 3, 2), (0, 10, 1), (0, 6, 10), (0, 4, 6)], [(6, 8, 4), (11, 8, 6), (2, 10, 1)], [(4, 1, 9), (4, 2, 1), (4, 6, 2)], [(3, 8, 0), (4, 1, 9), (4, 2, 1), (4, 6, 2)], [(6, 2, 4), (4, 2, 0)], [(3, 8, 2), (8, 4, 2), (4, 6, 2)], [(4, 6, 9), (6, 11, 3), (9, 6, 3), (9, 3, 1)], [(8, 6, 11), (4, 6, 8), (9, 0, 1)], [(11, 3, 6), (3, 0, 6), (0, 4, 6)], [(8, 6, 11), (4, 6, 8)], [(10, 7, 6), (10, 8, 7), (10, 9, 8)], [(3, 7, 0), (7, 6, 10), (0, 7, 10), (0, 10, 9)], [(6, 10, 7), (7, 10, 8), (8, 10, 1), (8, 1, 0)], [(6, 10, 7), (10, 1, 7), (1, 3, 7)], [(3, 2, 11), (10, 7, 6), (10, 8, 7), (10, 9, 8)], [(2, 9, 0), (10, 9, 2), (6, 11, 7)], [(0, 8, 3), (7, 6, 11), (1, 2, 10)], [(7, 6, 11), (1, 2, 10)], [(2, 1, 9), (2, 9, 7), (9, 8, 7), (6, 2, 7)], [(2, 7, 6), (3, 7, 2), (0, 1, 9)], [(8, 7, 0), (7, 6, 0), (6, 2, 0)], [(7, 2, 3), (6, 2, 7)], [(8, 1, 9), (3, 1, 8), (11, 7, 6)], [(11, 7, 6), (1, 9, 0)], [(6, 11, 7), (0, 8, 3)], [(11, 7, 6)], [(7, 11, 5), (5, 11, 10)], [(10, 5, 11), (11, 5, 7), (0, 3, 8)], [(7, 11, 5), (5, 11, 10), (0, 9, 1)], [(7, 11, 10), (7, 10, 5), (3, 8, 1), (8, 9, 1)], [(5, 2, 10), (5, 3, 2), (5, 7, 3)], [(5, 7, 10), (7, 8, 0), (10, 7, 0), (10, 0, 2)], [(0, 9, 1), (5, 2, 10), (5, 3, 2), (5, 7, 3)], [(9, 7, 8), (5, 7, 9), (10, 1, 2)], [(1, 11, 2), (1, 7, 11), (1, 5, 7)], [(8, 0, 3), (1, 11, 2), (1, 7, 11), (1, 5, 7)], [(7, 11, 2), (7, 2, 9), (2, 0, 9), (5, 7, 9)], [(7, 9, 5), (8, 9, 7), (3, 11, 2)], [(3, 1, 7), (7, 1, 5)], [(8, 0, 7), (0, 1, 7), (1, 5, 7)], [(0, 9, 3), (9, 5, 3), (5, 7, 3)], [(9, 7, 8), (5, 7, 9)], [(8, 5, 4), (8, 10, 5), (8, 11, 10)], [(0, 3, 11), (0, 11, 5), (11, 10, 5), (4, 0, 5)], [(1, 0, 9), (8, 5, 4), (8, 10, 5), (8, 11, 10)], [(10, 3, 11), (1, 3, 10), (9, 5, 4)], [(3, 2, 8), (8, 2, 4), (4, 2, 10), (4, 10, 5)], [(10, 5, 2), (5, 4, 2), (4, 0, 2)], [(5, 4, 9), (8, 3, 0), (10, 1, 2)], [(2, 10, 1), (4, 9, 5)], [(8, 11, 4), (11, 2, 1), (4, 11, 1), (4, 1, 5)], [(0, 5, 4), (1, 5, 0), (2, 3, 11)], [(0, 11, 2), (8, 11, 0), (4, 9, 5)], [(5, 4, 9), (2, 3, 11)], [(4, 8, 5), (8, 3, 5), (3, 1, 5)], [(0, 5, 4), (1, 5, 0)], [(5, 4, 9), (3, 0, 8)], [(5, 4, 9)], [(11, 4, 7), (11, 9, 4), (11, 10, 9)], [(0, 3, 8), (11, 4, 7), (11, 9, 4), (11, 10, 9)], [(11, 10, 7), (10, 1, 0), (7, 10, 0), (7, 0, 4)], [(3, 10, 1), (11, 10, 3), (7, 8, 4)], [(3, 2, 10), (3, 10, 4), (10, 9, 4), (7, 3, 4)], [(9, 2, 10), (0, 2, 9), (8, 4, 7)], [(3, 4, 7), (0, 4, 3), (1, 2, 10)], [(7, 8, 4), (10, 1, 2)], [(7, 11, 4), (4, 11, 9), (9, 11, 2), (9, 2, 1)], [(1, 9, 0), (4, 7, 8), (2, 3, 11)], [(7, 11, 4), (11, 2, 4), (2, 0, 4)], [(4, 7, 8), (2, 3, 11)], [(9, 4, 1), (4, 7, 1), (7, 3, 1)], [(7, 8, 4), (1, 9, 0)], [(3, 4, 7), (0, 4, 3)], [(7, 8, 4)], [(11, 10, 8), (8, 10, 9)], [(0, 3, 9), (3, 11, 9), (11, 10, 9)], [(1, 0, 10), (0, 8, 10), (8, 11, 10)], [(10, 3, 11), (1, 3, 10)], [(3, 2, 8), (2, 10, 8), (10, 9, 8)], [(9, 2, 10), (0, 2, 9)], [(8, 3, 0), (10, 1, 2)], [(2, 10, 1)], [(2, 1, 11), (1, 9, 11), (9, 8, 11)], [(11, 2, 3), (9, 0, 1)], [(11, 0, 8), (2, 0, 11)], [(3, 11, 2)], [(1, 8, 3), (9, 8, 1)], [(1, 9, 0)], [(8, 3, 0)], []]
verts=[[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [3, 7], [2, 6]]
f,df=calcderiv(allpoints,vis)
f=f.reshape(-1,8)
df=df.reshape(-1,8,3)
zerrovec=np.zeros(3)

vertices=[]
faces=[]
for i in range(len(combined_array)):
    voxelcornerp=combined_array[i]#cordinates of voxel corners
    voxelcornerdf=df[i]#derivatives of function on cordinates of voxel corners
    voxelcornerf=f[i]#evaluated function on cordinates of voxel corners

    parity=1
    key=1
    veclast=None#not needed but makes code clearer / better than declaring this variable in loop head
    for veclast in voxelcornerdf[::-1]:# not shure if this loop is needed veclast=zerrovec seems to work but this is saver i think
        if np.array_equal(zerrovec,veclast):
            break
    for ivec,shift in zip(range(1,8),[4,6,2,3,7,5,1]):
        #parity^=sum(d[ivec-1]*d[ivec])<0
        vecvor,vecact=voxelcornerdf[[ivec-1,ivec]]
        if not np.array_equal(zerrovec,vecvor):#if vec is zerro replace it with last workin in reverse
            veclast=vecvor#TODO find out what this iff statement does???
        else:
            vecvor=-veclast
        if np.array_equal(zerrovec,vecact):
            vecact=-veclast
        parity^=vecvor.dot(vecact)<0
        key|=parity<<shift

    for triangle in lot[key]:

        plst=[]
        for vert in triangle:

            istart=[0,7,3,4,1,6,2,5][verts[vert][0]]
            iend=[0,7,3,4,1,6,2,5][verts[vert][1]]
            #ecke1,ecke2=(punkte[(i+(ecke&1),j+((ecke>>1)&1),k+((ecke>>2)&1))] for ecke in verts[vert])
            #a,b=(vol[(i+(ecke&1),j+((ecke>>1)&1),k+((ecke>>2)&1))] for ecke in verts[vert])
            ecke1,ecke2=voxelcornerp[istart],voxelcornerp[iend]
            a,b=voxelcornerf[istart],voxelcornerf[iend]
            #lin inteerlpol von a,b
            a=abs(a)
            b=abs(b)
            lerpx=a/(a+b)
            plst.append(ecke1*(1-lerpx)+lerpx*ecke2)

            #plst.append(sum([punkte[(i+(ecke&1),j+((ecke>>1)&1),k+((ecke>>2)&1))] for ecke in verts[vert]])/2)

           
                
        #faces.extend([3,len(vertices),len(vertices)+1,len(vertices)+2])
        vertices.extend(plst)
#TODO reduce points
#TODO use strips
#TODO remap verts
#TODO remap lot
print(np.array(faces))


ids=np.arange(len(vertices)).reshape((-1, 3))
faces = np.concatenate(
    (np.full((len(vertices)//3, 1),3) , ids), axis=1
).ravel()

#print(faces)
mesh=pv.PolyData(np.array(vertices).ravel(), strips=np.array(faces).ravel())
#print(vertices)
plt.add_mesh(mesh,opacity=0.5,show_edges=0,)
plt.show()



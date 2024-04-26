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
maxvoxelnum=100000

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

    #print(x)
    #print(e1*x)
    #print(e2*y)
    #print(e1*x+e2*y)
    #print(ix*voxels.delta/2+intervallx.mid())
    #print(len(expr.lst))
    #print(np.stack([intervallx.min,intervallx.max,intervally.min,intervally.max,intervallz.min,intervallz.max]).T)
    #delete cells without 0
    #voxelsintervall=point(intervallx,intervally,intervallz).inner(vis)#calculate #TODO
    #print(evaluatefun(equations[0],cache))
    #voxelswithzerro=sum(abs(evaluatefun(e,cache)) for e in equations).containsnum()#TODO
    #voxelswithzerro=np.all([vec.containsnum() for vec in symeval(equations,intervallx,intervally,intervallz)],axis=0)#TODO
    voxelswithzerro=np.all([blade.magnitude.intervallnp().containsnum(0) for blade in expr.lst],axis=0)
    #TODO wenn ableitung sum(abs(evaluatefun(e,cache)) for e in equations).containsnum() nach x,y,z contains 0,0,0 
    #print(voxelswithzerro)
    voxels.removecells(voxelswithzerro)
    
    
    #print(voxelswithzerro)
    #print(expr.lst)
    print(len(voxelswithzerro),j)
    if len(voxelswithzerro)>maxvoxelnum:
        depth=j
        break
    voxels.subdivide()


print(time.time()-t0)







#print(time.time()-t0)
#grid = pv.UnstructuredGrid(cells_mat.ravel(), np.array([pv.CellType.VOXEL]*len(cells_mat), np.int8), all_nodes.ravel())
plt.add_mesh(voxels.gridify(),opacity=0.5)
pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(10)


plt.show()


#entweder f'==0 oder vorzeichenwechsel in den punkten
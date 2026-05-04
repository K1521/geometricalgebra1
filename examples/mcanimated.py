

import sys
import os

# Get the parent directory of the current file (my_script.py)
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
if parent_directory not in sys.path:
    sys.path.append(parent_directory)





from algebra.dcga import *
#algebra.dcga.mode="numpy"
import pyvista as pv
pv.global_theme.allow_empty_mesh = True
import time
#import functools
#import operator
import numpy as np
#import tensorflow as tf
#import matplotlib.pyplot as mplt
from intervallarethmetic.derivativexyz import xyzderiv
import myplotter
#t=toroid(1,.5)
#p=Plane(0.1,0.1,0.1,0.5)


t=toroid(1,.5)
p=Plane(0.1,0.1,0.1,0)
p=Plane(0.1,0,0.1732,0)
t=toroid(1,.5)
p=Plane(0.1,0.1,0.1,0.5)
vis=t^p#^Plane(0.1,0.1,0.001,0.5)#^Plane(0.001,0.001,0.1,0.1)
#vis=Plane(0.1,0.1,0.2,0.5)
#vis=point(0.5,0.7,0.3)
#vis=Plane(0.1,0.1,0.2,0.5)
vis=t

print(p)


f=lambda x,y,z:vis.inner(point(x,y,z))

plt=myplotter.mkplotter(dark=False)
t0=time.time() 
step=101







from intervallarethmetic.intervallareth3d1 import inter3d
from intervallarethmetic.intervallarethmetic1 import intervallareth
from intervallarethmetic.voxels import Voxels
t0=time.time()



def calcderiv(allpoints,vis):
    """calculate the derivative of vis(x,y,z) with respect to x,y,z"""

    xtf=xyzderiv.idx(allpoints[:,0])
    ytf=xyzderiv.idy(allpoints[:,1])
    ztf=xyzderiv.idz(allpoints[:,2])

    iprod=vis(xtf,ytf,ztf)#point(xtf,ytf,ztf).inner(vis)
    #print(f"{len(iprod.lst)=}")
    #voltf=sum(abs(blade.magnitude) for blade in iprod.lst)
    #voltf=sum(abs(blade.magnitude) for blade in iprod.lst)
    delta=[np.stack(blade.magnitude.df,axis=-1) for blade in iprod.lst]#make the derivatives to a array of vectors
    magnitude=[blade.magnitude.f for blade in iprod.lst]
    return magnitude,delta

def calcderivsus(allpoints,vis):
    """calculate the derivative of vis(x,y,z) with respect to x,y,z"""

    xtf=xyzderiv.idx(allpoints[:,0])
    ytf=xyzderiv.idy(allpoints[:,1])
    ztf=xyzderiv.idz(allpoints[:,2])

    iprod=vis(xtf,ytf,ztf)#point(xtf,ytf,ztf).inner(vis)
    #print(f"{len(iprod.lst)=}")
    #voltf=sum(abs(blade.magnitude) for blade in iprod.lst)
    #voltf=sum(abs(blade.magnitude) for blade in iprod.lst)
    sus=sum(x.magnitude*x.magnitude for x in iprod.lst)
    delta=np.stack(sus.df,axis=-1)
    magnitude=sus.f
    return magnitude,delta

def normalize(vecs):
    """normalize a array of vectors"""
    
    # Calculate the magnitudes of the vectors
    magnitudes = np.linalg.norm(vecs, axis=-1)
    # Avoid division by zero by setting zero magnitudes
    zerros=magnitudes == 0
    magnitudes[zerros] = 1
    # Normalize the vectors
    normalized_vecs = vecs / magnitudes[...,None]
    normalized_vecs[zerros]=0
    return normalized_vecs




def remove_mask_empty_voxels_by_alignement( rindices, magnitudes, derivs):
    #returns a mask wich is used to remove empty voxels
    #a voxel is considered empty if all of the derivatives point in the same direction

    keepidx=np.arange(len(rindices))

    for deriv,magnitude in zip(derivs,magnitudes):
        
        deriv=normalize(deriv)*np.sign(magnitude)[...,None]
        #vecs=deriv[rindices[cubeidx[idx]]]  #vecs[cube,point in cube (8),deriv (3)]
        vecs=deriv[rindices[keepidx]]

        upperleft=vecs[:,0,None,:]#normalize(vecs[:,0,None,:])
        #upperleft=normalize(np.sum(vecs,axis=1,keepdims=True))
        dotprod=np.all(np.sum(upperleft*vecs,axis=-1)>0.7,axis=-1)#for every cube all(dotproduct>0)# edit >cos(45)
        #alle >0 d.h. ungefähr eine richtung
        keepidx=keepidx[~dotprod]
        #print((~dotprod).sum(),(dotprod).sum())


        """for i, row in enumerate(np.sum(upperleft*vecs,axis=-1)[dotprod]):
            if np.any(row < 0.9):
                print(f"Row {i}: {row}")
        import matplotlib.pyplot as plt
        plt.hist(np.arccos(np.sum(upperleft*vecs,axis=-1)[dotprod].ravel())/(2*np.pi)*360, bins=1000, edgecolor='black')#,range=(0, 0.025))
        #plt.hist(np.arcsin(np.sum(upperleft*vecs,axis=-1)[dotprod].ravel())/(2*np.pi)*360, bins=100, edgecolor='black',range=(80,90))
        plt.title('Histogram of dot product values')
        plt.xlabel('Dot Product Values')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
        break"""
    return keepidx
    

def newtoniteration(startpoints,vis):
    magnitudes,derivs=calcderiv(startpoints,vis)

    J=np.stack(derivs,axis=1)#n x grade x 3
    f=np.stack(magnitudes,axis=1)#n x grade
    #print(J.shape)
    #print(f.shape)

    #goodidx=~np.any(f==0,axis=1)# these points wont work xD
    #x0 = smallpoints#[goodidx]
    f=f#[goodidx]
    J=J#[goodidx]

    #gaußnewton
    lambda_reg = 1e-5
    # Compute J^T J and J^T f(x^0) for each instance
    JTJ = np.einsum('ijk,ijl->ikl', J, J)  # Shape (8763, 3, 3)
    JTf = np.einsum('ijk,ij->ik', J, f)    # Shape (8763, 3)
    #print(JTJ)
    # Regularize JTJ to ensure it's invertible
    JTJ_reg = JTJ + lambda_reg * np.eye(JTJ.shape[1])

    # Invert JTJ for each instance
    JTJ_inv = np.linalg.inv(JTJ_reg)  # Shape (8763, 3, 3)

    # Compute the update step (J^T J)^{-1} J^T f(x^0)
    update = np.einsum('ijk,ij->ik', JTJ_inv, JTf)  # Shape (8763, 3)

    # Perform the Gauss-Newton step
    return startpoints-update

depth=12
depth2=16
maxvoxelnum=5000

voxels=Voxels(1)
#intervallx,intervally,intervallz=intervallx+17.1225253,intervally+13.127876,intervallz+32.135670
zerrovec=np.zeros(3)

lastvoxnum=1
ix=inter3d({(1,0,0):1})
iy=inter3d({(0,1,0):1})
iz=inter3d({(0,0,1):1})
for j in range(1,depth+1):
    


    x,y,z=voxels.cubemid().T
    

    #print("p")
    
    #import cProfile, pstats, io
    #from pstats import SortKey
    #pr = cProfile.Profile(builtins=False)
    #pr.enable()
    expr=f(ix*voxels.delta/2+x,
            iy*voxels.delta/2+y,
            iz*voxels.delta/2+z)
    
    #plt.add_mesh(voxels.gridify(),opacity=0.5)
    #pr.disable()
    #pstats.Stats(pr).sort_stats('tottime').print_stats(10)

    dat=[blade.magnitude.intervallnp().containsnum(0) for blade in expr.lst[:]]
    #voxelswithzerro=np.all([x for x in dat if not x is True],axis=0)
    voxelswithzerro=np.all(np.broadcast_arrays(*dat),axis=0)
    voxels.filter_cells(voxelswithzerro)
    
    
    #print(voxelswithzerro)
    #print(expr.lst)
    print(len(voxelswithzerro),j,len(voxelswithzerro)/8**j)
    if len(voxelswithzerro)>maxvoxelnum:
        depth=j
        break
    voxels.subdivide()



for j in range(depth+1,depth2+1):
    smallpoints=update=voxels.cubemid()
    for i in range(4):
        old=update
        update=newtoniteration(update,f)#2 newton iters
        #plt.add_arrows(old, update-old, mag=1)
    #plt.add_arrows(smallpoints, update-smallpoints, mag=1)
    voxels.filter_cells(np.linalg.norm(update-smallpoints,axis=1)<3**.5*voxels.delta)
    print(len(smallpoints),j,len(smallpoints)/8**j,1/voxels.delta)
    if len(smallpoints)>maxvoxelnum:
        depth=j
        break
    voxels.subdivide()
    



# cubepoints=voxels.cubecords() #remove_mask_empty_voxels_by_alignement
# allpoints=cubepoints.reshape(-1,3)
# cubeidx=voxels.cubeidx()
# smallpoints, rindices = uniquepoints(allpoints)
# magnitudes,derivs=calcderiv(smallpoints,vis)
verticesunique,rindices=voxels.cubecordsunique()
magnitudes,derivs=calcderiv(verticesunique,f)
voxels.filter_cells(remove_mask_empty_voxels_by_alignement( rindices, magnitudes, derivs))


plt.add_mesh(voxels.gridify())




# modified marching cubes
lot=[[], [(0, 3, 8)], [(0, 9, 1)], [(3, 8, 1), (1, 8, 9)], [(2, 11, 3)], [(8, 0, 11), (11, 0, 2)], [(3, 2, 11), (1, 0, 9)], [(11, 1, 2), (11, 9, 1), (11, 8, 9)], [(1, 10, 2)], [(0, 3, 8), (2, 1, 10)], [(10, 2, 9), (9, 2, 0)], [(8, 2, 3), (8, 10, 2), (8, 9, 10)], [(11, 3, 10), (10, 3, 1)], [(10, 0, 1), (10, 8, 0), (10, 11, 8)], [(9, 3, 0), (9, 11, 3), (9, 10, 11)], [(8, 9, 11), (11, 9, 10)], [(4, 8, 7)], [(7, 4, 3), (3, 4, 0)], [(4, 8, 7), (0, 9, 1)], [(1, 4, 9), (1, 7, 4), (1, 3, 7)], [(8, 7, 4), (11, 3, 2)], [(4, 11, 7), (4, 2, 11), (4, 0, 2)], [(0, 9, 1), (8, 7, 4), (11, 3, 2)], [(7, 4, 11), (11, 4, 2), (2, 4, 9), (2, 9, 1)], [(4, 8, 7), (2, 1, 10)], [(7, 4, 3), (3, 4, 0), (10, 2, 1)], [(10, 2, 9), (9, 2, 0), (7, 4, 8)], [(10, 2, 3), (10, 3, 4), (3, 7, 4), (9, 10, 4)], [(1, 10, 3), (3, 10, 11), (4, 8, 7)], [(10, 11, 1), (11, 7, 4), (1, 11, 4), (1, 4, 0)], [(7, 4, 8), (9, 3, 0), (9, 11, 3), (9, 10, 11)], [(7, 4, 11), (4, 9, 11), (9, 10, 11)], [(9, 4, 5)], [(9, 4, 5), (8, 0, 3)], [(4, 5, 0), (0, 5, 1)], [(5, 8, 4), (5, 3, 8), (5, 1, 3)], [(9, 4, 5), (11, 3, 2)], [(2, 11, 0), (0, 11, 8), (5, 9, 4)], [(4, 5, 0), (0, 5, 1), (11, 3, 2)], [(5, 1, 4), (1, 2, 11), (4, 1, 11), (4, 11, 8)], [(1, 10, 2), (5, 9, 4)], [(9, 4, 5), (0, 3, 8), (2, 1, 10)], [(2, 5, 10), (2, 4, 5), (2, 0, 4)], [(10, 2, 5), (5, 2, 4), (4, 2, 3), (4, 3, 8)], [(11, 3, 10), (10, 3, 1), (4, 5, 9)], [(4, 5, 9), (10, 0, 1), (10, 8, 0), (10, 11, 8)], [(11, 3, 0), (11, 0, 5), (0, 4, 5), (10, 11, 5)], [(4, 5, 8), (5, 10, 8), (10, 11, 8)], [(8, 7, 9), (9, 7, 5)], [(3, 9, 0), (3, 5, 9), (3, 7, 5)], [(7, 0, 8), (7, 1, 0), (7, 5, 1)], [(7, 5, 3), (3, 5, 1)], [(5, 9, 7), (7, 9, 8), (2, 11, 3)], [(2, 11, 7), (2, 7, 9), (7, 5, 9), (0, 2, 9)], [(2, 11, 3), (7, 0, 8), (7, 1, 0), (7, 5, 1)], [(2, 11, 1), (11, 7, 1), (7, 5, 1)], [(8, 7, 9), (9, 7, 5), (2, 1, 10)], [(10, 2, 1), (3, 9, 0), (3, 5, 9), (3, 7, 5)], [(7, 5, 8), (5, 10, 2), (8, 5, 2), (8, 2, 0)], [(10, 2, 5), (2, 3, 5), (3, 7, 5)], [(8, 7, 5), (8, 5, 9), (11, 3, 10), (3, 1, 10)], [(5, 11, 7), (10, 11, 5), (1, 9, 0)], [(11, 5, 10), (7, 5, 11), (8, 3, 0)], [(5, 11, 7), (10, 11, 5)], [(6, 7, 11)], [(7, 11, 6), (3, 8, 0)], [(6, 7, 11), (0, 9, 1)], [(9, 1, 8), (8, 1, 3), (6, 7, 11)], [(3, 2, 7), (7, 2, 6)], [(0, 7, 8), (0, 6, 7), (0, 2, 6)], [(6, 7, 2), (2, 7, 3), (9, 1, 0)], [(6, 7, 8), (6, 8, 1), (8, 9, 1), (2, 6, 1)], [(11, 6, 7), (10, 2, 1)], [(3, 8, 0), (11, 6, 7), (10, 2, 1)], [(0, 9, 2), (2, 9, 10), (7, 11, 6)], [(6, 7, 11), (8, 2, 3), (8, 10, 2), (8, 9, 10)], [(7, 10, 6), (7, 1, 10), (7, 3, 1)], [(8, 0, 7), (7, 0, 6), (6, 0, 1), (6, 1, 10)], [(7, 3, 6), (3, 0, 9), (6, 3, 9), (6, 9, 10)], [(6, 7, 10), (7, 8, 10), (8, 9, 10)], [(11, 6, 8), (8, 6, 4)], [(6, 3, 11), (6, 0, 3), (6, 4, 0)], [(11, 6, 8), (8, 6, 4), (1, 0, 9)], [(1, 3, 9), (3, 11, 6), (9, 3, 6), (9, 6, 4)], [(2, 8, 3), (2, 4, 8), (2, 6, 4)], [(4, 0, 6), (6, 0, 2)], [(9, 1, 0), (2, 8, 3), (2, 4, 8), (2, 6, 4)], [(9, 1, 4), (1, 2, 4), (2, 6, 4)], [(4, 8, 6), (6, 8, 11), (1, 10, 2)], [(1, 10, 2), (6, 3, 11), (6, 0, 3), (6, 4, 0)], [(11, 6, 4), (11, 4, 8), (10, 2, 9), (2, 0, 9)], [(10, 4, 9), (6, 4, 10), (11, 2, 3)], [(4, 8, 3), (4, 3, 10), (3, 1, 10), (6, 4, 10)], [(1, 10, 0), (10, 6, 0), (6, 4, 0)], [(4, 10, 6), (9, 10, 4), (0, 8, 3)], [(4, 10, 6), (9, 10, 4)], [(6, 7, 11), (4, 5, 9)], [(4, 5, 9), (7, 11, 6), (3, 8, 0)], [(1, 0, 5), (5, 0, 4), (11, 6, 7)], [(11, 6, 7), (5, 8, 4), (5, 3, 8), (5, 1, 3)], [(3, 2, 7), (7, 2, 6), (9, 4, 5)], [(5, 9, 4), (0, 7, 8), (0, 6, 7), (0, 2, 6)], [(3, 2, 6), (3, 6, 7), (1, 0, 5), (0, 4, 5)], [(6, 1, 2), (5, 1, 6), (4, 7, 8)], [(10, 2, 1), (6, 7, 11), (4, 5, 9)], [(0, 3, 8), (4, 5, 9), (11, 6, 7), (10, 2, 1)], [(7, 11, 6), (2, 5, 10), (2, 4, 5), (2, 0, 4)], [(8, 4, 7), (5, 10, 6), (3, 11, 2)], [(9, 4, 5), (7, 10, 6), (7, 1, 10), (7, 3, 1)], [(10, 6, 5), (7, 8, 4), (1, 9, 0)], [(4, 3, 0), (7, 3, 4), (6, 5, 10)], [(10, 6, 5), (8, 4, 7)], [(9, 6, 5), (9, 11, 6), (9, 8, 11)], [(11, 6, 3), (3, 6, 0), (0, 6, 5), (0, 5, 9)], [(11, 6, 5), (11, 5, 0), (5, 1, 0), (8, 11, 0)], [(11, 6, 3), (6, 5, 3), (5, 1, 3)], [(9, 8, 5), (8, 3, 2), (5, 8, 2), (5, 2, 6)], [(5, 9, 6), (9, 0, 6), (0, 2, 6)], [(1, 6, 5), (2, 6, 1), (3, 0, 8)], [(1, 6, 5), (2, 6, 1)], [(2, 1, 10), (9, 6, 5), (9, 11, 6), (9, 8, 11)], [(9, 0, 1), (3, 11, 2), (5, 10, 6)], [(11, 0, 8), (2, 0, 11), (10, 6, 5)], [(3, 11, 2), (5, 10, 6)], [(1, 8, 3), (9, 8, 1), (5, 10, 6)], [(6, 5, 10), (0, 1, 9)], [(8, 3, 0), (5, 10, 6)], [(6, 5, 10)], [(10, 5, 6)], [(0, 3, 8), (6, 10, 5)], [(10, 5, 6), (9, 1, 0)], [(3, 8, 1), (1, 8, 9), (6, 10, 5)], [(2, 11, 3), (6, 10, 5)], [(8, 0, 11), (11, 0, 2), (5, 6, 10)], [(1, 0, 9), (2, 11, 3), (6, 10, 5)], [(5, 6, 10), (11, 1, 2), (11, 9, 1), (11, 8, 9)], [(5, 6, 1), (1, 6, 2)], [(5, 6, 1), (1, 6, 2), (8, 0, 3)], [(6, 9, 5), (6, 0, 9), (6, 2, 0)], [(6, 2, 5), (2, 3, 8), (5, 2, 8), (5, 8, 9)], [(3, 6, 11), (3, 5, 6), (3, 1, 5)], [(8, 0, 1), (8, 1, 6), (1, 5, 6), (11, 8, 6)], [(11, 3, 6), (6, 3, 5), (5, 3, 0), (5, 0, 9)], [(5, 6, 9), (6, 11, 9), (11, 8, 9)], [(5, 6, 10), (7, 4, 8)], [(0, 3, 4), (4, 3, 7), (10, 5, 6)], [(5, 6, 10), (4, 8, 7), (0, 9, 1)], [(6, 10, 5), (1, 4, 9), (1, 7, 4), (1, 3, 7)], [(7, 4, 8), (6, 10, 5), (2, 11, 3)], [(10, 5, 6), (4, 11, 7), (4, 2, 11), (4, 0, 2)], [(4, 8, 7), (6, 10, 5), (3, 2, 11), (1, 0, 9)], [(1, 2, 10), (11, 7, 6), (9, 5, 4)], [(2, 1, 6), (6, 1, 5), (8, 7, 4)], [(0, 3, 7), (0, 7, 4), (2, 1, 6), (1, 5, 6)], [(8, 7, 4), (6, 9, 5), (6, 0, 9), (6, 2, 0)], [(7, 2, 3), (6, 2, 7), (5, 4, 9)], [(4, 8, 7), (3, 6, 11), (3, 5, 6), (3, 1, 5)], [(5, 0, 1), (4, 0, 5), (7, 6, 11)], [(9, 5, 4), (6, 11, 7), (0, 8, 3)], [(11, 7, 6), (9, 5, 4)], [(6, 10, 4), (4, 10, 9)], [(6, 10, 4), (4, 10, 9), (3, 8, 0)], [(0, 10, 1), (0, 6, 10), (0, 4, 6)], [(6, 10, 1), (6, 1, 8), (1, 3, 8), (4, 6, 8)], [(9, 4, 10), (10, 4, 6), (3, 2, 11)], [(2, 11, 8), (2, 8, 0), (6, 10, 4), (10, 9, 4)], [(11, 3, 2), (0, 10, 1), (0, 6, 10), (0, 4, 6)], [(6, 8, 4), (11, 8, 6), (2, 10, 1)], [(4, 1, 9), (4, 2, 1), (4, 6, 2)], [(3, 8, 0), (4, 1, 9), (4, 2, 1), (4, 6, 2)], [(6, 2, 4), (4, 2, 0)], [(3, 8, 2), (8, 4, 2), (4, 6, 2)], [(4, 6, 9), (6, 11, 3), (9, 6, 3), (9, 3, 1)], [(8, 6, 11), (4, 6, 8), (9, 0, 1)], [(11, 3, 6), (3, 0, 6), (0, 4, 6)], [(8, 6, 11), (4, 6, 8)], [(10, 7, 6), (10, 8, 7), (10, 9, 8)], [(3, 7, 0), (7, 6, 10), (0, 7, 10), (0, 10, 9)], [(6, 10, 7), (7, 10, 8), (8, 10, 1), (8, 1, 0)], [(6, 10, 7), (10, 1, 7), (1, 3, 7)], [(3, 2, 11), (10, 7, 6), (10, 8, 7), (10, 9, 8)], [(2, 9, 0), (10, 9, 2), (6, 11, 7)], [(0, 8, 3), (7, 6, 11), (1, 2, 10)], [(7, 6, 11), (1, 2, 10)], [(2, 1, 9), (2, 9, 7), (9, 8, 7), (6, 2, 7)], [(2, 7, 6), (3, 7, 2), (0, 1, 9)], [(8, 7, 0), (7, 6, 0), (6, 2, 0)], [(7, 2, 3), (6, 2, 7)], [(8, 1, 9), (3, 1, 8), (11, 7, 6)], [(11, 7, 6), (1, 9, 0)], [(6, 11, 7), (0, 8, 3)], [(11, 7, 6)], [(7, 11, 5), (5, 11, 10)], [(10, 5, 11), (11, 5, 7), (0, 3, 8)], [(7, 11, 5), (5, 11, 10), (0, 9, 1)], [(7, 11, 10), (7, 10, 5), (3, 8, 1), (8, 9, 1)], [(5, 2, 10), (5, 3, 2), (5, 7, 3)], [(5, 7, 10), (7, 8, 0), (10, 7, 0), (10, 0, 2)], [(0, 9, 1), (5, 2, 10), (5, 3, 2), (5, 7, 3)], [(9, 7, 8), (5, 7, 9), (10, 1, 2)], [(1, 11, 2), (1, 7, 11), (1, 5, 7)], [(8, 0, 3), (1, 11, 2), (1, 7, 11), (1, 5, 7)], [(7, 11, 2), (7, 2, 9), (2, 0, 9), (5, 7, 9)], [(7, 9, 5), (8, 9, 7), (3, 11, 2)], [(3, 1, 7), (7, 1, 5)], [(8, 0, 7), (0, 1, 7), (1, 5, 7)], [(0, 9, 3), (9, 5, 3), (5, 7, 3)], [(9, 7, 8), (5, 7, 9)], [(8, 5, 4), (8, 10, 5), (8, 11, 10)], [(0, 3, 11), (0, 11, 5), (11, 10, 5), (4, 0, 5)], [(1, 0, 9), (8, 5, 4), (8, 10, 5), (8, 11, 10)], [(10, 3, 11), (1, 3, 10), (9, 5, 4)], [(3, 2, 8), (8, 2, 4), (4, 2, 10), (4, 10, 5)], [(10, 5, 2), (5, 4, 2), (4, 0, 2)], [(5, 4, 9), (8, 3, 0), (10, 1, 2)], [(2, 10, 1), (4, 9, 5)], [(8, 11, 4), (11, 2, 1), (4, 11, 1), (4, 1, 5)], [(0, 5, 4), (1, 5, 0), (2, 3, 11)], [(0, 11, 2), (8, 11, 0), (4, 9, 5)], [(5, 4, 9), (2, 3, 11)], [(4, 8, 5), (8, 3, 5), (3, 1, 5)], [(0, 5, 4), (1, 5, 0)], [(5, 4, 9), (3, 0, 8)], [(5, 4, 9)], [(11, 4, 7), (11, 9, 4), (11, 10, 9)], [(0, 3, 8), (11, 4, 7), (11, 9, 4), (11, 10, 9)], [(11, 10, 7), (10, 1, 0), (7, 10, 0), (7, 0, 4)], [(3, 10, 1), (11, 10, 3), (7, 8, 4)], [(3, 2, 10), (3, 10, 4), (10, 9, 4), (7, 3, 4)], [(9, 2, 10), (0, 2, 9), (8, 4, 7)], [(3, 4, 7), (0, 4, 3), (1, 2, 10)], [(7, 8, 4), (10, 1, 2)], [(7, 11, 4), (4, 11, 9), (9, 11, 2), (9, 2, 1)], [(1, 9, 0), (4, 7, 8), (2, 3, 11)], [(7, 11, 4), (11, 2, 4), (2, 0, 4)], [(4, 7, 8), (2, 3, 11)], [(9, 4, 1), (4, 7, 1), (7, 3, 1)], [(7, 8, 4), (1, 9, 0)], [(3, 4, 7), (0, 4, 3)], [(7, 8, 4)], [(11, 10, 8), (8, 10, 9)], [(0, 3, 9), (3, 11, 9), (11, 10, 9)], [(1, 0, 10), (0, 8, 10), (8, 11, 10)], [(10, 3, 11), (1, 3, 10)], [(3, 2, 8), (2, 10, 8), (10, 9, 8)], [(9, 2, 10), (0, 2, 9)], [(8, 3, 0), (10, 1, 2)], [(2, 10, 1)], [(2, 1, 11), (1, 9, 11), (9, 8, 11)], [(11, 2, 3), (9, 0, 1)], [(11, 0, 8), (2, 0, 11)], [(3, 11, 2)], [(1, 8, 3), (9, 8, 1)], [(1, 9, 0)], [(8, 3, 0)], []]
verts=[[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [3, 7], [2, 6]]

graycode=np.array([[0,0,0,0,1,1,1,1],[0,0,1,1,1,1,0,0],[0,1,1,0,0,1,1,0]])
zerrovec=np.zeros(3)

corners,voxelidx=voxels.cubecordsunique()
magnitude,delta=calcderivsus(corners,f)

print(Voxels.subvox)
print([next(i for i,x in enumerate(Voxels.subvox) if np.all(x==row))for row in graycode.T])

vertices=[]
faces=[]
for cubeidxs in voxelidx:
    cubeidxsgrey=cubeidxs[[0, 4, 6, 2, 3, 7, 5, 1]]

    #points=vol[i+graycode[0],j+graycode[1],k+graycode[2]]#points=[vol[i,j,k],vol[i,j,k+1],vol[i,j+1,k+1],vol[i,j+1,k],vol[i+1,j+1,k],vol[i+1,j+1,k+1],vol[i+1,j,k+1],vol[i+1,j,k]]
    #d=delta[i+graycode[0],j+graycode[1],k+graycode[2],:]
    d=delta[cubeidxsgrey]
    #print(d)
    parity=1
    #dvecvor=d[-1]
    key=1
    #[0,4,6,2,3,7,5,1]=[x|y<<1|z<<2 for x,y,z in zip(*graycode)]
    #veclast=zerrovec
    #veclast=d[-1]
    veclast=None#not needed but makes code clearer / better than declaring this variable in loop head
    for veclast in d[::-1]:# not shure if this loop is needed veclast=zerrovec seems to work but this is saver i think
        if np.array_equal(zerrovec,veclast):
            break
    for ivec,shift in zip(range(1,len(d)),[4,6,2,3,7,5,1]):
        #parity^=sum(d[ivec-1]*d[ivec])<0
        vecvor,vecact=d[[ivec-1,ivec]]
        if not np.array_equal(zerrovec,vecvor):#if vec is zerro replace it with last workin in reverse
            veclast=vecvor
        else:
            vecvor=-veclast
        if np.array_equal(zerrovec,vecact):
            vecact=-veclast
        parity^=vecvor.dot(vecact)<0
        #if not (np.array_equal(zerrovec,vecvor) and np.array_equal(zerrovec,vecact)):
            #   parity^=vecvor.dot(vecact)<=0
        #else:
        #    print("zerro")
        #if np.array_equal(zerrovec,vecvor):
        #    print("errov")
        key|=parity<<shift
    #print(key)
    for triangle in lot[key]:

        plst=[]
        for vert in triangle:
            #ecke1,ecke2=(punkte[(i+(ecke&1),j+((ecke>>1)&1),k+((ecke>>2)&1))] for ecke in verts[vert])
            #a,b=(vol[(i+(ecke&1),j+((ecke>>1)&1),k+((ecke>>2)&1))] for ecke in verts[vert])

            ecke1,ecke2=(corners[cubeidxs[ecke]] for ecke in verts[vert])
            a,b=(magnitude[cubeidxs[ecke]] for ecke in verts[vert])
            #lin inteerlpol von a,b 
            a=abs(a)
            b=abs(b)
            lerpx=a/(a+b)
            plst.append(ecke1*(1-lerpx)+lerpx*ecke2)

            #plst.append(sum([punkte[(i+(ecke&1),j+((ecke>>1)&1),k+((ecke>>2)&1))] for ecke in verts[vert]])/2)

            #if plst[-1][2]>2.9:
            #    print(points,i,j,k)
                
        faces.extend([3,len(vertices),len(vertices)+1,len(vertices)+2])
        vertices.extend(plst)

print(np.array(vertices).ravel().reshape(-1,3))
mesh=pv.PolyData(np.array(vertices).ravel().reshape(-1,3), strips=np.array(faces).ravel())
#print(vertices)
plt.add_mesh(mesh,opacity=0.5,show_edges=0,)


# startpoints=update=voxels.cubemid()
# for i in range(8):
#     update=newtoniteration(update,f)#2 newton iters

# #plt.add_arrows(startpoints, update-startpoints, mag=1)
# #plt.add_mesh(voxels.gridify(),opacity=0.5,show_edges=1,)




# points=update
# from curveviz.pointstocurve import pointstocurve
# g=pointstocurve(points,bucketsize=voxels.delta)



# polydata = pv.PolyData(points)#,lines=np.array([(2,u,v) for u,v,d in g.getedges(True)]))
# plt.add_mesh(polydata)#, line_width=5)





# from collections import Counter
# print(Counter(map(len,g.adj_list.values())))
# print([(n,len(x)) for n,x in g.adj_list.items() if len(x)!=2])


# voxels=Voxels(1)#
# voxels.subdivide()
# voxels.subdivide()
# startpoints=update=voxels.cubemid()
# #for i in range(8):
# #    old=update
# #    update=newtoniteration(update,f)#2 newton iters
# #    plt.add_arrows(old, update-old, mag=1,show_scalar_bar=False)

# old=startpoints
# magnitudes,derivs=calcderivsus(startpoints,f)
# print(magnitudes.shape,derivs.shape,np.linalg.norm(derivs,axis=-1).shape)
# update=old-(magnitudes/(np.linalg.norm(derivs,axis=-1)**2))[...,None]*derivs
# plt.add_arrows(old, update-old, mag=1,show_scalar_bar=False)
# voxels.filter_cells(np.linalg.norm(update-startpoints,axis=1)<3**.5*voxels.delta*0.25)

# plt.add_mesh(voxels.gridify(),opacity=0.5,show_edges=1,)


# highlighted_points = points[[n for n,x in g.adj_list.items() if len(x)==3]]
# if len(highlighted_points):
#     highlighted_point_cloud = pv.PolyData(highlighted_points)
#     plt.add_mesh(highlighted_point_cloud, color='red', point_size=5, render_points_as_spheres=True)
# highlighted_points = points[[n for n,x in g.adj_list.items() if len(x)==1]]
# if len(highlighted_points):
#     highlighted_point_cloud = pv.PolyData(highlighted_points)
#     plt.add_mesh(highlighted_point_cloud, color='orange', point_size=5, render_points_as_spheres=True)

#plt.camera_position = 'yz'
plt.show()#interactive_update =True)
#plt.camera.azimuth = 45
# from pathlib import Path
# #print(Path(".").absolute())
# plt.open_gif("curveorbit.gif")
# path = plt.generate_orbital_path(n_points=360, shift=2)
# plt.orbit_on_path(path,step=0.05,progress_bar=True,write_frames=True)
# plt.close()
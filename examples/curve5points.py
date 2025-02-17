

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
#vis=t
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

depth=16
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



startpoints=update=voxels.cubemid()
for i in range(8):
    update=newtoniteration(update,f)#2 newton iters

#plt.add_arrows(startpoints, update-startpoints, mag=1)
#plt.add_mesh(voxels.gridify(),opacity=0.5,show_edges=1,)




points=update
from curveviz.pointstocurve import pointstocurve
g=pointstocurve(points,bucketsize=voxels.delta)



polydata = pv.PolyData(points)#,lines=np.array([(2,u,v) for u,v,d in g.getedges(True)]))
plt.add_mesh(polydata)#, line_width=5)



from collections import Counter
print(Counter(map(len,g.adj_list.values())))
print([(n,len(x)) for n,x in g.adj_list.items() if len(x)!=2])


# voxels=Voxels(1)#
# voxels.subdivide()
# startpoints=update=voxels.cubemid()
# for i in range(8):
#     old=update
#     update=newtoniteration(update,vis)#2 newton iters
#     plt.add_arrows(old, update-old, mag=1)
# voxels.removecells(np.linalg.norm(update-startpoints,axis=1)<3**.5*voxels.delta)
# plt.add_mesh(voxels.gridify(),opacity=0.5,show_edges=1,)


# highlighted_points = points[[n for n,x in g.adj_list.items() if len(x)==3]]
# if len(highlighted_points):
#     highlighted_point_cloud = pv.PolyData(highlighted_points)
#     plt.add_mesh(highlighted_point_cloud, color='red', point_size=5, render_points_as_spheres=True)
# highlighted_points = points[[n for n,x in g.adj_list.items() if len(x)==1]]
# if len(highlighted_points):
#     highlighted_point_cloud = pv.PolyData(highlighted_points)
#     plt.add_mesh(highlighted_point_cloud, color='orange', point_size=5, render_points_as_spheres=True)
plt.show()


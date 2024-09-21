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
vis=Plane(0.1,0.1,0.2,0.5)
#vis=point(0.5,0.7,0.3)
#vis=Plane(0.1,0.1,0.2,0.5)
vis=t
print(p)


f=lambda x,y,z:point(x,y,z)

plt=myplotter.mkplotter()
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
maxvoxelnum=5000

actors=[]

voxels=Voxels(16)


for j in range(1,depth+1):
    


    x,y,z=voxels.cubemid().T
    

    #print("p")
    
    #import cProfile, pstats, io
    #from pstats import SortKey
    #pr = cProfile.Profile(builtins=False)
    #pr.enable()
    expr=f(inter3d.ix*voxels.delta/2+x,
           inter3d.iy*voxels.delta/2+y,
           inter3d.iz*voxels.delta/2+z)
    
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
actors.append((plt.add_mesh(voxels.gridify(),opacity=0.3,color="red"),"red"))

voxels=Voxels(16)
for j in range(1,depth+1):
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


actors.append((plt.add_mesh(voxels.gridify(),opacity=0.3,color="green"),"green"))

from intervallarethmetic.intervallareth3d2 import poly3d
voxels=Voxels(16)
zerrovec=np.zeros(3)
expr=f(poly3d.ix,poly3d.iy,poly3d.iz)
for j in range(1,depth+1):
    


    x,y,z=voxels.cubemid().T
    

    #print("p")
    
    #import cProfile, pstats, io
    #from pstats import SortKey
    #pr = cProfile.Profile(builtins=False)
    #pr.enable()
    
    
    #plt.add_mesh(voxels.gridify(),opacity=0.5)
    #pr.disable()
    #pstats.Stats(pr).sort_stats('tottime').print_stats(10)

    dat=[blade.magnitude.intervallnp(x,y,z,voxels.delta/2).containsnum(0) for blade in expr.lst[:]]
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
#plt.add_mesh(voxels.gridify(),opacity=0.3,color="blue")
actors.append((plt.add_mesh(voxels.gridify(),opacity=0.3,color="blue"),"blue"))






for i,(actor,color) in enumerate(actors):
    #print((10+i*60,10+i*60))
    #actor=actor
    def fun(x, actor=actor):
        #nonlocal actor
        actor.visibility=x

    plt.add_checkbox_button_widget(
        fun,
        value=True,
        color_on=color,
        color_off='grey',
        background_color='white',
        position=(10+i*60,10)
    )

plt.show()

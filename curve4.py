from algebra.dcga import *
#algebra.dcga.mode="numpy"
import pyvista as pv
import time
#import functools
#import operator
import numpy as np
#import tensorflow as tf
#import matplotlib.pyplot as mplt
from intervallarethmetic.derivativexyz import xyzderiv
import myplotter
t=toroid(1,.5)
p=Plane(0.1,0.1,0.1,0.5)


t=toroid(1,1)
p=Plane(0.1,0.1,0.1,0)
vis=t^p#^Plane(0.1,0.1,0.001,0.5)#^Plane(0.001,0.001,0.1,0.1)
#vis=Plane(0.1,0.1,0.2,0.5)
#vis=point(0.5,0.7,0.3)
#vis=Plane(0.1,0.1,0.2,0.5)
#vis=t
print(p)


plt=myplotter.mkplotter()
t0=time.time() 
step=101







from intervallarethmetic.intervallareth3d1 import inter3d
from intervallarethmetic.intervallarethmetic1 import intervallareth
from intervallarethmetic.voxels import Voxels
t0=time.time()




depth=16
maxvoxelnum=5000

voxels=Voxels(64)
#intervallx,intervally,intervallz=intervallx+17.1225253,intervally+13.127876,intervallz+32.135670
zerrovec=np.zeros(3)

lastvoxnum=1
ix=inter3d({(1,0,0):1})
iy=inter3d({(0,1,0):1})
iz=inter3d({(0,0,1):1})
for j in range(1,depth+1):


    x,y,z=voxels.cubemid().T
    
    p=point(ix*voxels.delta/2+x,
            iy*voxels.delta/2+y,
            iz*voxels.delta/2+z)
    #print("p")
    
    #import cProfile, pstats, io
    #from pstats import SortKey
    #pr = cProfile.Profile(builtins=False)
    #pr.enable()
    expr=p.inner(vis)
    
    #plt.add_mesh(voxels.gridify(),opacity=0.5)
    #pr.disable()
    #pstats.Stats(pr).sort_stats('tottime').print_stats(10)

   
    voxelswithzerro=np.all([blade.magnitude.intervallnp().containsnum(0) for blade in expr.lst[:]],axis=0)
    voxels.filter_cells(voxelswithzerro)
    
    
    #print(voxelswithzerro)
    #print(expr.lst)
    print(len(voxelswithzerro),j,len(voxelswithzerro)/8**j)
    if len(voxelswithzerro)>maxvoxelnum:
        depth=j
        break
    voxels.subdivide()




def uniquepoints(allpoints):
    #reduce to unique to reduce redundant computation
    uniquepoints, rindices = np.unique(allpoints, return_inverse=True,axis=0)
    return uniquepoints, rindices

def calcderiv(allpoints,vis):
    #calculate the derivative of point(x,y,z).inner(vis) with respect to x,y,z
    xtf,ytf,ztf=(xyzderiv(var,d)for var,d in zip(allpoints.T,[[1,0,0],[0,1,0],[0,0,1]]))
    iprod=point(xtf,ytf,ztf).inner(vis)
    print(f"{len(iprod.lst)=}")
    #voltf=sum(abs(blade.magnitude) for blade in iprod.lst)
    #voltf=sum(abs(blade.magnitude) for blade in iprod.lst)
    delta=[np.stack(blade.magnitude.df,axis=-1) for blade in iprod.lst]#make the derivatives to a array of vectors
    magnitude=[blade.magnitude.f for blade in iprod.lst]
    return magnitude,delta

def normalize(vecs):
    
    # Calculate the magnitudes of the vectors
    magnitudes = np.linalg.norm(vecs, axis=-1)
    # Avoid division by zero by setting zero magnitudes
    zerros=magnitudes == 0
    magnitudes[zerros] = 1
    # Normalize the vectors
    normalized_vecs = vecs / magnitudes[...,None]
    normalized_vecs[zerros]=0
    return normalized_vecs




def remove_mask_empty_voxels_by_alignement(cubeidx, rindices, magnitudes, derivs):
    #returns a mask wich is used to remove empty voxels
    #a voxel is considered empty if all of the derivatives point in the same direction

    idx=np.arange(len(cubeidx))
    for deriv,magnitude in zip(derivs,magnitudes):
        print( "deriv,magnitude" )
        deriv=normalize(deriv)*np.sign(magnitude)[...,None]
        vecs=deriv[rindices[cubeidx[idx]]]  #vecs[cube,point in cube (8),deriv (3)]
        upperleft=normalize(vecs[:,0,None,:])
        #upperleft=normalize(np.sum(vecs,axis=1,keepdims=True))
        dotprod=np.all(np.sum(upperleft*vecs,axis=-1)>0.7,axis=-1)#for every cube all(dotproduct>0)# edit >cos(45)
        #alle >0 d.h. ungefähr eine richtung
        idx=idx[~dotprod]
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
    return idx
    

    



cubepoints=voxels.cubecords()
allpoints=cubepoints.reshape(-1,3)
cubeidx=voxels.cubeidx()
#print(np.allclose(allpoints[cubeidx],cubepoints))
smallpoints, rindices = uniquepoints(allpoints)
#print(np.allclose(allpoints[cubeidx],cubepoints))
#print(np.allclose(smallpoints[rindices[cubeidx]],cubepoints))

magnitudes,derivs=calcderiv(smallpoints,vis)




#idx=np.full(len(cubeidx),True)
#idx[remove_mask_empty_voxels_by_alignement(cubeidx, rindices, magnitudes, derivs)]=False
#voxels.removecells(idx)

def invertmask(index,size):
    mask=np.full(size,True)
    mask[index]=False
    return mask



#voxels.removecells(invertmask(remove_mask_empty_voxels_by_alignement(cubeidx, rindices, magnitudes, derivs),len(cubeidx)))
voxels.filter_cells(remove_mask_empty_voxels_by_alignement(cubeidx, rindices, magnitudes, derivs))



#cubepoints=voxels.cubecords()
#allpoints=cubepoints.reshape(-1,3)
#cubeidx=voxels.cubeidx()
#smallpoints, rindices = uniquepoints(allpoints)
smallpoints=voxels.cubemid()
def newtoniteration(smallpoints,vis):
    magnitudes,derivs=calcderiv(smallpoints,vis)
    #for deriv,magnitude in zip(derivs,magnitudes):#J Nx3
    J=np.stack(derivs,axis=1)
    f=np.stack(magnitudes,axis=1)
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
    return smallpoints-update

update=smallpoints
for i in range(2):
    update=newtoniteration(update,vis)#2 newton iters
plt.add_arrows(smallpoints, update-smallpoints, mag=1)


#smallpoints, rindices = uniquepoints(voxels.cubecords().reshape(-1,3))
#magnitudes,derivs=calcderiv(smallpoints,vis)
#plt.add_arrows(smallpoints, -normalize(derivs[0])*np.sign(magnitudes[0])[...,None]*voxels.delta, mag=1)

plt.add_mesh(voxels.gridify(),opacity=0.5,show_edges=1,)


points=update

import heapq

class BucketGrid:
    def __init__(self, points, bucketsize):
        """
        Initialize the BucketGrid with a set of 3D points and voxel parameters.

        :param points: nx3 array representing the points in 3D space
        :bucketsize: attribute defining the bucket size
        """
        self.points = points
        self.bucketsize = bucketsize
        self.buckets = self._create_buckets()

    def _get_bucket(self, vector):
        """
        Get the grid bucket for a given vector.

        :param vector: 3D vector
        :return: Tuple representing the bucket the vector belongs to
        """
        return np.floor(vector / self.bucketsize).astype(int)

    def _create_buckets(self):
        """
        Create the bucket dictionary that maps voxel grids to point indices.

        :return: Dictionary where keys are bucket coordinates and values are lists of point indices
        """
        buckets = {}
        for i, b in enumerate(self._get_bucket(self.points)):
            buckets.setdefault(tuple(b), []).append(i)
        return buckets

    def nearby_points(self, p):
        """
        Find nearby points to a given point p within adjacent voxel grid cells.

        :param p: 3D point to search around
        :return: List of tuples (distance, index) sorted by distance
        """
        a, b, c = self._get_bucket(p)
        nearby = []
        
        # Check all neighboring buckets
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    key = (a + i, b + j, c + k)
                    idx = self.buckets.get(key, [])
                    nearby.extend(idx)

        nearby = np.array(nearby)
        
        # Compute distances and sort
        distances = np.linalg.norm(self.points[nearby] - p, axis=1)
        sorted_indices = np.argsort(distances)
        
        return zip(distances[sorted_indices], nearby[sorted_indices])

buckets=BucketGrid(points,voxels.delta)


class Graph:
    def __init__(self):
        self.adj_list = {}

    def add_edge(self, u, v, d):
        # Using a dictionary to avoid double edges and store weights directly
        if u not in self.adj_list:
            self.adj_list[u] = {}
        if v not in self.adj_list:
            self.adj_list[v] = {}
        self.adj_list[u][v] = d
        self.adj_list[v][u] = d  # For undirected graph

    def getnextnodes(self, u):
        
        heap = [(0, u)]
        visited = set()
        while heap:
            d, v = heapq.heappop(heap)
            if v in visited:
                continue
            visited.add(v)
            yield d,v
            for n, dvn in self.adj_list[v].items():
                if n not in visited:
                    heapq.heappush(heap, (d + dvn, n))

    def __contains__(self, n):
        return n in self.adj_list

    def getedges(self, onlyonedirection=False):
        edges = []
        visited = set()

        for u, adj in self.adj_list.items():
            for v, d in adj.items():
                if onlyonedirection:
                    if (v, u) not in visited:
                        edges.append((u, v, d))
                        visited.add((u, v))
                else:
                    edges.append((u, v, d))

        return edges

    def has_edge(self, u, v):
        """Check if there is an edge between nodes u and v."""
        return u in self.adj_list and v in self.adj_list[u]

    
                



# Start from the first point
start_index = 123
# ordered_points = [points[start_index]]
# visited = set([start_index])

# current_index = start_index
# while True:
#     distances, indices = nearbypoints(points[current_index])
#     for i in range(1, len(indices)):
#         if indices[i] not in visited:
#             next_index = indices[i]
#             break
#     else:
#         break
#     if next_index == start_index:  # Check if we have looped back to the start
#         break
#     ordered_points.append(points[next_index])
#     visited.add(next_index)
#     current_index = next_index
#     #print(next_index)



g=Graph()
notvisited=set(range(len(points)))

while notvisited:
    start_index=notvisited.pop()
    potential_connections=[(d,start_index,n) for d,n in buckets.nearby_points(points[start_index]) if n!=start_index]
    heapq.heapify(potential_connections)
    #notvisited.discard(start_index)


    while potential_connections:

        dist,nodevisited,nodenew=heapq.heappop(potential_connections)
        #print(dist,nodevisited,nodenew)

        if nodenew in g:
            nodeinproximity=False
            for distingraph,graphnode in g.getnextnodes(nodenew):#walk through the graph to check if there is a close connection to nodevisited
                if distingraph>5*dist:
                    break
                if graphnode==nodevisited:
                    nodeinproximity=True
                    break
                
            if nodeinproximity:
                continue
        notvisited.discard(nodenew)
        g.add_edge(nodevisited,nodenew,dist)
        #print(nodevisited,nodenew)
        #print(g.adj_list)

        # TODO add new potentuial connections
        for d,n in buckets.nearby_points(points[nodenew]):
            if n==nodenew or g.has_edge(nodenew,n):
                continue
            heapq.heappush(potential_connections,(d,nodenew,n))


#print(np.array([(2,u,v) for u,v,d in g.getedges(True)]))
#print(g.adj_list)
polydata = pv.PolyData(points)
polydata.lines = np.array([(2,u,v) for u,v,d in g.getedges(True)])
from collections import Counter
print(Counter(map(len,g.adj_list.values())))
print([(n,len(x)) for n,x in g.adj_list.items() if len(x)!=2])
plt.add_mesh(polydata, line_width=1)
# ordered_points = np.array(ordered_points)
# #plt.add_points(ordered_points, color='orange', point_size=10, label='Ordered Points')
# line = pv.Spline(ordered_points, 1000)
# plt.add_mesh(line, color='green', line_width=3, label='Fitted Curve')
# plt.add_points(ordered_points[[0,-1]], color='orange', point_size=10, label='Ordered Points')



highlighted_points = points[[n for n,x in g.adj_list.items() if len(x)==3]]
highlighted_point_cloud = pv.PolyData(highlighted_points)
plt.add_mesh(highlighted_point_cloud, color='red', point_size=5, render_points_as_spheres=True)
highlighted_points = points[[n for n,x in g.adj_list.items() if len(x)==1]]
highlighted_point_cloud = pv.PolyData(highlighted_points)
plt.add_mesh(highlighted_point_cloud, color='orange', point_size=5, render_points_as_spheres=True)
plt.show()

"""ids=np.arange(len(vertices)).reshape((-1, 3))
faces = np.concatenate(
    (np.full((len(vertices)//3, 1),3) , ids), axis=1
).ravel()

#print(faces)
mesh=pv.PolyData(np.array(vertices).ravel(), strips=np.array(faces).ravel())
#print(vertices)
plt.add_mesh(mesh,opacity=0.5,show_edges=0,)
plt.show()"""

#[{123: 0.0009568563582173311}, {127: 0.0009568563582173311, 124: 0.005914018933245974, 131: 0.005974652613408568}]


#todo find point from counter
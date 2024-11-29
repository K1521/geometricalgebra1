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


import heapq

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

    
                
bucketsize=0.5
def generate_3d_circle_points(n, radius=1.0, z_value=0.0):
    """
    Generate n points in 3D space that lie on a circle.

    Parameters:
    n (int): Number of points.
    radius (float): Radius of the circle.
    z_value (float): The Z coordinate for all points (since the circle lies in the XY plane).

    Returns:
    np.ndarray: An array of shape (n, 3) containing the points.
    """
    # Angles for the points
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # X and Y coordinates
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # Z coordinate (same for all points, forming a circle in the XY plane)
    z = np.full_like(x, z_value)

    # Combine x, y, and z into an array of shape (n, 3)
    points = np.column_stack((x, y, z))

    return points

points=generate_3d_circle_points(12)


def nearbypoints(p):#testversion
    distances=np.linalg.norm(points-p,axis=1)
    s=np.argsort(distances)
    return zip(distances[s],np.arange(len(points))[s])

start_index=0


g=Graph()
potential_connections=[(d,start_index,n) for d,n in nearbypoints(points[start_index])]
heapq.heapify(potential_connections)
visited=set()

while potential_connections:

    dist,nodevisited,nodenew=heapq.heappop(potential_connections)
    #print(dist,nodevisited,nodenew)

    if nodenew in g:
        nodeinproximity=False
        for distingraph,graphnode in g.getnextnodes(nodenew):#walk through the graph to check if there is a close connection to nodevisited
            if distingraph>2*dist:
                print("stoop")
                break
            if graphnode==nodevisited:
                nodeinproximity=True
                print("hi")
                break
            
        if nodeinproximity:
            #continue
            print("lllllllllllll")
            continue
        else:print("ggggggggg")
    
    g.add_edge(nodevisited,nodenew,dist)
    print(nodevisited,nodenew)
    #print(g.adj_list)

    # TODO add new potentuial connections
    for d,n in nearbypoints(points[nodenew]):
        if n==nodenew or g.has_edge(nodenew,n):
            continue
        heapq.heappush(potential_connections,(d,nodenew,n))


#print(np.array([(2,u,v) for u,v,d in g.getedges(True)]))
#print(g.adj_list)
polydata = pv.PolyData(points)
polydata.lines = np.array([(2,u,v) for u,v,d in g.getedges(True)])
from collections import Counter
print(Counter(map(len,g.adj_list.values())))

plt.add_mesh(polydata, line_width=1)
# ordered_points = np.array(ordered_points)
# #plt.add_points(ordered_points, color='orange', point_size=10, label='Ordered Points')
# line = pv.Spline(ordered_points, 1000)
# plt.add_mesh(line, color='green', line_width=3, label='Fitted Curve')
# plt.add_points(ordered_points[[0,-1]], color='orange', point_size=10, label='Ordered Points')


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



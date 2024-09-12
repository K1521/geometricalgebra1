
import heapq
from .bucket_grid import BucketGrid
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

    
                
def pointstocurve(points,bucketsize,distfactor=3):

    buckets=BucketGrid(points,bucketsize,merge_close_points=True)
    g=Graph()
    notvisited=set(buckets.getallidxs())

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
                    if distingraph>distfactor*dist:
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


            for d,n in buckets.nearby_points(points[nodenew]):
                if n==nodenew or g.has_edge(nodenew,n):
                    continue
                heapq.heappush(potential_connections,(d,nodenew,n))
    
    return g

                
def pointstocurvesimple(points,bucketsize):

    buckets=BucketGrid(points,bucketsize,merge_close_points=True)
    g=Graph()
    notvisited=set(buckets.getallidxs())

    while notvisited:
        act_index=notvisited.pop()

        while True:
            for d,newindex in buckets.nearby_points(points[act_index]):#search closest point
                if newindex in notvisited:
                    g.add_edge(newindex,act_index,d)
                    notvisited.discard(newindex)
                    act_index=newindex
                    break
            else:
                break
    return g
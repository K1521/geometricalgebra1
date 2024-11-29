import numpy as np

class BucketGrid:
    def __init__(self, points, bucketsize,merge_close_points=False):
        """
        Initialize the BucketGrid with a set of 3D points and voxel parameters.

        :param points: nx3 array representing the points in 3D space
        :param bucketsize: Bucket size used for partitioning the space
        """
        self.points = points
        self.bucketsize = bucketsize
        self.buckets = self._create_buckets()
        if merge_close_points:
            self._merge_close_points(bucketsize/10)

    def _get_bucket(self, vector):
        """
        Get the grid bucket for a given vector.

        :param vector: 3D vector
        :return: Tuple representing the bucket the vector belongs to
        """
        return np.floor(vector / self.bucketsize).astype(int)
    
            

    def _merge_close_points(self,mergedist):
        #x=self.points.copy()
        for bucket,pointidxs in self.buckets.items():

            #basically this algorithm https://stackoverflow.com/a/19375910 wich is only a simple aproximation
            n=len(pointidxs)
            if n==1:
                continue
            points=self.points[pointidxs]
            taken=[False]*n
            for i in range(n):
                if taken[i]:
                    continue

                dists=np.linalg.norm(points[i+1:]-points[i],axis=1)
                count=1
                for j,d in enumerate(dists,start=i+1):# average of points with small dist
                    if d<mergedist:
                        taken[j]=True
                        count+=1
                        points[i]+=points[j]
                points[i]/=count
                #print(count)
            pointidxs=[idx for idx,took in zip(pointidxs,taken) if not took]
            self.buckets[bucket]=pointidxs
        #print(np.allclose(x,self.points))


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
    
    def getallidxs(self):
        return [pointsidx for pointsidxs in self.buckets.values() for pointsidx in pointsidxs]
            

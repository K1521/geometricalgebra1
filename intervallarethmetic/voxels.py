import numpy as np
import itertools

from intervallarethmetic.intervallarethmetic1 import intervallareth

def uniquereplacement(points):
    #points=list(map(tuple,points))
    indexdict=dict()
    #indexreverse=[]
    #for p in points:
    #    indexreverse.append(indexdict.setdefault(p,len(indexdict)))
        #indexdict.setdefault(p,len(indexdict))
        #indexreverse.append(indexdict[p])
        #index=indexdict.get(p,None)
        #if index is None:
        #    indexdict[p]=len()
        #indexreverse.append(index)
    indexreverse=[indexdict.setdefault(p,len(indexdict)) for p in zip(*points.T)]

    pointsnew=np.array(list(indexdict.keys()))
    #print(pointsnew)
    return pointsnew,np.array(indexreverse)




class Voxels:
    subvox=np.array(list(v[::-1] for v in itertools.product((0,1),repeat=3)))
    def __init__(self,delta):
        self.delta=delta
        self.voxels=Voxels.subvox-1#np.array(list(itertools.product((0,-1),repeat=3)))
    def __len__(self):
        return len(self.voxels)
    def intervallarethpoints(self):
        """min/max for x,y,z for each cube"""
        tvox=self.voxels.T
        return[intervallareth(axismin,axismax) for axismin,axismax in zip(tvox*self.delta,(tvox+1)*self.delta)]
    def subdivide(self):
        """subdivides each voxel in 8 smaller voxels"""
        self.voxels=np.vstack(self.voxels[:,None,:]*2+Voxels.subvox)
        self.delta/=2
    def filter_cells(self,mask):
        """removes cells for which the mask is False"""
        self.voxels=self.voxels[mask,:]
    def gridify(self):
        """makes a pyvista grid from the voxels"""

        import vtk
        from vtk.util import numpy_support as nps
        
        grid = vtk.vtkUnstructuredGrid()

        n_cells=len(self.voxels)
        points,ids=self.cubecordsunique()

        pts = vtk.vtkPoints()
        pts.SetData(nps.numpy_to_vtk(points))
        grid.SetPoints(pts)

        cells = vtk.vtkCellArray()
        cells_mat = np.hstack(
            (np.full((n_cells, 1),8) , ids)
        ).ravel()
        cells.SetNumberOfCells(n_cells)
        cells.SetCells(
            n_cells, nps.numpy_to_vtk(cells_mat, array_type=vtk.VTK_ID_TYPE)#deep=True,)
        )
        grid.SetCells(vtk.VTK_VOXEL, cells)

        return grid

    def cubecordsunique(self):
        """
        Computes unique voxel vertex coordinates and their indices.

        Returns:
            tuple: 
                - points (numpy.ndarray): Array of unique voxel vertex coordinates with shape (p, 3).
                - rindices (numpy.ndarray): Indices of these coordinates for each voxel with shape (n, 8).
        
        points,rindices=v.cubecordsunique()
        points[rindices] is the same as v.cubecords()
        
        """

        all_points=(self.voxels[:,None,:]+Voxels.subvox).reshape(-1,3)  #this was a nx8x3 array before reshaping
        points, rindices = np.unique(all_points , return_inverse=True, axis=0)
        return points*self.delta,rindices.reshape((-1, 8))#scaled voxel and the reverse indices


    def cubecords(self):
        """
        cordinate of the vertices of each voxel
        returns a nx8x3 array
        """

        return (self.voxels[:,None,:]+Voxels.subvox)*self.delta


    def cubemid(self):
        """cordinate of the middle of each voxel"""
        return self.voxels*self.delta+self.delta/2
    

    def cubeidx(self):
        """deprecated"""
        return np.arange(len(self.voxels)*8).reshape(-1,8)



#print(Voxels(1).voxels)
if __name__=="__main__":
    voxels=[]
    for i in [1,0]:
        for j in [1,0]:
            for k in [1,0]:
                voxels.append((i,j,k))
    print(Voxels.subvox)
    print(list(itertools.product((0,1),repeat=3)))
    v=Voxels(1)
    #print(v.cubecords())
    print(v.voxels)
    print(Voxels.subvox-1)

import numpy as np
import itertools
from vtk.util import numpy_support as nps
import vtk
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
    def intervallarethpoints(self):
        #min/max for x,y,z for each cube
        tvox=self.voxels.T
        return[intervallareth(axismin,axismax) for axismin,axismax in zip(tvox*self.delta,(tvox+1)*self.delta)]
    def subdivide(self):
        #subdivides each voxel
        self.voxels=np.vstack(self.voxels[:,None,:]*2+Voxels.subvox)
        self.delta/=2
    def removecells(self,mask):
        self.voxels=self.voxels[mask,:]
    def gridify(self):
        #makes a pyvista grid from the voxels
        grid = vtk.vtkUnstructuredGrid()

        #make voxels
        #self.voxels only contains the lower left rear point of the voxel
        #all_points contains all voxel points
        #this works because subvox has different dimensions than voxels
        #all_points=np.vstack(self.voxels[:,None,:]+Voxels.subvox)

        # all_points=(self.voxels[:,None,:]+Voxels.subvox).reshape(-1,3)
        

        # n_cells=len(self.voxels)

        # #remove duplicates
        # points, ind_nodes = np.unique(all_points , return_inverse=True, axis=0)
        # #points, ind_nodes =uniquereplacement(all_points)
        
        # #scale to cordinates
        # points=points*self.delta

        n_cells=len(self.voxels)
        points,ids=self.cubecordsunique()

        pts = vtk.vtkPoints()
        pts.SetData(nps.numpy_to_vtk(points))
        grid.SetPoints(pts)

        cells = vtk.vtkCellArray()
        #insert 8 every 8th index because a voxel has 8 points
        #ids=ind_nodes.reshape((n_cells, 8))
        # cells_mat = np.concatenate(
        #     (np.full((n_cells, 1),8) , ids), axis=1
        # ).ravel()

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
        # intervallx,intervally,intervallz=voxels.intervallarethpoints()
        # c_n0 = (intervallx.min, intervally.min, intervallz.min)#000
        # c_n1 = (intervallx.max, intervally.min, intervallz.min)#100
        # c_n2 = (intervallx.min, intervally.max, intervallz.min)#010
        # c_n3 = (intervallx.max, intervally.max, intervallz.min)#110
        # c_n4 = (intervallx.min, intervally.min, intervallz.max)#001
        # c_n5 = (intervallx.max, intervally.min, intervallz.max)#101
        # c_n6 = (intervallx.min, intervally.max, intervallz.max)#011
        # c_n7 = (intervallx.max, intervally.max, intervallz.max)#111

        # #combined_array=np.stack((c_n0, c_n1, c_n2, c_n3, c_n4, c_n5, c_n6, c_n7), axis=0)
        # combined_array=np.stack((c_n0, c_n4, c_n6, c_n2, c_n3, c_n7, c_n5, c_n1), axis=0)
        # combined_array=np.transpose(combined_array, (2, 0, 1))
        # return combined_array
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

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
        self.voxels=np.array(list(itertools.product((0,-1),repeat=3)))
    def intervallarethpoints(self):
        tvox=self.voxels.T
        return[intervallareth(axismin,axismax) for axismin,axismax in zip(tvox*self.delta,(tvox+1)*self.delta)]
    def voxelmid(self):
        return (self.voxels.T+0.5)*self.delta
    def subdivide(self):
        self.voxels=np.vstack(self.voxels[:,None,:]*2+Voxels.subvox)
        self.delta/=2
    def removecells(self,mask):
        self.voxels=self.voxels[mask,:]
    def gridify(self):
        grid = vtk.vtkUnstructuredGrid()

        #make voxels
        #self.voxels only contains the lower left rear point of the voxel
        #all_points contains all voxel points
        #this works because subvox has different dimensions than voxels
        #all_points=np.vstack(self.voxels[:,None,:]+Voxels.subvox)
        all_points=(self.voxels[:,None,:]+Voxels.subvox).reshape(-1,3)
        

        n_cells=len(self.voxels)

        #remove duplicates
        points, ind_nodes = np.unique(all_points , return_inverse=True, axis=0)
        #points, ind_nodes =uniquereplacement(all_points)
        
        #scale to cordinates
        points=points*self.delta
        pts = vtk.vtkPoints()
        pts.SetData(nps.numpy_to_vtk(points))
        grid.SetPoints(pts)

        cells = vtk.vtkCellArray()
        #insert 8 every 8th index because a voxel has 8 points
        ids=ind_nodes.reshape((n_cells, 8))
        cells_mat = np.concatenate(
            (np.full((n_cells, 1),8) , ids), axis=1
        ).ravel()

        cells.SetNumberOfCells(n_cells)
        cells.SetCells(
            n_cells, nps.numpy_to_vtk(cells_mat, deep=True, array_type=vtk.VTK_ID_TYPE)
        )
        grid.SetCells(vtk.VTK_VOXEL, cells)

        return grid


#print(Voxels(1).voxels)
if __name__=="__main__":
    voxels=[]
    for i in [1,0]:
        for j in [1,0]:
            for k in [1,0]:
                voxels.append((i,j,k))
    print(Voxels.subvox)
    print(list(itertools.product((0,1),repeat=3)))

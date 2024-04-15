import numpy as np
import vtk
grid = vtk.vtkUnstructuredGrid()



n_cells = len(intervallx.min)


c_n0 = np.stack((intervallx.min, intervally.min, intervallz.min), axis=1)
c_n1 = np.stack((intervallx.max, intervally.min, intervallz.min), axis=1)
c_n2 = np.stack((intervallx.min, intervally.max, intervallz.min), axis=1)
c_n3 = np.stack((intervallx.max, intervally.max, intervallz.min), axis=1)
# - Top
c_n4 = np.stack((intervallx.min, intervally.min, intervallz.max), axis=1)
c_n5 = np.stack((intervallx.max, intervally.min, intervallz.max), axis=1)
c_n6 = np.stack((intervallx.min, intervally.max, intervallz.max), axis=1)
c_n7 = np.stack((intervallx.max, intervally.max, intervallz.max), axis=1)

# - Concatenate
#all_nodes = np.concatenate(
#    (c_n1, c_n2, c_n3, c_n4, c_n5, c_n6, c_n7, c_n8), axis=0
#)
all_nodes=np.empty((n_cells*8,3))
all_nodes[0::8] = c_n0
all_nodes[1::8] = c_n1
all_nodes[2::8] = c_n2
all_nodes[3::8] = c_n3
all_nodes[4::8] = c_n4
all_nodes[5::8] = c_n5
all_nodes[6::8] = c_n6
all_nodes[7::8] = c_n7

#print(all_nodes[0::8,(0,1)])
all_nodes, ind_nodes = np.unique(all_nodes , return_inverse=True, axis=0)
#ind_nodes=np.arange(n_cells*8)
from vtk.util import numpy_support as nps
cells = vtk.vtkCellArray()
pts = vtk.vtkPoints()






# Add unique nodes as points in output
#pts.SetData(interface.convert_array(all_nodes))
pts.SetData(nps.numpy_to_vtk(all_nodes))
#pts.SetData(pv.vtk_points(all_nodes))
#pts=pv.vtk_points(all_nodes)
#pts=pv.PointSet(all_nodes)
#print(pts)
# Add cell vertices
#j = np.tile(np.arange(8), n_cells)* n_cells
#print(j)
#arridx = np.add(j, np.repeat(np.arange(n_cells), 8))
arridx=np.arange(8*n_cells)
#print(arridx)
print(ind_nodes[arridx][:10])
print(arridx[:10])
ids = ind_nodes[arridx].reshape((n_cells, 8))

cells_mat = np.concatenate(
    (np.full((n_cells, 1),8) , ids), axis=1
)

cells = vtk.vtkCellArray()
cells.SetNumberOfCells(n_cells)
cells.SetCells(
    n_cells, nps.numpy_to_vtk(cells_mat.ravel(), deep=True, array_type=vtk.VTK_ID_TYPE)
)
# Set the output
grid.SetPoints(pts)
grid.SetCells(vtk.VTK_VOXEL, cells)



print(time.time()-t0)
#grid = pv.UnstructuredGrid(cells_mat.ravel(), np.array([pv.CellType.VOXEL]*len(cells_mat), np.int8), all_nodes.ravel())
plt.add_mesh(grid,opacity=0.5)
plt.show()


#entweder f'==0 oder vorzeichenwechsel in den punkten
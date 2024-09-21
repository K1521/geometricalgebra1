from intervallarethmetic.intervallareth3d1 import inter3d
from intervallarethmetic.intervallareth3d2 import poly3d
from intervallarethmetic.voxels import Voxels
import numpy as np

from abc import ABC, abstractmethod

# Abstract base class for voxel processing
class VoxelProcessing(ABC):
    def __init__(self, voxels, f):
        self.voxels = voxels
        self.f = f
    
    @abstractmethod
    def process_step(self):
        pass
    
    def processn(self, depth, maxvoxelnum):
        """Class method for the common processing loop"""
        for j in range(1, depth + 1):
            self.process_step()
            if len(self.voxels) > maxvoxelnum:
                print(f"Exceeded max voxel number at depth {j}")
                break
            self.voxels.subdivide()

# Stateless processing classes
class RemoveEmptyVoxels(VoxelProcessing):
    def process_step(self):
        # Implement logic similar to remove_mask_empty_voxels_by_alignement
        print("Removing empty voxels by alignment")

class ProcessIntervalArithmetic(VoxelProcessing):
    def process_step(self):
        x, y, z = self.voxels.cubemid().T
        expr = self.f(VoxelIntervalArithmetic.ix * self.voxels.delta / 2 + x,
                     VoxelIntervalArithmetic.iy * self.voxels.delta / 2 + y,
                     VoxelIntervalArithmetic.iz * self.voxels.delta / 2 + z)
        dat = [blade.magnitude.intervallnp().containsnum(0) for blade in expr.lst[:]]
        voxelswithzerro = np.all(np.broadcast_arrays(*dat), axis=0)
        self.voxels.filter_cells(voxelswithzerro)
        print("Processed interval arithmetic")

# Stateful processing class
class ProcessPolynomialArithmetic(VoxelProcessing):
    def __init__(self, voxels, f):
        super().__init__(voxels, f)
        # Precompute expr based on f
        self.expr = self.f(VoxelIntervalArithmetic.ix, 
                           VoxelIntervalArithmetic.iy, 
                           VoxelIntervalArithmetic.iz)

    def process_step(self):
        x, y, z = self.voxels.cubemid().T
        # Reuse self.expr in calculations
        print("Processed polynomial arithmetic using precomputed expression")
        # Rest of the process logic

# Usage
def main():
    # Create some voxel data
    voxels = Voxels(16)
    f = lambda x, y, z: x * y + z  # Example function, replace with your own

    # Process using interval arithmetic
    interval_processor = ProcessIntervalArithmetic(voxels, f)
    interval_processor.processn(depth=10, maxvoxelnum=5000)
    
    # Process using polynomial arithmetic (with state)
    polynomial_processor = ProcessPolynomialArithmetic(voxels, f)
    polynomial_processor.processn(depth=10, maxvoxelnum=5000)

    # Process using remove empty voxels logic
    empty_voxels_processor = RemoveEmptyVoxels(voxels, f)
    empty_voxels_processor.processn(depth=10, maxvoxelnum=5000)

if __name__ == "__main__":
    main()

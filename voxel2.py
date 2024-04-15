import numpy as np
class Voxels:
    def __init__(self,delta):
        self.dx=self.dy=self.dz=delta
        
        #voxels=[]
        #for i in [-1,0]:
        #    for j in [-1,0]:
        #        for k in [-1,0]:
        #            voxels.append((i,j,k))
        self.voxels=np.array([(-1, -1, -1), (-1, -1, 0), (-1, 0, -1), (-1, 0, 0), (0, -1, -1), (0, -1, 0), (0, 0, -1), (0, 0, 0)])
    def subdivide(self):
        self.voxels=np.vstack()
#print(Voxels(1).voxels)
voxels=[]
for i in [-1,0]:
    for j in [-1,0]:
        for k in [-1,0]:
            voxels.append((i,j,k))
v=np.array(voxels)[:,None,:]
print(v)
#print(np.array(voxels))
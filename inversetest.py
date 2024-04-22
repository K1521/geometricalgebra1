
import numpy as np
def uniquepoints(points,maxmergedist=1):
    #merges most points wich are closer than maxmergedist
    #merges all points closer than maxmergedist/4
    #works for 2d or 3d points
    #distance metric is max(abs(dx),abs(dy),abs(dz))
    punique=points
    for offset in range(4):
        idx=np.unique(np.around(punique/maxmergedist+offset/4),axis=0,return_index=True)[1]
        punique=punique[idx]
    return punique


p=np.random.random((1000,2))*100
threshold = 3
clustered = uniquepoints(np.array(p),threshold)


import matplotlib.pyplot as plt

plt.plot(*p.T, marker='o', color='r', ls='')
plt.plot(*clustered.T, marker='.', color='g', ls='')
plt.show()
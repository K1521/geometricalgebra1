import numpy as np
import pyvista as pv




bounds=np.array((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0))/2
b1=pv.Box(bounds=bounds)
b2=pv.Box(bounds=bounds)
b3=pv.Box(bounds=bounds)
b1.points+=0#np.array([1,0,0])
b2.points+=1/4#+np.array([1,0,0])
b3.points+=2/4#+np.array([1,1,0])
#intersection1 = b1.intersection(b2)[0]
#intersection2 = b2.intersection(b3)[0]
#intersection3 = b1.intersection(b3)[0]



plt=pv.Plotter()
plt.add_mesh(b1,opacity=0.5,color="blue",show_edges=True)
plt.add_mesh(b2,opacity=0.5,color="red",show_edges=True)
plt.add_mesh(b3,opacity=0.5,color="green",show_edges=True)
##plt.add_mesh(intersection, line_width=5)






n=1000000
d=3
p1=np.array([10]+[0]*(d-1))*np.arange(n)[:,None]+np.random.random((n,d))*3
p2=p1+np.random.random((n,3))*0.1
pall=np.vstack([p1,p2])
pindex=np.concatenate([np.arange(n),np.arange(n)])
print(pindex)
punique=pall
idx=np.unique(np.around(punique),return_index=True,axis=0)[1]
punique=punique[idx]
pindex=pindex[idx]
idx=np.unique(np.around(punique+np.array([1,1,1])/4),return_index=True,axis=0)[1]
punique=punique[idx]
pindex=pindex[idx]
idx=np.unique(np.around(punique+np.array([2,2,2])/4),return_index=True,axis=0)[1]
punique=punique[idx]
pindex=pindex[idx]
idx=np.unique(np.around(punique+np.array([3,3,3])/4),return_index=True,axis=0)[1]
punique=punique[idx]
pindex=pindex[idx]

pindex,counts=np.unique(pindex,return_counts=True)
#print(pindex[counts==2])
badidx=pindex[counts==2]
print(n,len(badidx))
pdoppel=pall[np.concatenate([badidx,badidx+n])]
plt.add_points(
    pdoppel%1,
    render_points_as_spheres=True,
    point_size=20
)

plt.show()

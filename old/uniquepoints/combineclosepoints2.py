import numpy as np

n=10000
d=3
p1=np.array([10]+[0]*(d-1))*np.arange(n)[:,None]+np.random.random((n,d))*3
p2=p1+np.random.random((n,3))*0.1
pall=np.vstack([p1,p2])

punique=pall
print(len(punique))
punique=punique[np.unique(np.around(punique),return_index=True,axis=0)[1]]
print(len(punique))
punique=punique[np.unique(np.around(punique+1/3),return_index=True,axis=0)[1]]
print(len(punique))
punique=punique[np.unique(np.around(punique-1/3),return_index=True,axis=0)[1]]
print(len(punique))
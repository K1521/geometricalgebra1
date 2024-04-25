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

n=1000000
d=2
for i in range(1000):

    p1=np.array([10]+[0]*(d-1))*np.arange(n)[:,None]+np.random.random((n,d))*3
    p2=p1+np.random.random((n,d))*0.24
    pall=np.vstack([p1,p2])
    punique=pall
    punique = uniquepoints(punique)


    print(i,len(punique))
    if len(punique)!=n:
        print(":(")
        break
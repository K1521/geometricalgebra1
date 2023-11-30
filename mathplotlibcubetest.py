



import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d') 

ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(-20, 20)

#p1--p2
#|    |
#p3--p4

p1=[-10,10,0]
p2=[10,10,0]
p3=[-10,-10,0]
p4=[10,-10,0]

X,Y,Z=[np.asarray([[p1[i],p2[i]],[p3[i],p4[i]]]) for i in range(3)]

ax.plot_surface(X,Y,Z)
#ax.plot_surface(np.asarray([[-10,10],[-10,10]]), np.asarray([[10,10],[-10,-10]]), np.asarray([[0,0],[0,0]]))

plt.show()




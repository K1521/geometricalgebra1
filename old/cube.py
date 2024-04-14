



    
import numpy as np
import time
def rotate(M,P):
    P1 = np.zeros(16)
    P1[11] = (2.0 * M[10] * M[8] - 2.0 * M[0] * M[9]) * P[13] + (2.0 * M[8] * M[9] + 2.0 * M[0] * M[10]) * P[12] + ((-(M[9] * M[9])) + M[8] * M[8] - M[10] * M[10] + M[0] * M[0]) * P[11] - 2.0 * M[5] * M[9] + 2.0 * M[15] * M[8] + 2.0 * M[0] * M[7] - 2.0 * M[10] * M[6] # e0 ^ (e1 ^ e2)
    P1[12] = (2.0 * M[10] * M[9] + 2.0 * M[0] * M[8]) * P[13] + (M[9] * M[9] - M[8] * M[8] - M[10] * M[10] + M[0] * M[0]) * P[12] + (2.0 * M[8] * M[9] - 2.0 * M[0] * M[10]) * P[11] + 2.0 * M[15] * M[9] + 2.0 * M[5] * M[8] - 2.0 * M[10] * M[7] - 2.0 * M[0] * M[6] # e0 ^ (e1 ^ e3)
    P1[13] = ((-(M[9] * M[9])) - M[8] * M[8] + M[10] * M[10] + M[0] * M[0]) * P[13] + (2.0 * M[10] * M[9] - 2.0 * M[0] * M[8]) * P[12] + (2.0 * M[0] * M[9] + 2.0 * M[10] * M[8]) * P[11] + 2.0 * M[7] * M[9] + 2.0 * M[6] * M[8] + 2.0 * M[0] * M[5] + 2.0 * M[10] * M[15] # e0 ^ (e2 ^ e3)
    P1[14] = M[9] * M[9] + M[8] * M[8] + M[10] * M[10] + M[0] * M[0] # e1 ^ (e2 ^ e3)
    return P1



def geopoint(x, y, z):
    P = np.zeros(16)
    P[11] = (-z) # e0 ^ (e1 ^ e2)
    P[12] = y # e0 ^ (e1 ^ e3)
    P[13] = (-x) # e0 ^ (e2 ^ e3)
    P[14] = 1.0 # e1 ^ (e2 ^ e3)
    return P



def eucpoint(P):
    #print(P)
    z=-P[11] /P[14] # e0 ^ (e1 ^ e2)
    y=P[12]/P[14] # e0 ^ (e1 ^ e3)
    x=-P[13]/P[14]  # e0 ^ (e2 ^ e3)
    return x,y,z


M = np.zeros(16)
M[0] = 1 # 1.0
M[5] = 0 # e0 ^ e1
M[6] = 0 # e0 ^ e2
M[7] = 0 # e0 ^ e3
M[8] = 0 # e1 ^ e2
M[9] = 0 # e1 ^ e3
M[10] = 0 # e2 ^ e3
M[15] =  0# e0 ^ (e1 ^ (e2 ^ e3))

B = np.zeros(16)
B[5] = 0 # e0 ^ e1
B[6] = 0 # e0 ^ e2
B[7] = 0 # e0 ^ e3
B[8] = 0.004 # e1 ^ e2
B[9] = 0.002 # e1 ^ e3
B[10] = 0.001 # e2 ^ e3
B[15] = 0 # e0 ^ (e1 ^ (e2 ^ e3))

#?M = m0 + m1*(e0^e1) + m2*(e0 ^ e2) + m3*(e0 ^ e3)+m4*(e1 ^ e2)+m5*(e1 ^ e3)+m6*(e2 ^ e3)+m7*(e0 ^ (e1 ^ (e2 ^ e3)));
#?B = b1*(e0^e1) + b2*(e0 ^ e2) + b3*(e0 ^ e3)+b4*(e1 ^ e2)+b5*(e1 ^ e3)+b6*(e2 ^ e3)+b7*(e0 ^ (e1 ^ (e2 ^ e3)));
def mbnext(M,B, rate):
    MNew = np.zeros(16)
    MNew[0] = (B[9] / 2.0 * M[9] + B[8] / 2.0 * M[8] + B[10] / 2.0 * M[10]) * rate + M[0] # 1.0
    MNew[5] = ((-(B[7] / 2.0 * M[9])) - B[6] / 2.0 * M[8] + B[9] / 2.0 * M[7] + B[8] / 2.0 * M[6] + B[10] / 2.0 * M[15] + B[15] / 2.0 * M[10] - B[5] / 2.0 * M[0]) * rate + M[5] # e0 ^ e1
    MNew[6] = ((-(B[15] / 2.0 * M[9])) + B[5] / 2.0 * M[8] + B[10] / 2.0 * M[7] - B[8] / 2.0 * M[5] - B[9] / 2.0 * M[15] - B[7] / 2.0 * M[10] - B[6] / 2.0 * M[0]) * rate + M[6] # e0 ^ e2
    MNew[7] = (B[5] / 2.0 * M[9] + B[15] / 2.0 * M[8] - B[10] / 2.0 * M[6] - B[9] / 2.0 * M[5] + B[8] / 2.0 * M[15] + B[6] / 2.0 * M[10] - B[7] / 2.0 * M[0]) * rate + M[7] # e0 ^ e3
    MNew[8] = (B[10] / 2.0 * M[9] - B[9] / 2.0 * M[10] - B[8] / 2.0 * M[0]) * rate + M[8] # e1 ^ e2
    MNew[9] = ((-(B[10] / 2.0 * M[8])) + B[8] / 2.0 * M[10] - B[9] / 2.0 * M[0]) * rate + M[9] # e1 ^ e3
    MNew[10] = ((-(B[8] / 2.0 * M[9])) + B[9] / 2.0 * M[8] - B[10] / 2.0 * M[0]) * rate + M[10] # e2 ^ e3
    MNew[15] = (B[6] / 2.0 * M[9] - B[7] / 2.0 * M[8] - B[8] / 2.0 * M[7] + B[9] / 2.0 * M[6] - B[10] / 2.0 * M[5] - B[5] / 2.0 * M[10] - B[15] / 2.0 * M[0]) * rate + M[15] # e0 ^ (e1 ^ (e2 ^ e3))
    BNew = np.zeros(16)
    BNew[5] = (B[7] * B[9] + B[6] * B[8]) * rate + B[5] # e0 ^ e1
    BNew[6] = (B[10] * B[7] - B[5] * B[8]) * rate + B[6] # e0 ^ e2
    BNew[7] = ((-(B[5] * B[9])) - B[10] * B[6]) * rate + B[7] # e0 ^ e3
    BNew[8] = B[8] # e1 ^ e2
    BNew[9] = B[9] # e1 ^ e3
    BNew[10] = B[10] # e2 ^ e3
    BNew[15] = B[15] # e0 ^ (e1 ^ (e2 ^ e3))
    return MNew,BNew

PS=[geopoint(x,y,z) for x,y,z in 
    [(-1,-1,-1),(-1,-1,1),
     (-1,1,-1),(-1,1,1),

     (1,-1,-1),(1,-1,1),
     (1,1,-1),(1,1,1)]]

"""0 1
2 3
 
4 5
6 7"""



#PS=[geopoint(x,y) for x,y in [(0,0),(0,1),(1,1),(1,0)]]
#import pyvista
#chart = pyvista.Chart2D()
#plot = chart.line([0, 1, 2], [2, 1, 3])
#chart.show()

import pyvista as pv

pv.set_plot_theme('dark')
p = pv.Plotter()
p.add_axes()
#p.show_grid()
#print(np.array([eucpoint(point) for point in PS]))
cube=pv.PolyData(np.array([eucpoint(point) for point in PS]), faces=np.array([4,0,1,3,2, 4,4,5,7,6, 4,0,1,5,4, 4,2,3,7,6, 4,1,3,7,5, 4,0,2,6,4]))
p.add_mesh(cube,lighting=True)
#p.show()


p.show(interactive_update=True)
running=True
t0=time.time()
i=0
while running:
    for _ in range(100):
        M,B=mbnext(M,B,1/100)
    #print(np.array([rotate(M,p) for p in PS]))
    cube.points=np.array([eucpoint(rotate(M,p)) for p in PS])
    #time.sleep(1/10)
    p.update(0.1)

    i+=1
    if i!=1:
        print(1/((time.time()-t0)/i))

    #p.update_coordinates(np.array([eucpoint(rotate(M,p)) for p in PS]), render=False)
    #print([rotate(M,p) for p in PS])
    


    
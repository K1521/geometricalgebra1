



    
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
B[8] = 4 # e1 ^ e2
B[9] = 2 # e1 ^ e3
B[10] = 1 # e2 ^ e3
B[15] = 0 # e0 ^ (e1 ^ (e2 ^ e3))

#?M = m0 + m1*(e0^e1) + m2*(e0 ^ e2) + m3*(e0 ^ e3)+m4*(e1 ^ e2)+m5*(e1 ^ e3)+m6*(e2 ^ e3)+m7*(e0 ^ (e1 ^ (e2 ^ e3)));
#?B = b1*(e0^e1) + b2*(e0 ^ e2) + b3*(e0 ^ e3)+b4*(e1 ^ e2)+b5*(e1 ^ e3)+b6*(e2 ^ e3)+b7*(e0 ^ (e1 ^ (e2 ^ e3)));

"""
?M = m0 + m1*(e0^e1) + m2*(e0 ^ e2) + m3*(e0 ^ e3)+m4*(e1 ^ e2)+m5*(e1 ^ e3)+m6*(e2 ^ e3)+m7*(e0 ^ (e1 ^ (e2 ^ e3)));
?B = b1*(e0^e1) + b2*(e0 ^ e2) + b3*(e0 ^ e3)+b4*(e1 ^ e2)+b5*(e1 ^ e3)+b6*(e2 ^ e3)+b7*(e0 ^ (e1 ^ (e2 ^ e3)));
?P=createPoint(x,y,z)
Gravity = *(~M * (-9.81*(e0^e2)) * M)
//attach=(1-0.5*(e0^e2))*createPoint(0,0,0)
attach=createPoint(0,0,0)
Hooke   = -12*(*( (*(~M * attach * M)) ^ *P ))
//Hooke=0
Damping = *(-0.25 * B)
F=Gravity + Hooke + Damping
dM = -0.5*M*B;
?MNew = M + rate*dM
B_dual = *B

dB = *(F-0.5*(B_dual*B-B*B_dual))
?BNew =B + rate*dB
"""
def mbnext(M,B,P, rate):
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
	BNew[5] = ((12.0 * M[9] * M[9] + 12.0 * M[8] * M[8] + 12.0 * M[10] * M[10] + 12.0 * M[0] * M[0]) * P[13] + (19.62 * M[10] - 24.0 * M[7]) * M[9] + (19.62 * M[0] - 24.0 * M[6]) * M[8] + 24.0 * M[0] * M[5] + 24.0 * M[10] * M[15] + B[7] * B[9] + B[6] * B[8] - 0.25 * B[5]) * rate + B[5] # e0 ^ e1
	BNew[6] = (((-(12.0 * M[9] * M[9])) - 12.0 * M[8] * M[8] - 12.0 * M[10] * M[10] - 12.0 * M[0] * M[0]) * P[12] - 9.81 * M[9] * M[9] - 24.0 * M[15] * M[9] + 9.81 * M[8] * M[8] + 24.0 * M[5] * M[8] - 24.0 * M[10] * M[7] + 24.0 * M[0] * M[6] + 9.81 * M[10] * M[10] - 9.81 * M[0] * M[0] - B[5] * B[8] + B[10] * B[7] - 0.25 * B[6]) * rate + B[6] # e0 ^ e2
	BNew[7] = ((12.0 * M[9] * M[9] + 12.0 * M[8] * M[8] + 12.0 * M[10] * M[10] + 12.0 * M[0] * M[0]) * P[11] + (19.62 * M[8] + 24.0 * M[5]) * M[9] + 24.0 * M[15] * M[8] + 24.0 * M[0] * M[7] + 24.0 * M[10] * M[6] - 19.62 * M[0] * M[10] - B[5] * B[9] - 0.25 * B[7] - B[10] * B[6]) * rate + B[7] # e0 ^ e3
	BNew[8] = ((24.0 * M[15] * M[9] - 24.0 * M[5] * M[8] + 24.0 * M[10] * M[7] - 24.0 * M[0] * M[6]) * P[13] + (24.0 * M[7] * M[9] + 24.0 * M[6] * M[8] - 24.0 * M[0] * M[5] - 24.0 * M[10] * M[15]) * P[12] - 0.25 * B[8]) * rate + B[8] # e1 ^ e2
	BNew[9] = (((-(24.0 * M[5] * M[9])) - 24.0 * M[15] * M[8] - 24.0 * M[0] * M[7] - 24.0 * M[10] * M[6]) * P[13] + ((-(24.0 * M[7] * M[9])) - 24.0 * M[6] * M[8] + 24.0 * M[0] * M[5] + 24.0 * M[10] * M[15]) * P[11] - 0.25 * B[9]) * rate + B[9] # e1 ^ e3
	BNew[10] = ((24.0 * M[5] * M[9] + 24.0 * M[15] * M[8] + 24.0 * M[0] * M[7] + 24.0 * M[10] * M[6]) * P[12] + ((-(24.0 * M[15] * M[9])) + 24.0 * M[5] * M[8] - 24.0 * M[10] * M[7] + 24.0 * M[0] * M[6]) * P[11] - 0.25 * B[10]) * rate + B[10] # e2 ^ e3
	BNew[15] = B[15] - 0.25 * B[15] * rate # e0 ^ (e1 ^ (e2 ^ e3))
	
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
line=pv.PolyData(np.array([(0,0,0),eucpoint(PS[0])]))
line.lines=np.array([2,0,1])
p.add_mesh(cube,lighting=True)
p.add_mesh(line,lighting=True)

#p.show()
p.camera_position = 'xy'


p.show(interactive_update=True)
running=True
while running:
    for _ in range(10):
        M,B=mbnext(M,B,PS[0],1/1000)
    #print(np.array([rotate(M,p) for p in PS]))
    cube.points=np.array([eucpoint(rotate(M,p)) for p in PS])
    #time.sleep(1/10)
    line.points[1]=eucpoint(rotate(M,PS[0]))
    p.update()
    #print(p.camera_position)

    #p.update_coordinates(np.array([eucpoint(rotate(M,p)) for p in PS]), render=False)
    #print([rotate(M,p) for p in PS])
    


    
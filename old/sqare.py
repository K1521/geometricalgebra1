

"""
nextM = {M = _P(1);B = _P(2);
    dM = -0.5*M*B;
    M + rate*dM
}
nextB = {M = _P(1);B = _P(2);
    B_dual = *B;
    dB = -0.5*(*(B_dual*B-B*B_dual));
    B + rate*dB
}
M = m0 + m1*(e0^e1) + m2*(e2^e0) + m3*(e1^e2);
B = b1*(e0^e1) + b2*(e2^e0) + b3*(e1^e2);
?MNew = nextM(M,B);
?BNew = nextB(M,B);
"""

"""
M = m0 + m1*(e0^e1) + m2*(e2^e0) + m3*(e1^e2);
B = b1*(e0^e1) + b2*(e2^e0) + b3*(e1^e2);
dM = -0.5*M*B;
?MNew = M + rate*dM
B_dual = *B;
dB = -0.5*(*(B_dual*B-B*B_dual));
?BNew =B + rate*dB
"""

import numpy as np



    
import numpy as np
def rotate(M,P):
    #?P=x * e0 ^ e1-y * e0 ^ e2+ e1 ^ e2
    #?M = m0 + m1*(e0^e1) + m2*(e2^e0) + m3*(e1^e2);
    #?P1 = M * P * ~M;
    P1 = np.zeros(8)
    P1[4] = 2.0 * M[0] * M[6] * P[5] + (M[0] * M[0] - M[6] * M[6]) * P[4] + 2.0 * M[4] * M[6] - 2.0 * M[0] * M[5] # e0 ^ e1
    P1[5] = (M[0] * M[0] - M[6] * M[6]) * P[5] - 2.0 * M[0] * M[6] * P[4] + 2.0 * M[5] * M[6] + 2.0 * M[0] * M[4] # e0 ^ e2
    P1[6] = M[6] * M[6] + M[0] * M[0] # e1 ^ e2
    return P1


def geopoint(x,y):
    #?P = createPoint(x, y);
    P = np.zeros(8)
    P[4] = x # e0 ^ e1
    P[5] = -y # e0 ^ e2
    P[6] = 1.0 # e1 ^ e2
    return P

def eucpoint(P):
    x=P[4]/P[6]
    y=-P[5]/P[6]
    return x,y

m0,m1,m2,m3=1,0,0,0
b1,b2,b3=0.3, 0.1, 1
M = np.zeros(8)
M[0] = m0 # 1.0
M[4] = m1 # e0 ^ e1
M[5] = (-m2) # e0 ^ e2
M[6] = m3 # e1 ^ e2
B = np.zeros(8)

B[4] = b1 # e0 ^ e1
B[5] = (-b2) # e0 ^ e2
B[6] = b3 # e1 ^ e2


def mbnext(M,B, rate):
    """
    ?M = m0 + m1*(e0^e1) + m2*(e2^e0) + m3*(e1^e2);
    ?B = b1*(e0^e1) + b2*(e2^e0) + b3*(e1^e2);
    dM = -0.5*M*B;
    ?MNew = M + rate*dM
    B_dual = *B;
    dB = -0.5*(*(B_dual*B-B*B_dual));
    ?BNew =B + rate*dB
    """
    MNew = np.zeros(8)
    MNew[0] = B[6] / 2.0 * M[6] * rate + M[0] # 1.0
    MNew[4] = ((-(B[5] / 2.0 * M[6])) + B[6] / 2.0 * M[5] - B[4] / 2.0 * M[0]) * rate + M[4] # e0 ^ e1
    MNew[5] = (B[4] / 2.0 * M[6] - B[6] / 2.0 * M[4] - B[5] / 2.0 * M[0]) * rate + M[5] # e0 ^ e2
    MNew[6] = M[6] - B[6] / 2.0 * M[0] * rate # e1 ^ e2
    BNew = np.zeros(8)
    BNew[4] = B[5] * B[6] * rate + B[4] # e0 ^ e1
    BNew[5] = B[5] - B[4] * B[6] * rate # e0 ^ e2
    BNew[6] = B[6] # e1 ^ e2
    return  MNew,BNew

PS=[geopoint(x,y) for x,y in [(-0.5,-0.5),(0.5,-0.5),(0.5,0.5),(-0.5,0.5)]]
#PS=[geopoint(x,y) for x,y in [(0,0),(0,1),(1,1),(1,0)]]
#import pyvista
#chart = pyvista.Chart2D()
#plot = chart.line([0, 1, 2], [2, 1, 3])
#chart.show()

import pygame as pg

#print(mbnext(M, np.zeros(8),1/1000))
#exit()

screen_color = (49, 150, 100)
line_color = (255, 0, 0)

screen=pg.display.set_mode((1000,1000))#w,h
running=True
while running:
    M,B=mbnext(M,B,1/1000)
    screen.fill((0, 0, 0))
    
    pt=[rotate(M,p) for p in PS]
    for i in range(len(pt)):
        #print(pt[i])
        p1=eucpoint(pt[i])
        p2=eucpoint(pt[i-1])
        #print(p1,p2)
        pg.draw.line(screen,line_color, (p1[0]*50+100,p1[1]*50+100), (p2[0]*50+100,p2[1]*50+100),width=10)
    pg.display.flip()
    for events in pg.event.get():
        if events.type == pg.QUIT:
            running=False

    
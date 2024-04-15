#https://www.youtube.com/watch?v=4r_EvmPKOvY

import numpy as np

I=np.eye(3)
def cross_product_matrix_function(m):
    return np.cross(I,m)

#print(cross_product_matrix_function([1,2,3]))

class ridgidbody:
    def __init__(self) -> None:
        #self.mesh
        self.X=np.zeros(3)#position
        self.V=np.zeros(3)#velocity
        self.R=np.eye(3)#rotation matrix
        #self.omega=np.zeros(3)#rotation axis
        self.L=np.zeros(3)
        self.Inv0=np.eye(3)*6

cube=ridgidbody()

cube.X[:]=[0,1.5,0]
cube.V[:]=[1,0,1]
#cube.omega[:]=[1,1,0]
cube.L[:]=[1,0,0]

dt=0.001






import pyvista as pv

pv.set_plot_theme('dark')
p = pv.Plotter()
p.add_axes()

cubemesh=pv.Cube()
cubepoints=cubemesh.points[:]
p.add_mesh(cubemesh)

p.show(interactive_update=True)

while True:
    

    
    cube.X+=cube.V*dt
    #cube.R+=dt*cross_product_matrix_function(cube.omega)@cube.R
    omega=cube.R@cube.Inv0@cube.R.T@cube.L

    cube.R+=dt*np.cross(cube.R,omega)
    cubemesh.points=(cubepoints @ cube.R.T) +cube.X
    p.update()
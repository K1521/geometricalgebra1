from algebra.dcga import *
import pyvista as pv
import numpy as np
import time


def max_seconds(max_seconds, interval=1):
    current = start_time = time.time()
    end_time = start_time + max_seconds
    while  current<=end_time:
        yield current-start_time
        time.sleep(min(end_time-current,interval+time.time()-current))
        current=time.time()
    yield current-start_time


t=toroid(1,.5)
p=Plane(0.1,0.1,0.1,0.5)
vis=p^t


x = np.linspace(-1, 1, 30)
y = np.linspace(-1, 1, 30)
z = np.linspace(-1, 1, 30)
x, y, z = np.meshgrid(x, y, z, indexing='ij')


expr=vis.inner(point(x,y,z))

#[blade.magnitude for blade in expr.lst[:]]
# Define the grid


# Define the implicit function


# Evaluate the implicit function on the grid
plotter = pv.Plotter()
#plotter.set_background("black")  # Set background to black
actors=[]

for i in range(len(expr.lst)):
    field=expr.lst[i].magnitude#.toscalar()#sum(expr.lst[i].magnitude**2 for i in range(len(expr.lst)))
    

    grid = pv.StructuredGrid(x, y, z)
    grid["values"] = field.ravel(order="F")

    contours = grid.contour([0])
    if contours.n_points != 0:
        actor = plotter.add_mesh(contours, color="white", opacity=0.02)
        actors.append(actor)
print(len(expr.lst))

def highlight_mesh(index):
    for i, actor in enumerate(actors):
        if i == index:
            actor.GetProperty().SetColor(1, 0, 0)  # Highlighted mesh color red
            actor.GetProperty().SetOpacity(1.0)   # Highlighted mesh full opacity
        else:
            actor.GetProperty().SetColor(1, 1, 1)  # Normal mesh color white
            actor.GetProperty().SetOpacity(0.02)  # Normal mesh semi-transparent

# Initial render
plotter.show(interactive_update=True,auto_close=False)

# Animation loop
index = 0
try:
    while True:
        highlight_mesh(index)
        

        for s in max_seconds(1,0.01):
            #print(s)
            plotter.update()
        index = (index + 1) % len(actors)
        print(index)
finally:
    pass
exit()
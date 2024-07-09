import pyvista as pv
from pyvista import examples
import numpy as np
# Load an example mesh
mesh = examples.download_st_helens()
print(mesh)
# Create a plotter object
plotter = pv.Plotter()

# Add the mesh to the plotter
plotter.add_mesh(mesh)

# Set the initial camera position and center of rotation
#initial_position = [1000, 1000, 1000]
#plotter.camera_position = initial_position
#plotter.set_focus([0, 0, 0])  # Center of rotation

# Define movement increment
move_increment = 100

def move_camera(direction):
    #position = np.array(plotter.camera_position.position)
    #print(plotter.camera_position)
    delta=np.array([0,0,0])
    if direction == 'left':
        delta[0]=-1
    elif direction == 'right':
        delta[0]=1
    elif direction == 'up':
        delta[1]=1
    elif direction == 'down':
        delta[1]=-1
    elif direction == 'forward':
        delta[2]=1
    elif direction == 'backward':
        delta[2]=-1
    delta*=move_increment
    print(type(plotter.camera_position))
    print(plotter.camera_position.position,np.array(plotter.camera_position.position)+delta)
    #plotter.camera_position.position = tuple(np.array(plotter.camera_position.position)+delta)
    #plotter.camera_position.focal_point = tuple(np.array(plotter.camera_position.focal_point)+delta)
    plotter.camera_position=[np.array(plotter.camera_position.position)+delta, 
                             np.array(plotter.camera_position.focal_point)+delta, 
                             plotter.camera_position[2]]
    print(plotter.camera_position.position)
    plotter.render()

# Define the key press callbacks
def key_callback_left():
    move_camera('left')

def key_callback_right():
    move_camera('right')

def key_callback_up():
    move_camera('up')

def key_callback_down():
    move_camera('down')

def key_callback_forward():
    move_camera('forward')

def key_callback_backward():
    move_camera('backward')

# Bind the keys to the callbacks
plotter.add_key_event('Left', key_callback_left)
plotter.add_key_event('Right', key_callback_right)
plotter.add_key_event('Up', key_callback_up)
plotter.add_key_event('Down', key_callback_down)
plotter.add_key_event('w', key_callback_forward)
plotter.add_key_event('s', key_callback_backward)

# Start the interactive plotter
plotter.show()
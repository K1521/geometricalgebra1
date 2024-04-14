import numpy as np
import pyvista as pv

identity_matrix = np.eye(3)

def cross_product_matrix(m):
    return np.cross(identity_matrix, m)

class RigidBody:
    def __init__(self, mass=1.0) -> None:
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.rotation_matrix = np.eye(3)
        self.angular_momentum = np.zeros(3)
        self.inertia_tensor_inv = np.eye(3) * 6
        self.mass = mass

# Create a rigid body object with a specified mass
cube = RigidBody(mass=1.0)
cube.position[:] = [0, 3, 0]
cube.velocity[:] = [1, 0, 1]
cube.angular_momentum[:] = [1, 0, 0]

spring_constant = 10.0  # Adjust this constant based on your simulation
dt = 0.0001

# Create a PyVista plotter
pv.set_plot_theme('dark')
plotter = pv.Plotter()
plotter.add_axes()

# Create a cube mesh and add it to the plotter
cube_mesh = pv.Cube()
cube_points = cube_mesh.points[:]
plotter.add_mesh(cube_mesh)
line=pv.PolyData(np.array([(0,0,0),cube_points[0]]))
line.lines=np.array([2,0,1])
plotter.add_mesh(line,lighting=True)

# Show the plot
plotter.show(interactive_update=True)

def apply_force(rigid_body, vertex_index, force, dt):
    torque = np.cross(cube.rotation_matrix@cube_points[vertex_index], force)
    rigid_body.velocity += (force / rigid_body.mass) * dt
    rigid_body.angular_momentum += torque * dt
    #rigid_body.angular_velocity = np.linalg.inv(rigid_body.inertia_tensor_inv) @ rigid_body.angular_momentum

# Function to calculate the spring force between a vertex and the origin
def calculate_spring_force(vertex_position, spring_constant):
    displacement = vertex_position - np.array([0, 0, 0])
    return -spring_constant * displacement

# Function to apply spring force at a specific vertex
def apply_spring_force(rigid_body, vertex_index, spring_constant, dt):
    vertex_position = (cube.rotation_matrix@cube_points[vertex_index]) + cube.position
    force = calculate_spring_force(vertex_position, spring_constant)
    apply_force(rigid_body, vertex_index, force, dt)

# Main loop
while True:
    #print(cube.velocity)
    # Apply spring force at a specific vertex (e.g., vertex 0)

    for i in range(10):

        apply_spring_force(cube, 0, spring_constant, dt)
        cube.velocity+= np.array([0, -9.8, 0])* dt
        
        cube.velocity+= -0.25 * cube.velocity*dt


        # Angular damping
        cube.angular_momentum +=  -0.25 * cube.angular_momentum * dt

        # Update cube's position and orientation
        cube.position += cube.velocity * dt
        angular_velocity = cube.rotation_matrix @ cube.inertia_tensor_inv @ cube.rotation_matrix.T @ cube.angular_momentum
        cube.rotation_matrix += dt * np.cross(cube.rotation_matrix, angular_velocity)

    # Update cube's mesh visualization
    cube_mesh.points = (cube_points @ cube.rotation_matrix.T) + cube.position
    line.points[1]=cube.rotation_matrix@cube_points[0] + cube.position

    # Visualization update
    plotter.update()

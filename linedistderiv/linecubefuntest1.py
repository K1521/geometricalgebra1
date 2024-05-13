import numpy as np
import pyvista as pv


def distance_to_line(P, line_origin, line_direction):
    # Point P
    P = np.array(P)
    
    # Line origin Q and direction v
    Q = line_origin
    v = line_direction
    
    # Vector PQ
    PQ = P - Q
    
    # Cross product (PQ x v)
    cross_product = np.cross(PQ, v)
    
    # Magnitude of cross product
    cross_product_mag = np.linalg.norm(cross_product)
    
    # Magnitude of line direction vector v
    v_mag = np.linalg.norm(v)
    
    # Distance from point P to the line
    distance = cross_product_mag / v_mag
    
    return distance

def distance_gradient(P, line_origin, line_direction):
    # Point P
    P = np.array(P)
    
    # Line origin Q and direction v
    Q = line_origin
    v = line_direction
    
    # Vector PQ
    PQ = P - Q
    
    # Cross product (PQ x v)
    cross_product = np.cross(PQ, v)
    
    # Magnitude of PQ
    PQ_mag = np.linalg.norm(PQ)
    
    # Magnitude of cross product
    cross_product_mag = np.linalg.norm(cross_product)
    
    # Magnitude of line direction vector v
    v_mag = np.linalg.norm(v)
    
    # Partial derivatives of distance function with respect to x, y, z
    d_dx = (v[1] * PQ[2] - v[2] * PQ[1]) / (PQ_mag * v_mag)
    d_dy = (v[2] * PQ[0] - v[0] * PQ[2]) / (PQ_mag * v_mag)
    d_dz = (v[0] * PQ[1] - v[1] * PQ[0]) / (PQ_mag * v_mag)
    
    return np.array([d_dx,d_dy,d_dz])



#distance = distance_to_line(x, y, z, line_origin, line_direction)
#d_dx, d_dy, d_dz = distance_gradient(x, y, z, line_origin, line_direction)




# Create PyVista plotter
p = pv.Plotter()
#p.background_color = 'black'

p.add_mesh(pv.Box((0, 1,0, 1, 0, 1)),show_edges=True)
# Create initial line
line_origin = np.array([0.0, 0.0, 0.0])
line_direction = np.array([1.0, 0.0, 0.0])
line_direction /= np.linalg.norm(line_direction)  # Normalize direction vector
line_points = line_origin + np.outer(np.linspace(-5, 5, 100), line_direction)
line_mesh = pv.PolyData(line_points)


def calcentryyz():
    topface=np.array([(1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])

    distances = [distance_to_line(P, line_origin, line_direction) for P in topface]
    deriv = [distance_gradient(P, line_origin, line_direction)for P in topface]

    #f(x,y)=a*x+b*x**2+c*y+d*y**2+e
    #f(0,0)=e
    #f(0,1)=a+c+d+e
    #f(1,0)=a+b+e
    #f(1,1)=f(1,1)=a+b+c+d+e
    #df/dx(0,0)=a
    #df/dx(0,1)=
    #df/dx(1,0)=
    #df/dx(1,1)=
    #df/dy(0,0)=c
    #df/dy(0,1)=
    #df/dy(1,0)=
    #df/dy(1,1)=



# Create initial points
points = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), 
                   (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])
points_mesh = pv.PolyData(points)
 





# Add line and points to the plotter
p.add_mesh(line_mesh, color='blue', line_width=3.0)
p.add_mesh(points_mesh, color='red', point_size=10.0)
p.show_grid(bounds=[0, 2, -3, 4, -3, 4])
# Define update function for sliders
def update_line_and_points(param, value):
    global line_origin, line_direction, line_mesh, points_mesh
    if param == 'Origin X':
        line_origin[0] = value
    elif param == 'Origin Y':
        line_origin[1] = value
    elif param == 'Origin Z':
        line_origin[2] = value
    elif param == 'Direction X':
        line_direction[0] = value
        line_direction /= np.linalg.norm(line_direction)  # Normalize direction vector
    elif param == 'Direction Y':
        line_direction[1] = value
        line_direction /= np.linalg.norm(line_direction)  # Normalize direction vector
    elif param == 'Direction Z':
        line_direction[2] = value
        line_direction /= np.linalg.norm(line_direction)  # Normalize direction vector
    
    # Update line points
    line_points = line_origin + np.outer(np.linspace(-5, 5, 100), line_direction)
    line_mesh.points = line_points
    

    # Render the updated plot
    p.render()

# Add slider widgets for line origin and direction

p.add_slider_widget(callback=lambda value: update_line_and_points('Origin X', value), rng=[0,1], value=0.0, title="Origin X", pointa=(0.025, 0.1), pointb=(0.31, 0.1), style='modern')
p.add_slider_widget(callback=lambda value: update_line_and_points('Origin Y', value), rng=[0,1], value=0.0, title="Origin Y", pointa=(0.35, 0.1), pointb=(0.64, 0.1), style='modern')
p.add_slider_widget(callback=lambda value: update_line_and_points('Origin Z', value), rng=[0,1], value=0.0, title="Origin Z", pointa=(0.67, 0.1), pointb=(0.98, 0.1), style='modern')
p.add_slider_widget(callback=lambda value: update_line_and_points('Direction X', value), rng=[-1.0, 1.0], value=1.0, title="Direction X", pointa=(0.025, 0.2), pointb=(0.31, 0.2), style='modern')
p.add_slider_widget(callback=lambda value: update_line_and_points('Direction Y', value), rng=[-1.0, 1.0], value=0.0, title="Direction Y", pointa=(0.35, 0.2), pointb=(0.64, 0.2), style='modern')
p.add_slider_widget(callback=lambda value: update_line_and_points('Direction Z', value), rng=[-1.0, 1.0], value=0.0, title="Direction Z", pointa=(0.67, 0.2), pointb=(0.98, 0.2), style='modern')

# Show the interactive plot
p.show()

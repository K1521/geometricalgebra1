import numpy as np
import scipy


def line_point_distance(point, line_origin, line_direction):
    # Calculate the vector from line origin to the point
    vector_to_point = np.array(point) - np.array(line_origin)
    
    # Project the vector onto the line direction to find the component along the line
    projected_length = np.dot(vector_to_point, line_direction)[:,None]
    print(projected_length)
    
    # Calculate the closest point on the line to the given point
    closest_point_on_line = np.array(line_origin) + projected_length * np.array(line_direction)
    
    # Calculate the vector from the point to the closest point on the line
    vector_to_line = np.array(point) - closest_point_on_line
    
    return vector_to_line



line_origin = np.random.random(3)  # Origin point of the line
line_direction = np.random.random(3)*2-1  # Direction vector of the line
#line_direction=(1,0,0)
line_direction/=np.linalg.norm(line_direction)
#print(line_direction)
# Normalize direction vector (optional, to ensure it's a unit vector)
line_direction = np.array(line_direction) / np.linalg.norm(line_direction)

# List of points to calculate vectors to the line
points = np.array([(0, 0, 0),(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)])
print(line_point_distance(points,line_origin,line_direction))
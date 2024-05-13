import numpy as np

def line_cube_intersection(p0, d, cube_min, cube_max):
    intersections = []
    tmin = -float('inf')
    tmax = float('inf')

    for i in range(3):
        if d[i] != 0:
            # Calculate parameter values for intersections with cube faces
            t1 = (cube_min[i] - p0[i]) / d[i]
            t2 = (cube_max[i] - p0[i]) / d[i]

            # Update tmin and tmax
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
    
    # Check if valid intersection exists within the line segment
    if tmin <= tmax:
        # Calculate intersection points
        intersection_min = p0 + tmin * d
        intersection_max = p0 + tmax * d
        
        # Add intersection points to the list
        intersections.append(intersection_min)
        if not np.isclose(tmin, tmax):
            intersections.append(intersection_max)

    return intersections

# Example usage:
p0 = np.array([1.0, 1.0, 1.0])  # Line starting point
d = np.array([1.0, 2.0, 3.0])   # Direction vector of the line
cube_min = np.array([0.0, 0.0, 0.0])  # Cube minimum coordinates
cube_max = np.array([2.0, 2.0, 2.0])  # Cube maximum coordinates

intersections = line_cube_intersection(p0, d, cube_min, cube_max)
print("Intersection Points:")
for intersection in intersections:
    print(intersection)
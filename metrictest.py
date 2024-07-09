import numpy as np
import matplotlib.pyplot as plt

def metric_distance_weighted_normal_alignment(point, normal, candidate_point):
    direction = candidate_point - point
    distance = np.linalg.norm(direction)
    direction = direction / distance  # Normalize the direction vector
    cos_theta = np.dot(normal, direction)
    return distance * (1 - cos_theta)
def metric_dot_product_normal_direction(point, normal, candidate_point, alpha=0.5):
    direction = candidate_point - point
    distance = np.linalg.norm(direction)
    direction = direction / distance  # Normalize the direction vector
    normal_alignment = np.dot(normal, direction)
    return 0*distance - alpha * normal_alignment
def metric_combined_distance_normal_alignment(point, normal, candidate_point, beta=0.5):
    direction = candidate_point - point
    distance = np.linalg.norm(direction)
    direction = direction / distance  # Normalize the direction vector
    cos_theta = np.dot(normal, direction)
    return np.sqrt(distance**2 + beta * (1 - cos_theta)**2)
# Define a function to compute the metric
def compute_metric(point, normal, candidate_point):
    return metric_dot_product_normal_direction(point, normal, candidate_point,10)
    n=normal/np.linalg.norm(normal)
    d=0
    d+=np.linalg.norm(np.cross(point-candidate_point,n))
    d+=np.abs(np.dot(normal,point-candidate_point)) 
    return d

# Define the grid size and the range for x and y
grid_size = 200
x_range = (-10, 10)
y_range = (-10, 10)

# Create the grid
x = np.linspace(x_range[0], x_range[1], grid_size)
y = np.linspace(y_range[0], y_range[1], grid_size)
xx, yy = np.meshgrid(x, y)

# Initialize the heatmap array
heatmap = np.zeros((grid_size, grid_size))

# Define the fixed point and normal for the metric computation
fixed_point = np.array([0, 0, 0])
fixed_normal = np.array([0, 1, 0])

# Compute the metric for each point in the grid
for i in range(grid_size):
    for j in range(grid_size):
        candidate_point = np.array([xx[i, j], yy[i, j], 0])
        heatmap[i, j] = compute_metric(fixed_point, fixed_normal, candidate_point)

# Plot the heatmap with isolines
plt.figure(figsize=(8, 6))
contour = plt.contour(xx, yy, heatmap, levels=100, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.imshow(heatmap, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower', cmap='viridis', alpha=0.7)
plt.colorbar(label='Metric Value')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Heatmap of Metric Values with Isolines')
plt.show()

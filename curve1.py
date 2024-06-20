import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the implicit functions
def f1(vars, z):
    x, y = vars
    return x**2 + y**2 - 1

def f2(vars, z):
    x, y = vars
    return y**2 + z**2 - 1

def equations(vars, z):
    x, y = vars
    return [f1((x, y), z), f2((x, y), z)]

# Create a range of z values
z_range = np.linspace(-1, 1, 10)

# Store the solution points
points = []

for z in z_range:
    x_range = np.linspace(-1, 1, 10)
    y_range = np.linspace(-1, 1, 10)
    
    for x0 in x_range:
        for y0 in y_range:
            try:
                sol, infodict, ier, mesg = fsolve(equations, (x0, y0), args=(z,), full_output=True)
                if ier == 1:  # Check if the solution converged
                    points.append([sol[0], sol[1], z])
            except Exception as e:
                print(f"Error: {e}")

# Convert the points to a numpy array for easier plotting
points = np.array(points)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b')

# Optionally, you can connect the points to form curves
for i in range(len(points) - 1):
    ax.plot([points[i, 0], points[i + 1, 0]],
            [points[i, 1], points[i + 1, 1]],
            [points[i, 2], points[i + 1, 2]], color='r')

plt.show()

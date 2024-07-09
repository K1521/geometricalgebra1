import numpy as np
import matplotlib.pyplot as plt

# Define the function and its gradient
def fcass(x, y, a, c):
    return (x**2 + y**2)**2 - 2*c**2 * (x**2 - y**2) - (a**4 - c**4)

def gradfcass(x, y, a, c):
    gx = 4*x*(x**2 + y**2) - 4*c**2*x
    gy = 4*y*(x**2 + y**2) + 4*c**2*y
    return gx, gy

# Curve point function using Newton's method
def curve_point(p0, f, gradf, curvept_delta, a, c):
    xi, yi = p0[0], p0[1]
    delta = np.inf
    while delta > curvept_delta:
        gx, gy = gradf(xi, yi, a, c)
        fi = f(xi, yi, a, c)
        cc = gx**2 + gy**2
        if cc > 0:
            t = -fi / cc
        else:
            t = 0
        xi1 = xi + t * gx
        yi1 = yi + t * gy
        delta = np.abs(xi - xi1) + np.abs(yi - yi1)
        xi, yi = xi1, yi1
    return xi, yi

# Function to compute the implicit curve points
def implicit_curvepts(startpt, endpt, start_tangentvt, tangent_stepl, curvept_delta, n2max, f, gradf, a, c):
    p = [np.array(startpt)]
    curve_endpt = np.array(curve_point(endpt, f, gradf, curvept_delta, a, c))
    stepl = tangent_stepl
    delta = stepl
    tv = np.array(start_tangentvt)
    fac = stepl / np.linalg.norm(tv)
    ps = p[0] + fac * tv
    
    p.append(np.array(curve_point(ps, f, gradf, curvept_delta, a, c)))
    i = 1
    test = 0
    
    while delta > 0.7 * stepl and i < n2max:
        dv = p[i] - p[i-1]
        gx, gy = gradf(p[i][0], p[i][1], a, c)
        tv = np.array([-gy, gx])
        cc = np.linalg.norm(tv)
        if cc > 0:
            fac = stepl / cc
            test = np.dot(tv, dv)
            if test < 0:
                fac = -fac
            ps = p[i] + fac * tv
        else:
            ps = 2 * p[i] - p[i-1]
        
        i += 1
        p.append(np.array(curve_point(ps, f, gradf, curvept_delta, a, c)))
        delta = np.linalg.norm(p[i] - curve_endpt)
    
    if i < n2max:
        p.append(curve_endpt)
        n2 = i + 1
    else:
        n2 = i
    
    return p[:n2]

# Parameters
a = 1.0
c = 1.0
startpt = (1.0, 1.0)
endpt = startpt
start_tangentvt = (0.0, 1.0)
tangent_stepl = 0.05
curvept_delta = 1e-6
n2max = 290

# Compute curve points
curve_points = implicit_curvepts(startpt, endpt, start_tangentvt, tangent_stepl, curvept_delta, n2max, fcass, gradfcass, a, c)

# Plotting
plt.figure(figsize=(8, 6))
plt.title('Implicit Curve using Steepest Descent Method')
plt.plot([pt[0] for pt in curve_points], [pt[1] for pt in curve_points], marker='o', linestyle='-', color='b')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
plt.show()

import sympy as sy

# Define symbols for variables
ax, ay, az = sy.symbols('ax ay az')  # Coordinates of point A
px, py, pz = sy.symbols('px py pz')  # Coordinates of point P
dx, dy, dz = sy.symbols('dx dy dz')  # Components of direction vector d

# Define vectors A, P, and d
A = sy.Matrix([ax, ay, az])
P = sy.Matrix([px, py, pz])
d = sy.Matrix([dx, dy, dz])

# Compute vector AP = A - P
AP = A - P

# Compute cross product (AP x d)
cross_product = AP.cross(d)

# Compute magnitude of the cross product |AP x d|
cross_product_magnitude = sy.sqrt(sum([component**2 for component in cross_product]))

# Compute magnitude of vector d |d|
d_magnitude = sy.sqrt(sum([component**2 for component in d]))

# Compute distance D
D = cross_product_magnitude / d_magnitude

# Compute partial derivatives of D with respect to px, py, pz
dD_dpx = sy.simplify(D.diff(px))
dD_dpy = sy.simplify(D.diff(py))
dD_dpz = sy.simplify(D.diff(pz))

# Print results
print("Distance D:")
print(D)
print("\nPartial derivative of D with respect to px:")
print(dD_dpx)
print("\nPartial derivative of D with respect to py:")
print(dD_dpy)
print("\nPartial derivative of D with respect to pz:")
print(dD_dpz)
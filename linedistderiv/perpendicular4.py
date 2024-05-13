import numpy as np
import pyvista as pv

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

a=np.array([.2,.3,-.1])
d=np.array([1,2,3])
#d=d/np.linalg.norm(d)

def f(p):
    apd=np.cross(a-p,d,axis=-1)
    
    fun=np.linalg.norm(apd,axis=-1)/np.linalg.norm(d,axis=-1)
    return fun
    deriv=np.cross(apd,d)/(np.linalg.norm(apd)*np.linalg.norm(d))
    #return deriv
    #delta*(voltf.f/delta_norm_squared_safe)
    
    return deriv*(fun/np.linalg.norm(deriv)**2)#gauss newton step
    
def f_(p):
    apd=np.cross(a-p,d,axis=-1)
    
    deriv=np.cross(apd,d,axis=-1)/np.expand_dims((np.linalg.norm(apd,axis=-1)*np.linalg.norm(d,axis=-1)),-1)
    return deriv
    #delta*(voltf.f/delta_norm_squared_safe)
    fun=np.linalg.norm(apd)/np.linalg.norm(d)
    return deriv*(fun/np.linalg.norm(deriv)**2)#gauss newton step    
def gaussnewton(p):
    deriv=f_(p)
    return deriv*np.expand_dims(f(p)/np.linalg.norm(deriv,axis=-1)**2,-1)

# Define grid dimensions
n=2
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
z = np.linspace(-1, 1, n)

# Create meshgrid
X,Y,Z = np.meshgrid(x, y, z, indexing='ij')
points=np.stack([X,Y,Z],axis=-1)
#points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

vectors = gaussnewton(points)#np.apply_along_axis(gaussnewton, -1,points)
grid = pv.StructuredGrid(X, Y, Z)

p1,p2=line_cube_intersection(a,d,[-1,-1,-1],[1,1,1])
pdata = pv.PolyData(np.array([p1,p2]))
pdata.lines = np.array([[2,0,1]])

squareidx=([0,0,1,1],[0,1,0,1],0)#xy
festeaxis=2


squarep=points[squareidx]
#squarev = np.apply_along_axis(gaussnewton, -1,squarep)
#ebene:(x-p)*n=0 -> (x*n=p*n)
p=squarep-gaussnewton(squarep)
n=f_(squarep)
#n=n/(np.linalg.norm(n,axis=-1)**2)[:,None]#small derivative-> higher weight. TODO smaller f-> higher weight
n=n/f(squarep)[:,None]
pn=np.einsum('ij,ij->i', n, p)#dot along axis
#pn=pn/f(squarep)

squareoffset=squarep[0,festeaxis]
A=np.delete(n, festeaxis, -1)
b=pn-squareoffset * n[:,festeaxis]

x,*_=np.linalg.lstsq(A,b)
xfull=np.insert(x, festeaxis, squareoffset, axis=-1)

print(squareoffset,x)
#print(p)
#print(n)
#print(pn)
#print(square)
#print(points)
#exit()
print(np.linalg.lstsq(A,b))
print(sum((b-A@x)**2))
plotter = pv.Plotter()
plotter.add_mesh(pdata)
plotter.add_mesh(pv.PolyData([xfull]), color='red', point_size=10)
plotter.add_mesh(grid, color='white', opacity=0.5)  # Add grid as a background
plotter.add_arrows(points.reshape(-1,3), -vectors.reshape(-1,3), mag=1)  # Add arrows for vector field
plotter.show()


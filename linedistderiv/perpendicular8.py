import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

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
def transform(M,P):
    homogenpoints=np.insert(P, 3, 1, axis=-1)
    transformed=homogenpoints@M.T
    return transformed[:,:3]/transformed[:,3][:,None]
class implicitfun:
    def __init__(self):
        self.setrt(np.eye(4))#set rotation/translation
    def setrt(self,rt):
        self.rt=rt
        self.rtinv=np.linalg.inv(rt)
    def f(self,points):
        pass
    def f_(self,points):
        pass
    def apply(self,points):
        prevshape=points.shape
        points=points.reshape(-1,3)
        #print(points)
        pointst=transform(self.rt,points)
        val=self.f(pointst)
        #print(self.f_(points).T)
        vect=self.f_(pointst)
        gnt=vect*np.expand_dims(val/np.linalg.norm(vect,axis=-1)**2,-1)
        zero=transform(self.rtinv,np.zeros((1,3)))
        gn=transform(self.rtinv,gnt)-zero
        vec=transform(self.rtinv,vect)-zero
        return val.reshape(prevshape[:-1]),vec.reshape(prevshape),gn.reshape(prevshape)
        #return f(x),f'(x),gaussnewton(x)
    #def gaussnewton(p):
    #    deriv=f_(p)
    #    return deriv*np.expand_dims(f(p)/np.linalg.norm(deriv,axis=-1)**2,-1)

class implicitfuntorus(implicitfun):
    def __init__(self,r,R):
        super().__init__()
        self.r=r
        self.R=R
    def f(self,p):
        x2,y2,z2=p.T**2
        return(self.R**2-self.r**2+x2+y2+z2)**2-4*self.R**2*(x2+y2)
    def f_(self,p):
        x2,y2,z2=p.T**2
        x,y,z=p.T
        t1=(self.R**2-self.r**2+x2+y2+z2)
        dx=4*t1*x-8*self.R**2*x
        dy=4*t1*y-8*self.R**2*y
        dz=4*t1*z
        return np.vstack([dx,dy,dz]).T
        
t=implicitfuntorus(0.05,3)
#t=implicitfuntorus(0.2,3)
rot=np.eye(4)
rot[:3,3]=[3.1,0,0.1]
#rot[:3,3]=[3.4,0,0.6]
rot[:3,:]*=3
rot[:3,:3]=rot[:3,:3]@Rotation.from_rotvec([1,2,3]).as_matrix()
print(rot)
t.setrt(np.linalg.inv(rot))


def f(p):
    return t.apply(p)[0]
    
def f_(p):
    return t.apply(p)[1]
def gaussnewton(p):
    return t.apply(p)[2]

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
gridc = pv.StructuredGrid(X, Y, Z)

p1,p2=line_cube_intersection(a,d,[-1,-1,-1],[1,1,1])
pdata = pv.PolyData(np.array([p1,p2]))
pdata.lines = np.array([[2,0,1]])

squareidx=([0,0,1,1],[0,1,0,1],0)#xy
festeaxis=2

def evalsquare(festeaxis,offset):
    squareidx=[0,np.array([0,0,1,1]),np.array([0,1,0,1])]#index for square
    squareidx=squareidx[-festeaxis:]+squareidx[:-festeaxis]#index for square alligned with axis
    #print(squareidx)
    for i in range(3):#move the square
        squareidx[i]+=offset[i]
    #print(offset_indices)
    squarep=points[tuple(squareidx)]
    #squarev = np.apply_along_axis(gaussnewton, -1,squarep)
    #ebene:(x-p)*n=0 -> (x*n=p*n)
    pi=squarep
    ni=gaussnewton(squarep)
    nmi=ni.copy()
    nmi[:,festeaxis]=0
    n=4
    
    p_=np.sum(pi,axis=0)/n
    nm_=np.sum(nmi,axis=0)/n
    A=np.sum((pi-p_)**2)
    B=np.sum((pi-p_)*(ni-nm_))
    C=np.sum((ni-nm_)*(ni-nm_))
    B2=np.sum((pi-p_)*(nmi-nm_))
    C2=np.sum((nmi-nm_)*(nmi-nm_))

    a=-B/C
    avgpoint=p_+a*nm_
    e=A-B**2/C
    e2=A-B2**2/C2
    #print(avgpoint)
    x=np.delete(avgpoint,festeaxis, axis=-1)
    #xfull=np.insert(x, festeaxis, squareoffset, axis=-1)#[:3]
    xfull=avgpoint
    #print(xfull)
    return xfull,e2,x[:2],a

#print(squareoffset,x)
#print(p)
#print(n)
#print(pn)
#print(square)
#print(points)
#exit()
#print(np.linalg.lstsq(A,b))
#print(sum((b-A@x)**2))
predictedpoints=[]
plotter = pv.Plotter()

for i,o in [(0,[0,0,0]),(0,[1,0,0]),(1,[0,0,0]),(1,[0,1,0]),(2,[0,0,0]),(2,[0,0,1])]:
    xfull,e,x,a=evalsquare(i,o)
    pointonside=np.all((-1<=x) & (x<=1))
    predictedpoints.append(xfull)
    print(xfull,e,a,pointonside)

    #r=(30+np.log10(e))/30/5
    r=e/30
    #print(r,e,a)
    sphere = pv.Sphere(radius=r, center=xfull)
    plotter.add_mesh(sphere, color='red', opacity=0.5)




x, y, z = 5*np.mgrid[-1:1:61j, -1:1:61j, -1:1:61j]
grid = pv.StructuredGrid(x, y, z)
grid["vol"] = f(grid.points).flatten()
print(np.min(grid["vol"]))
contours = grid.contour([0])
plotter.add_mesh(contours,scalars=contours.points[:, 2], show_scalar_bar=False,opacity=0.5)

plotter.add_mesh(pdata)
#plotter.add_mesh(pv.PolyData(predictedpoints), color='red', point_size=10)
plotter.add_mesh(gridc, color='white', opacity=0.5)  # Add grid as a background
plotter.add_arrows(points.reshape(-1,3), -vectors.reshape(-1,3), mag=1)  # Add arrows for vector field
plotter.add_axes(interactive=True, line_width=2, color='red', x_color='red', y_color='green', z_color='blue')
plotter.show()






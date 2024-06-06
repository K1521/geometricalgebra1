import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation,Slerp
from collections import defaultdict
pv.global_theme.allow_empty_mesh = True

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
def f(p):
    return t.apply(p)[0]
def f_(p):
    return t.apply(p)[1]
def gaussnewton(p):
    return t.apply(p)[2]


#t=implicitfuntorus(0.2,3)
rot=np.eye(4)
rot[:3,3]=[3,0,0]
#rot[:3,3]=[3.4,0,0.6]
rot[:3,:]*=3
rot=np.linalg.inv(rot)

t.setrt(rot)

def evalsquare(festeaxis,offset):
    squareidx=[0,np.array([0,0,1,1]),np.array([0,1,0,1])]#index for square
    squareidx=squareidx[-festeaxis:]+squareidx[:-festeaxis]#index for square alligned with axis
    #print(squareidx)
    for i in range(3):#move the square
        squareidx[i]+=offset[i]
    #print(offset_indices)
    squarep=points[tuple(squareidx)]
    squareoffset=squarep[0,festeaxis]
    n=gaussnewton(squarep)
    U, s, Vh = np.linalg.svd(n)
    d=Vh[2,:]
    #d=Vh[:,2]
    #print(Vh)

    #print(n,d)
    C=np.cross(d[None], n)
    #print(C)
    #print(np.array([np.cross(d, ni)for ni in n]))
    C/=np.linalg.norm(C,axis=1)[:,None]
    #print(np.linalg.norm(C,axis=0))
    rhs=np.einsum("ij,ij->i",C,squarep)#[:,None]
    print(rhs)

    A=np.delete(C, festeaxis, -1)
    b=rhs-squareoffset * C[:,festeaxis]
    x,*_=np.linalg.lstsq(A,b)
    xfull=np.insert(x, festeaxis, squareoffset, axis=-1)#[:3]
    xfull=xfull[:3]
    #print(xfull)
    return xfull,sum((b-A@x)**2),x[:2],d,np.mean(squarep,0)

x, y, z = 5*np.mgrid[-1:1:61j, -1:1:61j, -1:1:61j]
grid = pv.StructuredGrid(x, y, z)
grid["vol"] = f(grid.points).flatten()
grid= grid.contour([0])


plotter = pv.Plotter()

n=2
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
z = np.linspace(-1, 1, n)

# Create meshgrid
X,Y,Z = np.meshgrid(x, y, z, indexing='ij')
points=np.stack([X,Y,Z],axis=-1)
#points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


gridc = pv.StructuredGrid(X, Y, Z)
plotter.add_mesh(gridc, color='white', opacity=0.5)  # Add grid as a background
plotter.add_axes(interactive=True, line_width=2, color='red', x_color='red', y_color='green', z_color='blue')
plotter.show(interactive_update=True)


def newmesh():
    d=pv.PolyData()
    #plotter.add_mesh(d)
    return d
plottedmeshes=defaultdict(newmesh)
keys=['sphere0[0, 0, 0]', 'arrow0[0, 0, 0]', 'sphere0[1, 0, 0]', 'arrow0[1, 0, 0]', 'sphere1[0, 0, 0]', 'arrow1[0, 0, 0]', 'sphere1[0, 1, 0]', 'arrow1[0, 1, 0]', 'sphere2[0, 0, 0]', 'arrow2[0, 0, 0]', 'sphere2[0, 0, 1]', 'arrow2[0, 0, 1]', 'd', 'a0', 'a1', 'a2', 
'a3', 'a4', 'a5', 'a6', 'a7']
for key in keys:
    if "sphere" in key:
        plotter.add_mesh(plottedmeshes[key],color='red')
    elif key=="d":
        plotter.add_mesh(plottedmeshes[key],color='blue',opacity=0.5)
    else:
        pass
#############################################################
rotlerp=Slerp(np.linspace(0,99,100),Rotation.from_rotvec(np.random.random((100, 3))*3))
for R in rotlerp(np.linspace(0,99,1000)):
    transform_matrix =np.eye(4)
    transform_matrix[:3,:3]=R.as_matrix()#Rotation.from_rotvec([0.5,0,0]).as_matrix()
    transform_matrix[:3,3]=[0,0,0.6]
    addedactors=[]

    donughnout=grid.copy().transform(transform_matrix)
    t.setrt(rot@np.linalg.inv(transform_matrix))






    squareidx=([0,0,1,1],[0,1,0,1],0)#xy
    festeaxis=2
    predictedpoints=[]
    for i,o in [(0,[0,0,0]),(0,[1,0,0]),(1,[0,0,0]),(1,[0,1,0]),(2,[0,0,0]),(2,[0,0,1])]:
        xfull,e,x,d,p=evalsquare(i,o)
        pointonside=np.all((-1<=x) & (x<=1))
        predictedpoints.append(xfull)
        print(xfull,e,pointonside)

        r=(30+np.log10(e))/5
        #r=e/30
        r=0.1
        #print(r,e,a)
        
        if np.all((-5<=x) & (x<=5)):
            sphere = pv.Sphere(radius=r, center=xfull)
            plottedmeshes[f"sphere{i}{o}"].copy_from(sphere)
        plottedmeshes[f"arrow{i}{o}"].copy_from(pv.Arrow(p.reshape(3), d.reshape(3)))

    plottedmeshes[f"d"].copy_from(donughnout)

    vectors = gaussnewton(points)
    pointsr,vectorsr=points.reshape(-1,3), -vectors.reshape(-1,3)
    for i in range(8):
        plottedmeshes[f"a{i}"].copy_from(pv.Arrow(pointsr[i], vectorsr[i]))

    plotter.update(0.5)
    print(plottedmeshes.keys())






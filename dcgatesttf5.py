
import tensorflow as tf
from blademul5 import *
import numpy as np
import pyvista as pv
import time
import itertools
dcga=algebra(8,2)

#e1=1,e2=1,e3=1,e4=1,e5=-1,e6=1,e7=1,e8=1,e9=1,e10=-1
#skalar,e1,e2,e3,e4,e6,e7,e8,e9,e5,e10,*_=dcga.allblades()
#multivec=sortgeo(dcga)
multivec=sortgeotf(dcga,[])
e1,e2,e3,e4,e6,e7,e8,e9,e5,e10=multivec.monoblades()

dcga.bladenames="1,2,3,4,6,7,8,9,5,10".split(",")
print(e1,e2,e3,e4,e6,e7,e8,e9,e5,e10)
eo1=0.5*e5-0.5*e4
eo2=0.5*e10-0.5*e9
ei1=e4+e5
ei2=e9+e10


def point1(x,y,z):
    return e1*x+e2*y+e3*z+ei1*(.5*(x*x+y*y+z*z))+eo1
def point2(x,y,z):
    return e6*x+e7*y+e8*z+ei2*(.5*(x*x+y*y+z*z))+eo2

def point(x,y,z):
    return point1(x,y,z).outer(point2(x,y,z))

eo=eo1.outer(eo2)
ei=ei1.outer(ei2)
Txx=e6.outer(e1)
Tyy=e7.outer(e2)

def CGA1_Plane(x,y,z,h):
    vec=(x*e1+y*e2+z*e3)
    return vec*(1/np.sqrt(vec.inner(vec).toscalar()))+h*ei1
def CGA2_Plane(x,y,z,h):
    vec=(x*e6+y*e7+z*e8)
    return vec*(1/np.sqrt(vec.inner(vec).toscalar()))+h*ei2
#CGA2_Plane = { Normalize(_P(1)*e6 + _P(2)*e7 + _P(3)*e8) + _P(4)*ei2 }
def Plane(x,y,z,h):
    return CGA1_Plane(x,y,z,h)^CGA2_Plane(x,y,z,h)
#Plane = {
# CGA1_Plane(_P(1),_P(2),_P(3),_P(4))^CGA2_Plane(_P(1),_P(2),_P(3),_P(4))
#}

T1=-ei
Tt2=eo2.outer(ei1)+ei2.outer(eo1)
Tt4=-4*eo

def toroid(R,r):
    dSq=R*R-r*r
    return Tt4+2*Tt2*dSq+T1*dSq*dSq-4*R*R*(Txx+Tyy)




t=toroid(1,0.5)
p=Plane(2,0,0,0)

for x in np.mgrid[-1:1:5j]:
    print(p.inner(point(x,0,0)))


tovisualise=[t]
usecuda=False

pv.set_plot_theme('dark')
p = pv.Plotter()
p.add_axes()
p.show_grid()

t0=time.time() 
step=30
#x, y, z = 2*np.mgrid[-1:1:100j, -1:1:100j, -1:1:100j]
x, y, z = 2*np.mgrid[-1:1:step*1J, -1:1:step*1J, -1:1:step*1J]
grid = pv.StructuredGrid(x, y, z)
polydatas=[]
if usecuda:
    import cupy as cp
    points=point(cp.array(x.flatten()),cp.array(y.flatten()),cp.array(z.flatten()))
    for shape in tovisualise:
        grid["vol"]=cp.asnumpy(points.inner(shape).toscalar())
        polydatas.append(grid.contour([0]))
elif False:
    points=point(x.flatten(),y.flatten(),z.flatten())
    for shape in tovisualise:
        grid["vol"]=points.inner(shape).toscalar()
        print(grid["vol"])
        polydatas.append(grid.contour([0]))
else:
    for shape in tovisualise:

        xtf,ytf,ztf=[tf.Variable(i) for i in (x,y,z)]
        with tf.GradientTape(persistent=True) as tape:
            points=point(xtf,ytf,ztf)
            vol=points.inner(shape).toscalar()
            #voltf=vol**2#tf.abs(vol)
            voltf=vol


        #arr=np.stack([tape.gradient(vol, i) for i in (xtf,y,z)],axis=0)
        #print(arr)
        #print()
        #print(tape.gradient(vol, xtf))
        #parity=(tape.gradient(vol, xtf).numpy()>=0) ^ (tape.gradient(vol, y).numpy()>=0) ^ (tape.gradient(vol, z).numpy()>=0)
        #parity=(tape.gradient(vol, xtf).numpy()>=0) 
        #vol*=(-1)**parity
        #del tape
        #vol*=(-1)**(arr[np.abs(arr).argmax(axis=0),range(arr.shape[1])]>1)
        #vol=(tape.gradient(vol, xtf).numpy()**2+ tape.gradient(vol, y).numpy()**2+tape.gradient(vol, z).numpy()**2)-0.1
        #vol=vol-0.1
        delta=np.stack([tape.gradient(voltf, xtf).numpy(),tape.gradient(voltf, ytf).numpy(),tape.gradient(voltf, ztf).numpy()],axis=3)
        del tape
        #from sklearn.preprocessing import normalize
        def normalized(a, axis=-1, order=2):
            l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
            l2[l2==0] = 1
            return a / np.expand_dims(l2, axis)
        delta=normalized(delta)
        #print(delta)
        
        

        #import matplotlib.pyplot as plt
        #plt.hist(np.array([np.sum(delta[1:,:,:]*delta[:-1,:,:],axis=3).ravel(),np.sum(delta[:,1:,:]*delta[:,:-1,:],axis=3).ravel(),np.sum(delta[:,:,1:]*delta[:,:,:-1],axis=3).ravel()]).flatten(),100)
        #plt.hist(np.array([np.sum(delta[1:,:,:]*delta[:-1,:,:],axis=3).ravel(),np.sum(delta[:,1:,:]*delta[:,:-1,:],axis=3).ravel(),np.sum(delta[:,:,1:]*delta[:,:,:-1],axis=3).ravel()]).T,100)
        #plt.title("dotprod")
        #plt.xlabel("Wert")
        #plt.ylabel("Häufigkeit")
        #plt.show()

        for i in range(step-1):
            for j in range(step-1):
                for k in range(step-1):
                    pass
        
        
        #subprod=np.sum(delta[1:,:,:]*delta[:-1,:,:],axis=3)
        #polydatas.append(pv.PolyData((np.stack([x,y,z], axis=-1)[1:,:,:][np.where((subprod>=0)&(subprod<=0.75)&(vol[1:,:,:]>0.1))])))
        #subprod=np.sum(delta[:,1:,:]*delta[:,:-1,:],axis=3)
        #polydatas.append(pv.PolyData((np.stack([x,y,z], axis=-1)[:,1:,:][np.where((subprod>=0)&(subprod<=0.75)&(vol[:,1:,:]>0.1))])))
        #subprod=np.sum(delta[:,:,1:]*delta[:,:,:-1],axis=3)
        #polydatas.append(pv.PolyData((np.stack([x,y,z], axis=-1)[:,:,1:][np.where((subprod>=0)&(subprod<=0.75)&(vol[:,:,1:]>0.1))])))

        
        #subprod=np.sum(delta[1:,:,:]*delta[:-1,:,:],axis=3)
        #polydatas.append(pv.PolyData((np.stack([x,y,z], axis=-1)[1:,:,:][np.where((subprod>=0)&(subprod<=0.75))])))
        #subprod=np.sum(delta[:,1:,:]*delta[:,:-1,:],axis=3)
        #polydatas.append(pv.PolyData((np.stack([x,y,z], axis=-1)[:,1:,:][np.where((subprod>=0)&(subprod<=0.75))])))
        #subprod=np.sum(delta[:,:,1:]*delta[:,:,:-1],axis=3)
        #polydatas.append(pv.PolyData((np.stack([x,y,z], axis=-1)[:,:,1:][np.where((subprod>=0)&(subprod<=0.75))])))

        #subprod=np.sum(delta[1:,:,:]*delta[:-1,:,:],axis=3)
        #polydatas.append(pv.PolyData((np.stack([x,y,z], axis=-1)[1:,:,:][np.where((subprod<0))])))
        #subprod=np.sum(delta[:,1:,:]*delta[:,:-1,:],axis=3)
        #polydatas.append(pv.PolyData((np.stack([x,y,z], axis=-1)[:,1:,:][np.where((subprod<0))])))
        #subprod=np.sum(delta[:,:,1:]*delta[:,:,:-1],axis=3)
        #polydatas.append(pv.PolyData((np.stack([x,y,z], axis=-1)[:,:,1:][np.where((subprod<0))])))



        #subprod=np.sum(delta[1:,:,:]*delta[:-1,:,:],axis=3)
        #polydatas.append(pv.PolyData((np.stack([x,y,z], axis=-1)[1:,:,:][np.where((subprod<0))])))
        #subprod=np.sum(delta[:,1:,:]*delta[:,:-1,:],axis=3)
        #polydatas.append(pv.PolyData((np.stack([x,y,z], axis=-1)[:,1:,:][np.where((subprod<0))])))
        
        """subprod=np.sum(delta[:,:,1:]*delta[:,:,:-1],axis=3)
        pointsgrid=np.stack([x,y,z], axis=-1)
        pointsgrids=(pointsgrid[:,:,1:]+pointsgrid[:,:,:-1])/2
        #print(pointsgrids.shape)

        #print(pointsgrids[:,:,:,[0,1,2]].shape)
        dotgrid=pv.StructuredGrid(pointsgrids[:,:,:,0],pointsgrids[:,:,:,1],pointsgrids[:,:,:,2])
        dotgrid["vol"]=subprod.ravel()
        
        p.add_mesh(dotgrid.threshold((-2,0.5)),opacity=0.5)"""




        #p.add_mesh(pv.PolyData(pointsgrid[:,:,1:]).reshape((-1,3)).threshold(value=(-1.1,1.1),preference="point"), scalars=subprod.ravel(), show_scalar_bar=False)

        #print((np.stack([x,y,z], axis=-1)[:,:,1:]).reshape((-1,3)))

        
        grid["vol"]=vol.numpy().flatten()
        #polydatas.append(grid.contour([0]))
        cont=grid.contour([0])
        

        #vectors=delta.reshape((-1,3))
        grid["vectors"] =delta.reshape((-1,3)) #vectors.copy()
        grid.set_active_vectors("vectors")
        #glyph=grid.glyph(factor=0.2,scale=False)
        #p.add_mesh(glyph)
        #print(glyph)




        #streamlines=grid.streamlines(max_steps=10000)
        #p.add_mesh(streamlines)
        p.add_mesh(cont, show_scalar_bar=True,opacity=0.4,)
        #delta2 = delta.swapaxes(0,3)

        
        #glyphs=[(i,grid.glyph(factor=0.02,scale=(x==x[i,0,0]).ravel().astype(float)))for i in range(step)]
        
        #glyph=glyphs[0][1].copy()
        
        
        scalarsmask,_,_=1.*np.mgrid[:step,:step,:step]


        glyphs=[]
        for i in range(step):
            grid[f"s{i}"]=(scalarsmask==i).ravel().astype(float)
            grid.set_active_scalars(f"s{i}")
            glyphs.append(grid.glyph(factor=0.2,scale=True) )

        actors=[p.add_mesh(g,show_scalar_bar=False) for g in glyphs]
        for a in actors:
            a.visibility = False

        #print(scalarsmask.shape,x.shape)
        #p#rint(glyph)
        #print(grid)
        #glyph.scale((scalarsmask==0)*1.)

        alast=actors[-1]
        def slideraction(v):
            global alast
            a=actors[int(v)]
            if a!=alast:
                a.visibility = True
                alast.visibility = False
            alast=a
        p.add_slider_widget(slideraction, [0, step-1], title='Resolution')
        #p.show(interactive_update=True)

        p.show()

        #alast=actors[-1]
        #while 1:
            #a=actors[int(input())]
            #for a in actors:
                #grid["vectors"]=np.where((x==i),delta2,0).reshape((-1,3))
            #a.visibility = True
            #alast.visibility = False
            #alast=a
            #p.update()
            #print(i)
            #time.sleep(0.2)
                
        """
        #polydatas.append(grid.contour([0]))
        grid["vol"]=vol.numpy().flatten()
        #print(grid["vol"])


        surfaces = [(v,grid.contour([v])) for v in 0.55+0.03*np.mgrid[-1:1:100J]]
        #surfaces=[(v,s.smooth(100)) for v,s in surfaces if s.n_points>0]
        surfaces=[(v,s.smooth(100)) for v,s in surfaces if s.n_points>0]
        surface = surfaces[-1][1].copy()

        p.add_mesh(
            surface,
            opacity=0.9,
            clim=grid.get_data_range(),
            show_scalar_bar=False,
            diffuse=0.5, specular=0.5, ambient=0.5
        )
        p.show(interactive_update=True)
        for v,surf in itertools.cycle(itertools.chain(surfaces,reversed(surfaces))):
            surface.copy_from(surf)
            p.update()
            print(v)

        """
        #for c in polydatas:
        #p.add_mesh(c, scalars=c.points[:, 2], show_scalar_bar=False)
        #p.show()


        #polydatas.append(grid.contour([-0.1,0,0.1]))
print(time.time()-t0)
for c in polydatas:
    if c.n_points:
        p.add_mesh(c, scalars=c.points[:, 2], show_scalar_bar=False)
    else:
        print("empty")
p.show()
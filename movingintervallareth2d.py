from intervallarethmetic1 import intervallareth

import sympy
import numpy as np
x,ix,y,iy,z,iz=sympy.symbols("x ix y iy z iz")


xstep=10
ystep=10

#f=x**2+2*x*y+2*y**2-x+2*x*y*x
f=x*x+y*y-0.0001*x**4-0.0001*y**4
fa=sympy.Poly(f.subs(x,(ix+1)*0.5*xstep+x).subs(y,(iy+1)*0.5*ystep+y),ix,iy)
#fa=sympy.Poly(f.subs(x,ix*xstep+x).subs(y,iy*ystep+y),ix,iy)#für [0,1] intervall
for i in range(3,10):
    if i%2==0:
        fa=fa.subs(ix**i,ix**2).subs(iy**i,iy**2)
    else:
        fa=fa.subs(ix**i,ix).subs(iy**i,iy)
fa=fa.collect([ix,iy])
fa=fa.as_expr()
fa=str(fa)#.replace("x","X").replace("y","Y")
f=str(f)
print(fa)

#import matplotlib.pyplot as plt

#xipn=(-1,1)
#xizp=(0,1)
#xinz=(-1,0)
#xizp=-xinz
#-xizp=xinz
#xipn=-xipn
#xipn**2=xizp




#fig = plt.figure()
#ax = plt.axes(projection='3d')
X, Y = np.mgrid[-100:100:xstep,-100:100:ystep]

ix=intervallareth(-1,1)
iy=intervallareth(-1,1)
#ix=intervallareth(0,1)
#iy=intervallareth(0,1)

Z=eval(str(f).replace("x","X").replace("y","Y"))
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0,zorder =0)

import pyvista as pv
plotter = pv.Plotter()

rect=pv.Rectangle([[xstep,0,0],[xstep,ystep,0],[0,ystep,0]])
vectors=[]
vectors2=[]
for i in range(X.shape[0]-1):
    for j in range(X.shape[1]-1):
        
        #x=intervallareth(X[i,j],X[i+1,j])
        #y=intervallareth(Y[i,j],Y[i,j+1])
        #h=eval(f)
        x=X[i,j]
        y=Y[i,j]
        h=eval(fa)
        vectors.append([x,y,h.max])
        vectors2.append([x,y,h.min])

        #ax.plot_surface(X[i:i+2,j:j+2], Y[i:i+2,j:j+2], flat*h,zorder =1)

plotter.add_mesh(pv.PolyData(np.array(vectors)).glyph(False,False,geom=rect),color='r',opacity=0.9)
plotter.add_mesh(pv.PolyData(np.array(vectors2)).glyph(False,False,geom=rect),color='b',opacity=0.9)
# Create and plot structured grid
grid = pv.StructuredGrid(X,Y,Z)

plotter.add_mesh(grid, scalars=grid.points[:, -1], show_edges=True,
                 scalar_bar_args={'vertical': True},opacity=1)
plotter.set_scale(xscale=1, yscale=X.ptp()/Y.ptp(), zscale=X.ptp()/Z.ptp())
plotter.show_grid()
plotter.show()





#print(X)



#plt.show()
from algebra.dcga import *
import pyvista as pv
import time
import functools
import operator
import numpy as np
#import tensorflow as tf
import matplotlib.pyplot as mplt

t=toroid(1,.5)
p=Plane(0.1,0.1,0.1,0.5)

vis=t#^Plane(0.1,0.1,0.001,0.5)#^Plane(0.001,0.001,0.1,0.1)






#pyvista setup
pv.set_plot_theme('dark')
plt = pv.Plotter()
plt.add_axes()
plt.show_grid()

t0=time.time() 
step=101



x,y,z=multivec.scalar(sympy.symbols('x y z'))
expr=point(x,y,z).inner(vis)
equations=[sympy.simplify(blade.magnitude*2**10) for blade in expr.lst]
equationsnew=[]
for e in equations:
    if e.is_number:
        #print(sympy.factor(e,deep=True))
        continue
    #equationsnew.extend([f for f,p in sympy.factor_list(e)[1]])
    equationsnew.append(sympy.Mul(*[f for f,p in sympy.factor_list(e)[1]]))
    #print(sympy.factor(e,deep=True).args)
    #equationsnew.append(sympy.factor(e))
    #equationsnew.append(sympy.factor(e,deep=True))
equations=equationsnew

#equations="""
#(64*x**4 + 128*x**2*y**2 + 128*x**2*z**2 + 96*x**2 + 64*y**4 + 128*y**2*z**2 - 160*y**2 + 64*z**4 + 96*z**2 + 35)
#y*(16*x - 1)
#(4*x**4 - 16*x**3 + 8*x**2*y**2 + 8*x**2*z**2 + 7*x**2 - 16*x*y**2 - 16*x*z**2 - 13*x + 4*y**4 + 8*y**2*z**2 - 9*y**2 + 4*z**4 + 7*z**2 + 3)
#(16*x - 1)
#(x**4 - 8*x**3 + 2*x**2*y**2 + 2*x**2*z**2 + 14*x**2 - 8*x*y**2 - 8*x*z**2 - 8*x + y**4 + 2*y**2*z**2 - 2*y**2 + z**4 + 2*z**2 + 1)
#(4*x**4 + 112*x**3 + 8*x**2*y**2 + 8*x**2*z**2 - x**2 + 112*x*y**2 + 112*x*z**2 + 83*x + 4*y**4 + 8*y**2*z**2 - 17*y**2 + 4*z**4 - z**2 - 3)
#(x**4 + 24*x**3 + 2*x**2*y**2 + 2*x**2*z**2 - 116*x**2 + 24*x*y**2 + 24*x*z**2 + 32*x + y**4 + 2*y**2*z**2 - 4*y**2 + z**4 - 1)
#(x**4 + 56*x**3 + 2*x**2*y**2 + 2*x**2*z**2 + 778*x**2 + 56*x*y**2 + 56*x*z**2 - 56*x + y**4 + 2*y**2*z**2 - 6*y**2 + z**4 - 2*z**2 + 1)
#""".split("\n")
#equations.remove("")
#equations.remove("")
#equations=[sympy.parse_expr(e) for e in equations]

print(equations)


equations=list({sympy.sstr(e,full_prec=True):e for e in equations}.values())
print("------------------------------------------------------")
print(*equations,sep="\n")
#print(sorted(equations,key=str))

#equations=sorted(equations,key=str)
#print(sympy.python(equations[0]))
#exit()
#print(sympy.factor_list(equations[0]))






def symeval(equations,x,y,z):
    cache={"x":x,"y":y,"z":z}
    def evaluate(equation):
        eqs=sympy.sstr(equation,full_prec=True)
        if (ret:=cache.get(eqs,None))is not None:
            return ret
        elif equation.is_number:
            ret=float(equation)
        elif equation.func==sympy.Add:
            ret=functools.reduce(operator.add,map(evaluate,equation.args))
        elif equation.func==sympy.Mul:
            ret=functools.reduce(operator.mul,map(evaluate,equation.args))
        elif equation.func==sympy.Pow:
            a,e=equation.args
            ret=evaluate(a)**evaluate(e)
            #print(equation.args)
        else:
            print(equation.func)
            raise Exception(f"type {equation.func} not supported")
        cache[eqs]=ret
        return ret
    return [evaluate(eq) for eq in equations]


#exit()
#print(print(sympy.simplify(sum(abs(blade.magnitude) for blade in expr.lst))))


        
from intervallarethmetic.intervallarethmetic1 import intervallareth
from intervallarethmetic.voxels import Voxels
t0=time.time()




depth=16
maxvoxelnum=100000

voxels=Voxels(64)
#intervallx,intervally,intervallz=intervallx+17.1225253,intervally+13.127876,intervallz+32.135670
zerrovec=np.zeros(3)

lastvoxnum=1
for j in range(1,depth+1):

    """intervalls=[]
    for intervall,order in zip((intervallx,intervally,intervallz),(1,2,4)):
        mid=intervall.mid()
        intervalls.append(
            intervallareth(
                np.concatenate([[intervall.min,mid][i&order==0] for i in range(8)]),
                np.concatenate([[mid,intervall.max][i&order==0] for i in range(8)])
                #TODO intervall concatenate method
            )
        )"""
    intervallx,intervally,intervallz=voxels.intervallarethpoints()


    #print(np.stack([intervallx.min,intervallx.max,intervally.min,intervally.max,intervallz.min,intervallz.max]).T)
    #delete cells without 0
    #voxelsintervall=point(intervallx,intervally,intervallz).inner(vis)#calculate #TODO
    #print(evaluatefun(equations[0],cache))
    #voxelswithzerro=sum(abs(evaluatefun(e,cache)) for e in equations).containsnum()#TODO
    voxelswithzerro=np.all([vec.containsnum() for vec in symeval(equations,intervallx,intervally,intervallz)],axis=0)#TODO

    #TODO wenn ableitung sum(abs(evaluatefun(e,cache)) for e in equations).containsnum() nach x,y,z contains 0,0,0 
    print(voxelswithzerro)
    voxels.removecells(voxelswithzerro)
    
    """if len(intervallx.min)>10000 or 1:
        
        c_n0 = (intervallx.min, intervally.min, intervallz.min)
        c_n1 = (intervallx.max, intervally.min, intervallz.min)
        c_n2 = (intervallx.min, intervally.max, intervallz.min)
        c_n3 = (intervallx.max, intervally.max, intervallz.min)
        c_n4 = (intervallx.min, intervally.min, intervallz.max)
        c_n5 = (intervallx.max, intervally.min, intervallz.max)
        c_n6 = (intervallx.min, intervally.max, intervallz.max)
        c_n7 = (intervallx.max, intervally.max, intervallz.max)

        
        punkte=[c_n0,c_n1,c_n2,c_n3,c_n4,c_n5,c_n6,c_n7]
        deltas=[]
        
        for(x,y,z)in punkte:
            xtf,ytf,ztf=[tf.Variable(i) for i in (x,y,z)]
            with tf.GradientTape(persistent=True) as tape:
                voltf=sum([tf.abs(vec) for vec in symeval(equations[:5], xtf,ytf,ztf)])

            delta=np.stack([tape.gradient(voltf, xtf).numpy(),tape.gradient(voltf, ytf).numpy(),tape.gradient(voltf, ztf).numpy()],axis=1)
            
            del tape
            deltas.append(delta)
        
        #hasinversion=np.zeros(len(intervallx.min), dtype=bool)
        #for i in range(1,len(deltas)):
            #print(np.sum(deltas[i-1]*deltas[i],axis=1))
            #hasinversion|=np.sum(deltas[i-1]*deltas[i],axis=1)<=0
        #print(hasinversion)
        #intervallx,intervally,intervallz=[intervallareth(intervall.min[hasinversion],intervall.max[hasinversion]) for intervall in (intervallx,intervally,intervallz)]

        deltaslen=[np.einsum('ij,ij->i',d,d)for d in deltas]
        for d in deltaslen:
            d[d==0] = 1
        #sum(np.sum(deltas[i-1]*deltas[i],axis=1))
        angle=sum(np.arccos(np.clip(np.einsum('ij,ij->i',deltas[i-1],deltas[i])/np.sqrt(deltaslen[i-1]*deltaslen[i]),-1,1))for i in range(1,len(deltas)))
        #mplt.hist(angle,100)
        #mplt.title("angles")
        #mplt.xlabel("anglesum")
        #mplt.ylabel("Häufigkeit")
        #mplt.show()
        #print(np.isnan(angle).any())
        angle=angle>1
        print(len(intervallx.min),sum(voxelswithzerro==0),sum(angle==0),j)
        intervallx,intervally,intervallz=[intervallareth(intervall.min[angle],intervall.max[angle]) for intervall in (intervallx,intervally,intervallz)]
    """
        


    #print(len(voxelswithzerro),j)
    if len(voxelswithzerro)>maxvoxelnum:
        depth=j
        break
    voxels.subdivide()


print(time.time()-t0)






import cProfile, pstats, io
from pstats import SortKey
pr = cProfile.Profile(builtins=False)
pr.enable()

#print(time.time()-t0)
#grid = pv.UnstructuredGrid(cells_mat.ravel(), np.array([pv.CellType.VOXEL]*len(cells_mat), np.int8), all_nodes.ravel())
plt.add_mesh(voxels.gridify(),opacity=0.5)
pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(10)


plt.show()


#entweder f'==0 oder vorzeichenwechsel in den punkten
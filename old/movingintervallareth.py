from intervallareth.intervallarethmetic1 import intervallareth

import sympy
import numpy as np
x,a=sympy.symbols("x a")

#f=x**7 - 2*x**6 - 4*x**5 + 10*x**4 + x**3 - 12*x**2 + 6*x
#f=(x-3)**2*(x+3)*(x+37)*(x-82)
#f=f.subs(x,(x+0.25)*20)
f=-5.302721495*10**(-8)*x**(10)+0.000006386725359*x**(9)-0.0003226963201*x**(8)+0.009013570906*x**(7)-0.1535425783*x**(6)+1.652938279*x**(5)-11.23137846*x**(4)+46.50445519*x**(3)-108.1093811*x**(2)+118.6680891*x-39.98491616
#f=-x**5-0.01*3*x**6+0.001*x**7
#f=x+0.05*x**2
#f=x**2

#f=sympy.simplify(f.subs(x,x-30))
fa=sympy.Poly(f.subs(x,x+a),x).as_expr()
fa2=sympy.Poly(fa.subs(x,x+1),x).as_expr()
fa3=sympy.Poly(fa.subs(x,(x+1)*0.5),x).as_expr()
#fa3=sympy.Poly(fa.subs(x,x+1),x).as_expr()
#f4=f
f4=sympy.expand(f)
print(fa)
print(fa2)
print(fa3)
print(f4)
f0=fa.subs(x,0)
f1=fa.subs(x,1)
#print(f0)
#print(f1)
#exit()
#f=sympy.simplify(f.subs(x,x-30))

#print(str(fa))

xlower=-10
xupper=10
#xpoints2=np.linspace(-100,100,1000)
xpoints=np.arange(xlower,xupper,dtype=float)

zp=intervallareth(0,1)
fastr=str(fa).replace("a","i").replace("x","zp")

zn=intervallareth(-1,0)
fastr2=str(fa2).replace("a","i").replace("x","zn")#zn=intervall zero negative 

inp=intervallareth(-1,1)
fastr3=str(fa3).replace("a","i").replace("x","inp")#inp =intervall negative positive


fastr4=str(f4).replace("x","ix")
#xpointsmm=intervallareth(xpoints2-0.5,xpoints2+0.5)
#ami=[]
#ama=[]
#ami2=[]
#ama2=[]

import matplotlib.pyplot as plt
import matplotlib.patches as patches
ax = plt.gca()
#for mi,ma,xval in zip(ami,ama,xpoints):
#    ax.add_patch(patches.Rectangle((xval,mi),1,ma-mi,edgecolor=None,facecolor="r",alpha=0.5))
for i in xpoints:

    mima=eval(fastr)
    ax.add_patch(patches.Rectangle((i,mima.min),1,mima.max-mima.min,edgecolor=None,facecolor="r",alpha=0.5))
    mima=eval(fastr2)
    ax.add_patch(patches.Rectangle((i,mima.min),1,mima.max-mima.min,edgecolor=None,facecolor="b",alpha=0.5))
    mima=eval(fastr3)
    ax.add_patch(patches.Rectangle((i,mima.min),1,mima.max-mima.min,edgecolor="g",facecolor="None",alpha=1))

    ix=intervallareth(i,i+1)
    mima=eval(fastr4)
    ax.add_patch(patches.Rectangle((i,mima.min),1,mima.max-mima.min,edgecolor="y",facecolor="None",alpha=0.5))
    #print(i)
    #print(sympy.expand(f.subs(x,-x+i)))
    #mima=eval(fastr)#.eval({a: i, x: zerro})
    #mima=eval(str(sympy.expand(f.subs(x,-x+i))).replace("x","zerro"))
    #ami.append(mima.min)
    #ama.append(mima.max)

    #mima2=eval(fastr2)
    #mima3=eval(fastr3)
    #ami.append(mima3.min)
    #ama.append(mima3.max)
    #ami.append(max(mima.min,mima2.min))
    #ama.append(min(mima.max,mima2.max))
    #ami2.append(mima2.min)
    #ama2.append(mima2.max)
#mima=eval(str(f).replace("x","xpointsmm"))
#ami2=mima.min
#ama2=mima.max

#ami=eval(str(f0).replace("a","xpoints"))
#ama=eval(str(f1).replace("a","xpoints"))

xpoints=np.arange(xlower,xupper,0.01,dtype=float)

ypoints = eval(str(f).replace("x","xpoints"))
#ypoints2 = eval(str(f).replace("x","xpoints"))


    

#plt.plot(xpoints, ama,"r--")
#plt.plot(xpoints, ami,"b--")
#plt.plot(xpoints, ami2)
#plt.plot(xpoints, ama2)
plt.plot(xpoints, ypoints,"g")
#plt.plot(xpoints2, ami2)
#plt.plot(xpoints2, ama2)
#plt.plot(xpoints, ypoints2)
#plt.ylim(-100, 100)
#plt.xlim(-100, 100)
plt.show()
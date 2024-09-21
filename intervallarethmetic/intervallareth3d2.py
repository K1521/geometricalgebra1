import numpy as np

from intervallarethmetic.intervallarethmetic1 import intervallareth
from algebra.algebrabase import SimpleAlgebraBase

class poly3d(SimpleAlgebraBase):

    ix: 'poly3d'  # Declared as a poly3d instance wich gets initialized later
    iy: 'poly3d'
    iz: 'poly3d'

    def __str__(self):
        return str(self.coeffs)
    def __init__(self,coeffs=None):
        self.coeffs=coeffs or dict()
    
    def convert(self, x) -> "poly3d":
        if isinstance(x,poly3d):
            return x
        if isinstance(x, (int, float)) and x==0:
            return poly3d()
        return poly3d({(0,0,0):x})

    def mul(s,o) -> "poly3d":
        coeffs=dict()
        for es,cs in s.coeffs.items():
            for eo,co in o.coeffs.items():
                e=tuple(a+b for a,b in zip(es,eo))
                c=coeffs.get(e,0)
                #coeffs[e]=cs*co+ c if c else cs*co
                coeffs[e]=cs*co+ c
        return poly3d(coeffs)
    
    def add(s,o) -> "poly3d":
        
        small, large = sorted([s.coeffs, o.coeffs], key=len)
        result = large.copy()
        # Update values from the smaller dictionary
        for key, value in small.items():
            r=result.get(key, 0) 
            #result[key] = value+ r if r else value#doesnt work for numpy arrays
            result[key] = value+ r
        
        return poly3d(result)


    def intervallnp(self,ix=1,iy=1,iz=1):#ix,iy,iz represent the intervall [-ix,ix],[-iy,iy],[-iz,iz]
        zcoeff=self.coeffs.get((0,0,0),0)
        posicoeff=[]
        othercoeff=[]
        for (ex,ey,ez),coeff in self.coeffs.items():
            polymag=ix**ex*iy**ey*iz**ez#intervallx**exponentx*...
            coeff=coeff*polymag
            if ex==ey==ez==0:
                pass
            elif ex%2==0 and ey%2==0 and ez%2==0:
                posicoeff.append(coeff) 
            else:
                othercoeff.append(coeff)
        othercoeffnp=sum([np.abs(c) for c in othercoeff])
        #print([np.where(c<0,c,0)for c in posicoeff])
        #print(posicoeff)
        low=zcoeff-othercoeffnp+sum([np.where(c<0,c,0)for c in posicoeff])
        high=zcoeff+othercoeffnp+sum([np.where(c>0,c,0)for c in posicoeff])
        return intervallareth(low,high)
    
    def intervallnp(self,x=1,y=1,z=1,delta=1):

        x=intervallareth(x-delta,x+delta)
        y=intervallareth(y-delta,y+delta)
        z=intervallareth(z-delta,z+delta)

        monomcachex={exp:x**exp for exp in set(ex for (ex,ey,ez) in self.coeffs.keys())}
        monomcachey={exp:y**exp for exp in set(ey for (ex,ey,ez) in self.coeffs.keys())}
        monomcachez={exp:z**exp for exp in set(ez for (ex,ey,ez) in self.coeffs.keys())}
        
        
        s=0
        for (ex,ey,ez),coeff in self.coeffs.items():
            #print(coeff)
            s+=coeff*(monomcachex[ex]*monomcachey[ey]*monomcachez[ez])
        #print(s)
        #if s.min==
        return s

poly3d.ix=poly3d({(1,0,0):1})
poly3d.iy=poly3d({(0,1,0):1})
poly3d.iz=poly3d({(0,0,1):1})
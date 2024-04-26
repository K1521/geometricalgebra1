import numpy as np

from intervallarethmetic.intervallarethmetic1 import intervallareth
from algebra.algebrabase import SimpleAlgebraBase

class inter3d(SimpleAlgebraBase):
    def __str__(self):
        return str(self.coeffs)
    def __init__(self,coeffs=None):
        self.coeffs=coeffs or dict()
    
    def convert(self, x) -> "inter3d":
        if isinstance(x,inter3d):
            return x
        if isinstance(x, (int, float)) and x==0:
            return inter3d()
        return inter3d({(0,0,0):x})

    def mul(s,o) -> "inter3d":
        coeffs=dict()
        for es,cs in s.coeffs.items():
            for eo,co in o.coeffs.items():
                e=tuple(a+b for a,b in zip(es,eo))
                c=coeffs.get(e,0)
                #coeffs[e]=cs*co+ c if c else cs*co
                coeffs[e]=cs*co+ c
        return inter3d(coeffs)
    
    def add(s,o) -> "inter3d":
        
        small, large = sorted([s.coeffs, o.coeffs], key=len)
        result = large.copy()
        # Update values from the smaller dictionary
        for key, value in small.items():
            r=result.get(key, 0) 
            #result[key] = value+ r if r else value#doesnt work for numpy arrays
            result[key] = value+ r
        
        return inter3d(result)

    def tointer(self):#maybe it would be possible to do something like i**2-i**3 to eliminate i**3 and reduce i**2 because i**2>i**3 if i=[0,1]
        l=0
        h=0
        #TODO handle even exponents
        for es,cs in self.coeffs.items():
            if cs>0:
                h+=cs
            else:
                l+=cs
        cs=self.coeffs.get((0,0,0),0)
        if cs<=0:
            h+=cs
        else:
            l+=cs
        return l,h
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



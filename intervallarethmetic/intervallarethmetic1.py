import numpy as np

class intervallareth:
    def __init__(self,min,max) -> None:
        self.min=min
        self.max=max
    
    def __str__(self):
        return f"[{self.min},{self.max}]"

    def __mul__(self,other):
        #todo optimise case if self==other
        if other==self:return self**2
        if isinstance(other,intervallareth):
            combis=[self.min*other.min,self.min*other.max,self.max*other.min,self.max*other.max]
            return intervallareth(np.min(combis, axis=0),np.max(combis, axis=0))
        if other==0:return 0
        if other==1:return self
        if other==-1:return intervallareth(-self.max,-self.min)
        if other>0:
            return intervallareth(self.min*other,self.max*other)
        else:
            return intervallareth(self.max*other,self.min*other)
    def __rmul__(self,other):
        return self*other
    
    def __add__(self,other):
        if isinstance(other,intervallareth):
            return intervallareth(self.min+other.min,self.max+other.max)
        #if other==0:return self
        return intervallareth(self.min+other,self.max+other)
    def __radd__(self,other):
        return self+other
    
    def __sub__(self,other):
        if isinstance(other,intervallareth):
            return intervallareth(self.min-other.max,self.max-other.min)
        #if other==0:return self
        return intervallareth(self.min-other,self.max-other)
    def __rsub__(self,other):
        return intervallareth(other-self.max,other-self.min)
    def __neg__(self):
        return intervallareth(-self.max,-self.min)

    def mid(self):
        #return (self.min+self.max)/2
        #print(self.min,self.max)
        return np.average([self.min,self.max], axis=0)
    def containsnum(self,num=0,maxdelta=0):
        return (self.min<=num+maxdelta)&(self.max>=num-maxdelta)
    def __abs__(self):
        combis=[abs(self.min),abs(self.max)]
        return intervallareth(np.where((self.min<=0)&(0<=self.max), 0, np.min(combis, axis=0)),np.max(combis, axis=0))

    def __pow__(self,exponent):
        if exponent < 0:
            raise ValueError("Power must be non-negative")
        elif exponent == 0:
            #return 1
            return intervallareth(1,1)
        
        
        if exponent % 2 == 0:
            combis=[self.min**exponent,self.max**exponent]
            return intervallareth(np.where((self.min<=0)&(0<=self.max), 0, np.min(combis, axis=0)),np.max(combis, axis=0))
        else:
            return intervallareth(self.min**exponent,self.max**exponent)
    def __truediv__(self,other):
        if isinstance(other,intervallareth):
            raise ValueError("division only suported with scalars")
        return self*(1/other)
    
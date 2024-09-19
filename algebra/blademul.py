import itertools
from collections import defaultdict

class blade:
    __slots__ = ['basis', 'magnitude']
    def __init__(self,basis,magnitude=1) -> None:
        self.magnitude=magnitude
        self.basis=basis
    def __repr__(self):
        return f"blade({self.basis:b},{self.magnitude})"
    

class algebra:
    def __init__(self,posi=3,nega=0,neut=0,order="pnz"):
        self.zero=blade(0,0)
        self.posi=posi
        self.nega=nega
        self.neut=neut
        self.dim=posi+nega+neut

        #self.bladenames=[str(i) for i in range(self.dim)]#list(map(str,range(self.dim)))
        self.bladenames=None
        self.trenner=""
        self.setbladenames([str(i) for i in range(self.dim)])

        
        if sorted(order)!=sorted("pnz"):
            raise Exception("order must be a permutatioon of pnz like pzn or npz")
        s=0
        for o in order:
            if o=="p":
                self.posimask=(2**posi-1)<<s
                s+=posi
            if o=="n":
                self.negamask=(2**nega-1)<<s
                s+=nega
            if o=="z":
                self.neutmask=(2**neut-1)<<s
                s+=neut

    
    def setbladenames(self,bladenames,trenner=None):
        if len(bladenames)!=self.dim:
            raise ValueError("length of bladenames must match dim")
        self.bladenames=bladenames
        if trenner is not None:
            self.trenner=trenner
        else:
            trenner="," if max(map(len,bladenames))>1 else ""


    
    def bladesortkey(self,blade):
        l=self.basedecode(blade.basis)
        return len(l),l

    def basedecode(self,basis):
        #return tuple(i for i,x in enumerate(bin(basis)[:1:-1],0)if x=="1")
        return tuple(i for i in range(basis.bit_length()) if basis&(1<<i))
        #int("1100001",2) -> (0, 5, 6)

    def basename(self,basis):
        if basis==0:
            return "1"
        return "e"+self.trenner.join(self.bladenames[i] for i in self.basedecode(basis))

    def bladestr(self,blade:blade):
        return f"({blade.magnitude})*{self.basename(blade.basis)}"
    
    def iszero(self,blade):
        if blade is self.zero:
            return True
        if isinstance(blade.magnitude, (int, float)) and blade.magnitude == 0:
            return True
        return False
    
    def geo(self,blade1:blade,blade2:blade):
        #if both have alligning neutral vector return 0
        if self.neutmask&blade1.basis&blade2.basis or self.iszero(blade1) or self.iszero(blade2):
            return self.zero
        
        #count inversions
        bas1acc=blade1.basis>>1
        for i in range(blade1.basis.bit_length().bit_length()):#lol
            bas1acc^=(bas1acc>>(1<<i))
        invert=(bas1acc&blade2.basis).bit_count()&1

        
        #calculate negative alligned "inversions"
        invert^=(self.negamask&blade1.basis&blade2.basis).bit_count()&1

        #calculate magnitude
        magnitude=blade1.magnitude*blade2.magnitude
        if invert:
            magnitude=-magnitude
        return blade(blade1.basis^blade2.basis, magnitude)

    def outer(self,blade1:blade,blade2:blade):
        if blade1.basis&blade2.basis:
            return self.zero
        return self.geo(blade1,blade2)

    def inner(self,blade1:blade,blade2:blade):
        if  (((~blade1.basis)&blade2.basis) and (blade1.basis&(~blade2.basis))):#works analog zu 
            return self.zero
        return self.geo(blade1,blade2)
    

    def reverse(self,blade):
        if blade.basis.bit_count()&1:
            return blade(blade.basis, -blade.magnitude)
        else:
            return blade
    
    def grade(self,bladeo):
        return bladeo.basis.bit_count()

    def reverse(self,bladeo):
        g=self.grade(bladeo)
        sign=(-1)**(g*(g-1)//2)
        if sign==1:
            return bladeo
        else:
            return blade(bladeo.basis,-bladeo.magnitude)
            
    def conjugate(self,bladeo):
        g=self.grade(bladeo)
        sign=(-1)**(g*(g+1)//2)
        if sign==1:
            return bladeo
        else:
            return blade(bladeo.basis,-bladeo.magnitude)
    def involute(self,bladeo):
        g=self.grade(bladeo)
        sign=(-1)**g
        if sign==1:
            return bladeo
        else:
            return blade(bladeo.basis,-bladeo.magnitude)



class sortgeo:
    #@staticmethod
    #def filterzero(it):
    #    return list(i for i in it if i.magnitude!=0)

    def __init__(self,algebra:algebra,lst=None) -> None:
        if lst is None:
            lst=[]
        self.algebra=algebra

        if not isinstance(lst, list):
            lst=list(lst)
        
        self.lst=lst

        #invariants:
        #lst is always sorted by blade.base
        #any(self.algebra.iszero(x) for x in lst)==False
    
    def allblades(self,sort=True):
        lst=[sortgeo(self.algebra,[blade(i)]) for i in range(2**self.algebra.dim)]
        if sort:
            return sorted(lst,key=lambda x:self.algebra.bladesortkey(x.lst[0]))
        return lst
    def monoblades(self):
        return [sortgeo(self.algebra,[blade(1<<i)])for i in range(self.algebra.dim)]
   
    def scalar(self,num):
        s=blade(0,num)
        if self.algebra.iszero(s):
            return sortgeo(self.algebra)
        return sortgeo(self.algebra,[s])
    
    def __str__(self) -> str:
        if not self.lst:
            return self.algebra.bladestr(self.algebra.zero)
        return " + ".join(self.algebra.bladestr(b) for b in sorted(self.lst,key=self.algebra.bladesortkey))
    
    def __repr__(self):
        return self.__str__()
    
    def _check_algebra(self, other):
        if self.algebra != other.algebra:
            raise ValueError(f"Incompatible algebras: {self.algebra} and {other.algebra}")
    
    def _map(self,func):
        return sortgeo(self.algebra, [func(b) for b in self.lst])
    
    
    def involute(self):
        return self._map(self.algebra.involute)
    def conjugate(self):
        return self._map(self.algebra.conjugate)
    def reverse(self):
        return self._map(self.algebra.reverse)

    def _crosscompress(self,other,func):
        #self._check_algebra(other)

        bladelst=(func(x,y) for x in self.lst for y in other.lst)
        return self._makecompressed(bladelst)
    
    def _makecompressed(self,bladelst):
        #bladelst=(x for x in bladelst if not self.algebra.iszero(x))
        getblades=lambda x:x.basis
        bladelst=sorted(bladelst,key=getblades)

        lstnew=[]
        for k,g in itertools.groupby(bladelst,key=getblades):
            #g=list(g.magnitude)
            #s=
            b=blade(k,sum(x.magnitude for x in g if not self.algebra.iszero(x)))
            #b=blade(k,sum(x.magnitude for x in g))
            if not self.algebra.iszero(b):
                lstnew.append(b)

        return sortgeo(self.algebra, lstnew)
        
    def inner(self,other):
        if not isinstance(other,sortgeo):
            #other=self.scalar(other)
            return self*other
        else:
            self._check_algebra(other)
        return self._crosscompress(other,self.algebra.inner)
        #return sortgeo(self.algebra,(self.algebra.inner(x,y) for x in self.lst for y in other.lst),compress=True)

    def __xor__(self,other):
        return self.outer(other)
    def outer(self,other):
        if not isinstance(other,sortgeo):
            #other=self.scalar(other)
            return self*other
        else:
            self._check_algebra(other)
        return self._crosscompress(other,self.algebra.outer)

    def __add__(self,other):
        if not isinstance(other,sortgeo):
            other=self.scalar(other)
        else:
            self._check_algebra(other)
        return self._makecompressed(self.lst+other.lst)

    def _mapmagnitude(self,func):
        return sortgeo(self.algebra, [blade(b.basis,func(b.magnitude)) for b in self.lst])
    def __sub__(self,other):
        return self+ (-other)
    def __rsub__(self,other):
        return (-self)+ other
    def __neg__(self):
        return self._mapmagnitude(lambda x: -x)

    def __mul__(self,other):
        if not isinstance(other,sortgeo):
            #other=self.scalar(other)
            return self._mapmagnitude(lambda x:x*other)#skalarmul
        else:
            self._check_algebra(other)
            return self._crosscompress(other,self.algebra.geo)#geometric product
        
    def __rmul__(self,other):#skalarmul
        return self._mapmagnitude(lambda x:other*x)
    def __truediv__(self,other):
        if isinstance(other,sortgeo):
            raise Exception("currently only integer/float division is supported")
        return self._mapmagnitude(lambda x:x/other)
    def toscalar(self):
        if not self.lst:
            return 0
        if len(self.lst)==1 and self.lst[0].basis==0:
            return self.lst[0].magnitude
        raise Exception("not convertible")
    



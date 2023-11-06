import itertools
from collections import defaultdict

class blade:
    def __init__(self,basis,magnitude=1) -> None:
        self.magnitude=magnitude
        self.basis=basis
    def __neg__(self):
        return blade(self.basis,-self.magnitude)
    def __mul__(self,x):
        #TODO no computation if magnitude is 1
        #if self.magnitude
        return blade(self.basis,self.magnitude*x)

class algebra:
    def __init__(self,posi=3,nega=0,neut=0,order="pnz"):
        self.zero=blade(0,0)
        self.posi=posi
        self.nega=nega
        self.neut=neut
        self.dim=posi+nega+neut

        self.bladenames=[str(i) for i in range(self.dim)]#list(map(str,range(self.dim)))

        s=0
        if sorted(order)!=sorted("pnz"):
            raise Exception("order must be a permutatioon of pnz like pzn or npz")
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
    
    def basedecode(self,basis):
        return tuple(i for i,x in enumerate(bin(basis)[:1:-1],0)if x=="1")
    def bladestr(self,blade:blade):
        if self.bladenames==0:
            return "+0"
        if blade.basis==0:
            return f"{blade.magnitude:+}"
        trenner="," if max(map(len,self.bladenames))>1 else ""
        return f"{blade.magnitude:+}*e"+trenner.join(self.bladenames[i] for i in self.basedecode(blade.basis))
    
    def geo(self,blade1:blade,blade2:blade):
        #if both have alligning neutral vector return 0
        if self.neutmask&blade1.basis&blade2.basis:
            return self.zero
        
        #count inversions
        bas1acc=blade1.basis^(blade1.basis.bit_count()&1)
        i=1
        l=min(blade1.basis.bit_length(),blade2.basis.bit_length())
        mask=(2<<l)-1
        while i<=l:
            bas1acc^=(bas1acc<<i)&mask
            i<<=1
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
    
    def bladesortkey(self,blade):
        l=self.basedecode(blade.basis)
        return len(l),l
    



    

class sortgeo:
    #@staticmethod
    #def filterzero(it):
    #    return list(i for i in it if i.magnitude!=0)

    def __init__(self,algebra,lst=None,compress=False) -> None:
        self.algebra=algebra
        if lst is None:
            self.lst=[]
        elif not isinstance(lst, list):
            self.lst=[i for i in lst if i.magnitude!=0]
        else:
            self.lst=lst
        if compress:
            self.compress()
    def allblades(self,sort=True):
        lst=[sortgeo(self.algebra,[blade(i)]) for i in range(2**self.algebra.dim)]
        if sort:
            return sorted(lst,key=lambda x:self.algebra.bladesortkey(x.lst[0]))
        return lst
    def monoblades(self):
        return [sortgeo(self.algebra,[blade(1<<i)])for i in range(self.algebra.dim)]

    def __str__(self) -> str:
        if not self.lst:
            return self.algebra.bladestr(self.algebra.zero)
        return "".join(self.algebra.bladestr(b) for b in sorted(self.lst,key=self.algebra.bladesortkey))
    
    def __repr__(self):
        return self.__str__()

    def compress(self):
        
        lstnew=[]
        getblades=lambda x:x.basis
        self.lst.sort(key=getblades)
        for k,g in itertools.groupby((i for i in self.lst if i.magnitude!=0),key=getblades):
            g=list(g)
            #print(len(g))
            if len(g)==1:
                lstnew.append(g[0])
            else:
                lstnew.append(blade(k,sum(x.magnitude for x in g)))
        self.lst=lstnew
        #actblades=lst[0].blades
        #actmagnitude=lst[0].blades#i have desided to make blades imutable
        #for i in range(1,len(lst)):
        #    if lst[i].blades==lst[ilast].blades:
    def __matmul__(self,othe):
        #lst=[]
        #for x in self.lst:
            #for y in othe.lst:
                #lst.append(x@y)
        return sortgeo(self.algebra,(self.algebra.geo(x,y) for x in self.lst for y in othe.lst),compress=True)
    def inner(self,othe):
        return sortgeo(self.algebra,(self.algebra.inner(x,y) for x in self.lst for y in othe.lst),compress=True)
    def outer(self,othe):
        return sortgeo(self.algebra,(self.algebra.outer(x,y) for x in self.lst for y in othe.lst),compress=True)
    def __add__(self,othe):
        return sortgeo(self.algebra,self.lst+othe.lst,compress=True)
    def __sub__(self,othe):
        return self+ (-othe)
    def __neg__(self):
        return sortgeo(self.algebra,(-i for i in self.lst),compress=False)
    def __mul__(self,x):
        return sortgeo(self.algebra,(i*x for i in self.lst),compress=False)
    def __rmul__(self,x):
        return self*x
    def toscalar(self):
        if not self.lst:
            return 0
        if len(self.lst)==1:
            if self.lst[0].basis==0:
                return self.lst[0].magnitude
        self.compress()
        if not self.lst:
            return 0
        if len(self.lst)==1:
            if self.lst[0].basis==0:
                return self.lst[0].magnitude
        raise Exception("not convertible")

class dictgeo:
    #@staticmethod
    #def filterzero(it):
    #    return list(i for i in it if i.magnitude!=0)

    def __init__(self,algebra,lst=None) -> None:
        self.algebra=algebra
        if lst is None:
            self.d=dict()
        elif isinstance(lst, dict) or isinstance(lst, defaultdict):
            self.d=lst
        else:
            self.d={i.basis:i.magnitude for i in lst if i.magnitude!=0}
        
    def getblades(self):
        return [blade(*x)for x in self.d.items()]
    def allblades(self,sort=True):
        lst=[dictgeo(self.algebra,[blade(i)]) for i in range(2**self.algebra.dim)]
        if sort:
            return sorted(lst,key=lambda x:self.algebra.bladesortkey(x.getblades()[0]))
        return lst
    def monoblades(self):
        return [dictgeo(self.algebra,[blade(1<<i)])for i in range(self.algebra.dim)]

    def __str__(self) -> str:
        if not self.d:
            return self.algebra.bladestr(self.algebra.zero)
        return "".join(self.algebra.bladestr(b) for b in sorted(self.getblades(),key=self.algebra.bladesortkey))
    
    def __repr__(self):
        return self.__str__()

   
    def __matmul__(self,othe):
        #lst=[]
        #for x in self.lst:
            #for y in othe.lst:
                #lst.append(x@y)
        d=defaultdict(int)
        for x in self.d.items():
            for y in othe.d.items():
                b=self.algebra.geo(blade(*x),blade(*y))
                if b.magnitude!=0:
                    d[b.basis]+=b.magnitude
        return dictgeo(self.algebra,d)

        #return dictgeo(self.algebra,(self.algebra.geo(blade(*x),blade(*y)) for x in self.d.items() for y in othe.d.items()))
    def inner(self,othe):
        d=defaultdict(int)
        for x in self.d.items():
            for y in othe.d.items():
                b=self.algebra.inner(blade(*x),blade(*y))
                if b.magnitude!=0:
                    d[b.basis]+=b.magnitude
        return dictgeo(self.algebra,d)
    def outer(self,othe):
        d=defaultdict(int)
        for x in self.d.items():
            for y in othe.d.items():
                b=self.algebra.outer(blade(*x),blade(*y))
                if b.magnitude!=0:
                    d[b.basis]+=b.magnitude
        return dictgeo(self.algebra,d)
    def __add__(self,othe):
        return dictgeo(self.algebra,{basis:magnitude for basis in (self.d.keys()|othe.d.keys()) if (magnitude:=(self.d.get(basis,0)+othe.d.get(basis,0)))!=0})
    def __sub__(self,othe):
        return self+ (-othe)
    def __neg__(self):
        return dictgeo(self.algebra,{basis:-magnitude for basis,magnitude in self.d.items() if magnitude!=0})
    def __mul__(self,x):
        return dictgeo(self.algebra,{basis:magnitude*x for basis,magnitude in self.d.items() if magnitude!=0})
    def __rmul__(self,x):
        return self*x
    def toscalar(self):
        return self.d.get(0,0)
        #todo raise Exception("not convertible")



class sortgeotf:
    #@staticmethod
    #def filterzero(it):
    #    return list(i for i in it if i.magnitude!=0)

    def __init__(self,algebra,lst=None,compress=False) -> None:
        self.algebra=algebra
        self.lst=[i for i in lst if i!=self.algebra.zero]
        if compress:
            self.compress()
    def allblades(self,sort=True):
        lst=[sortgeotf(self.algebra,[blade(i)]) for i in range(2**self.algebra.dim)]
        if sort:
            return sorted(lst,key=lambda x:self.algebra.bladesortkey(x.lst[0]))
        return lst
    def monoblades(self):
        return [sortgeotf(self.algebra,[blade(1<<i)])for i in range(self.algebra.dim)]

    def __str__(self) -> str:
        if not self.lst:
            return self.algebra.bladestr(self.algebra.zero)
        return "".join(self.algebra.bladestr(b) for b in sorted(self.lst,key=self.algebra.bladesortkey))
    
    def __repr__(self):
        return self.__str__()

    def compress(self):
        
        lstnew=[]
        getblades=lambda x:x.basis
        self.lst.sort(key=getblades)
        for k,g in itertools.groupby((i for i in self.lst if self.algebra.zero!=0),key=getblades):
            g=list(g)
            if len(g)==1:
                lstnew.append(g[0])
            else:
                lstnew.append(blade(k,sum(x.magnitude for x in g)))
        self.lst=lstnew
        #actblades=lst[0].blades
        #actmagnitude=lst[0].blades#i have desided to make blades imutable
        #for i in range(1,len(lst)):
        #    if lst[i].blades==lst[ilast].blades:
    def __matmul__(self,othe): 
        return sortgeotf(self.algebra,(self.algebra.geo(x,y) for x in self.lst for y in othe.lst),compress=True)
    def inner(self,othe):
        return sortgeotf(self.algebra,(self.algebra.inner(x,y) for x in self.lst for y in othe.lst),compress=True)
    def __xor__(self,othe):
        return self.outer(othe)
    def outer(self,othe):
        return sortgeotf(self.algebra,(self.algebra.outer(x,y) for x in self.lst for y in othe.lst),compress=True)
    def __add__(self,othe):
        return sortgeotf(self.algebra,self.lst+othe.lst,compress=True)
    def __sub__(self,othe):
        return self+ (-othe)
    def __neg__(self):
        return sortgeotf(self.algebra,(-i for i in self.lst),compress=False)
    def __mul__(self,x):
        return sortgeotf(self.algebra,(i*x for i in self.lst),compress=False)
    def __rmul__(self,x):
        return self*x
    def toscalar(self):
        if not self.lst:
            return 0
        if len(self.lst)==1:
            if self.lst[0].basis==0:
                return self.lst[0].magnitude
        self.compress()
        if not self.lst:
            return 0
        if len(self.lst)==1:
            if self.lst[0].basis==0:
                return self.lst[0].magnitude
        raise Exception("not convertible")
#e1=blade(1<<0)
#e2=blade(1<<1)
#e3=blade(1<<2)

"""
algp1n1z2=algebra(1,1,2,order="zpn")
#algp1n1z2=algebra()

row =algp1n1z2.allblades()
#print(row)
import pandas as pd
print("imported")
table = [[str(x.inner(y))for y in row]for x in row]
df = pd.DataFrame(table, columns = row, index=row)
print(df)


print(bin(algp1n1z2.neutmask))
print(bin(algp1n1z2.posimask))
print(bin(algp1n1z2.negamask))
"""

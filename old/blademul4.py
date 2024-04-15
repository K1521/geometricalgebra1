import itertools

class blade:
    def __init__(self,basis,magnitude=1) -> None:
        self.magnitude=magnitude
        self.basis=basis
    def __neg__(self):
        return blade(self.basis,-self.magnitude)

class algebra:
    def __init__(self,posi=3,nega=0,neut=0,order="pnz"):
        self.zero=blade(0,0)
        self.posi=posi
        self.nega=nega
        self.neut=neut
        self.dim=posi+nega+neut

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
        if blade.magnitude==0:
            return "+0"
        if self.dim>=10:
            return f"{blade.magnitude:+}*e"+','.join(map(str,self.basedecode(blade.basis)))
        else:
            return f"{blade.magnitude:+}*e"+ ''.join(map(str,self.basedecode(blade.basis)))
    
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
    def allblades(self,sort=True):
        lst=[sortgeo(self,[blade(i)]) for i in range(2**self.dim)]
        if sort:
            return sorted(lst,key=lambda x:self.bladesortkey(x.lst[0]))
        return lst
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
        for k,g in itertools.groupby(self.lst,key=getblades):
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
        
#e1=blade(1<<0)
#e2=blade(1<<1)
#e3=blade(1<<2)

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
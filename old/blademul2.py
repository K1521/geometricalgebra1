import itertools


def basedecode(basis):
    return [i for i,x in enumerate(bin(basis)[:1:-1],1)if x=="1"]
class blade:
    #__slots__=("basis","magnitude")
    def __init__(self,basis,magnitude=1) -> None:
        self.magnitude=magnitude
        self.basis=basis
    def __str__(self):
        return f"{self.magnitude:+}"+(("*e"+''.join(map(str,basedecode(self.basis))))if self.basis else "")
    
    def __matmul__(self,othe):#geometric product
        invert=0
        count=self.basis.bit_count()
        for b1,b2 in zip(bin(self.basis)[:1:-1],bin(othe.basis)[:1:-1]):
            count-=b1=="1"
            if b2=="1":
                invert+=count
        magnitude=self.magnitude*othe.magnitude
        if invert%2==1:
            magnitude*=-1
        return blade(self.basis^othe.basis, magnitude)
    def __neg__(self):
        return blade(self.basis,-self.magnitude)
    

class sortgeo:
    def __init__(self,lst=None) -> None:
        if lst is None:
            self.lst=[]
        else:
            self.lst=lst
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
        lst=[]
        for x in self.lst:
            for y in othe.lst:
                lst.append(x@y)
        ret=sortgeo(lst)
        ret.compress()
        return ret
    def __add__(self,othe):
        ret=sortgeo(self.lst+othe.lst)
        ret.compress()
        return ret
        
#e1=blade(1<<0)
#e2=blade(1<<1)
#e3=blade(1<<2)

e1=sortgeo([blade(1<<0)])
e2=sortgeo([blade(1<<1)])
e3=sortgeo([blade(1<<2)])


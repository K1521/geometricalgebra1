import itertools


def basedecode(basis):
    return [i for i,x in enumerate(bin(basis)[:1:-1],0)if x=="1"]
class algebra:
    def __init__(self,posi=3,nega=0,neut=0,order="pnz"):
        self.zero=blade(self,0,0)
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
        #self.posimask=2**posi-1
        #self.negamask=(2**nega-1)<<posi
        #self.neutmask=(2**neut-1)<<(posi+nega)
    
        
    


class blade:
    #__slots__=("basis","magnitude")
    def __init__(self,algebra,basis,magnitude=1) -> None:
        self.magnitude=magnitude
        self.basis=basis
        self.algebra=algebra
        
    def __str__(self):
        return f"{self.magnitude:+}"+(("*e"+''.join(map(str,basedecode(self.basis))))if self.basis else "")
    
    def __matmul__(self,othe):#geometric product
        #if both have alligning neutral vector return 0
        if self.algebra.neutmask&self.basis&othe.basis:
            return self.algebra.zero
        
        #count inversions
        invert=0
        count=self.basis.bit_count()
        for b1,b2 in zip(bin(self.basis)[:1:-1],bin(othe.basis)[:1:-1]):
            #this is also possible with shift operator
            #but by preconverting this should be faster for high dimensions
            #assuming shift has O(n) complexity instead of O(1)
            #this inversion count should have O(n) complexity
            #with n=max(self.basis.bit_length(),othe.basis.bit_length())
            count-=b1=="1"
            if b2=="1":
                invert+=count
        
        #calculate negative alligned "inversions"
        invert+=(self.algebra.negamask&self.basis&othe.basis).bit_count()



        magnitude=self.magnitude*othe.magnitude*(-1 if invert%2 else 1)
        #if invert%2==1:
            #magnitude*=-1
        return blade(self.algebra,self.basis^othe.basis, magnitude)
    def outer(self,othe):
        if self.basis&othe.basis:
            return self.algebra.zero
        return self@othe
    def inner(self,othe):
        if  (((~self.basis)&othe.basis) and (self.basis&(~othe.basis))):#works analog zu 
            return self.algebra.zero
        return self@othe
        #return self@othe
    def __neg__(self):
        return blade(self.basis,-self.magnitude)
    

class sortgeo:
    #@staticmethod
    #def filterzero(it):
    #    return list(i for i in it if i.magnitude!=0)
    def __init__(self,lst=None,compress=False) -> None:
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
            return "+0"
        return "".join(map(str,sorted(self.lst,key=lambda x:x.basis.bit_count())))
    
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
                lstnew.append(blade(g[0].algebra,k,sum(x.magnitude for x in g)))
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
        return sortgeo((x@y for x in self.lst for y in othe.lst),compress=True)
    def inner(self,othe):
        return sortgeo((x.inner(y) for x in self.lst for y in othe.lst),compress=True)
    def outer(self,othe):
        return sortgeo((x.outer(y) for x in self.lst for y in othe.lst),compress=True)
    def __add__(self,othe):
        return sortgeo(self.lst+othe.lst,compress=True)
    def __sub__(self,othe):
        return self+ (-othe)
    def __neg__(self):
        return sortgeo((-i for i in self.lst),compress=False)
        
#e1=blade(1<<0)
#e2=blade(1<<1)
#e3=blade(1<<2)

algp1n1z2=algebra(1,1,2,order="zpn")
#algp1n1z2=algebra()

e0=sortgeo([blade(algp1n1z2,1<<0)])
e1=sortgeo([blade(algp1n1z2,1<<1)])
e2=sortgeo([blade(algp1n1z2,1<<2)])
e3=sortgeo([blade(algp1n1z2,1<<3)])
#e1=sortgeo([blade(algp1n1z2,1<<1)])
#e2=sortgeo([blade(algp1n1z2,1<<2)])



def strtoblade(x):
    b=0
    for i in x:
        b|=1<<int(i)
    return sortgeo([blade(algp1n1z2,b)])

#print(sorted((str(blade(algp1n1z2,x))[4:] for x in range(2**algp1n1z2.dim)),key=lambda x:(len(x),x)))
#row=[strtoblade(x)for x in "0 1 2 3 01 02 03 12 13 23 012 013 023 123 0123".split(" ")]
#row=[strtoblade(x)for x in "1 2 3 12 13 23 123".split(" ")]
row=[sortgeo([blade(algp1n1z2,i)]) for i in sorted(range(2**algp1n1z2.dim),key=lambda x:x.bit_count())]
row=sorted([sortgeo([blade(algp1n1z2,i)]) for i in range(2**algp1n1z2.dim)],key=lambda x:(len(str(x)),str(x)))
#print(row)
import pandas as pd
print("imported")
table = [[str(x.inner(y))for y in row]for x in row]
df = pd.DataFrame(table, columns = row, index=row)
print(df)


print(bin(algp1n1z2.neutmask))
print(bin(algp1n1z2.posimask))
print(bin(algp1n1z2.negamask))

def basedecode(basis):
    return [i for i,x in enumerate(f"{basis:b}"[::-1],1)if x=="1"]
class blade:
    #__slots__=("basis","magnitude")
    def __init__(self,basis,magnitude=1) -> None:
        self.magnitude=magnitude
        self.basis=basis
    def __str__(self):
        
        return f"{self.magnitude:+}"+(("*e"+''.join(map(str,basedecode(self.basis))))if self.basis else "")
    #def __matmul__(self,othe):
    #    baselst1=basedecode(self.basis)
    #    baselst2=basedecode(othe.basis)
    #    invert=0
    #    j=0
    #    for i,x in zip(reversed(range(len(baselst1))),baselst1):
    #        #print(i,x)
    #        for j in range(j,len(baselst2)):
    #            #print(j)
    #            if x <=baselst2[j]:
    #                invert+=i
    #                break
    #    print(invert)


    #    if invert%2==1:
    #        return blade(self.basis^othe.basis,-self.magnitude*othe.magnitude)
    #    else:
    #        return blade(self.basis^othe.basis, self.magnitude*othe.magnitude)
    #    #print(baselst1,baselst2)
    #    #print(f"{self.basis:b}",f"{othe.basis:b}")
    def __matmul__(self,othe):
        invert=0
        base1=self.basis
        base2=othe.basis
        l1=base1.bit_length()
        l2=base2.bit_length()
        #if l1>l2:
            #base1&=(1<<l2)-1#hopfully this makes it faster but it is not actually nessecary
            #invert+=idk
            #print(base1.bit_length(),base2.bit_length(),bin(base1),bin((1<<l2)-1))
        #base1>>=1
        #while base2 and base1:
            #if base2&1:
            #    invert+=base1.bit_count()
                #print((base1>>1).bit_count())
            
            #base2>>=1
            #base1>>=1
        
        
        #while base2 and base1:
        #    i=1
        #    while not base2&1:
        #        base2>>=1
        #        i+=1
        #    base1>>=i
            
        #    invert+=base1.bit_count()
            #print((base1>>1).bit_count())
            
        #    base2>>=1



        #base1>>=1
        #count=base1.bit_count()
        #while base2 and base1:
            #count-=base1&1
            #if base2&1:
                #invert+=count
            #base2>>=1
            #base1>>=1
        count=base1.bit_count()
        #for b1,b2 in zip(format(base1, 'b')[::-1],format(base2, 'b')[::-1]):
        for b1,b2 in zip(bin(base1)[:1:-1],bin(base2)[:1:-1]):
            count-=b1=="1"
            if b2=="1":
                invert+=count
        #print(invert)
        if invert%2==1:
            return blade(self.basis^othe.basis,-self.magnitude*othe.magnitude)
        else:
            return blade(self.basis^othe.basis, self.magnitude*othe.magnitude)
    def __neg__(self):
        return blade(self.basis,-self.magnitude)
    

e1=blade(1<<0)
e2=blade(1<<1)
e3=blade(1<<2)

#e1@e2
#print(e1@e3@e2)

print("-------------")
#print(bin((-1)&1))
#for i in range(10):
#    print(i,i*(i-1)//2,i%4)

import random
b1=blade(int("1101",2))
b2=blade(int("1011",2))
row=[blade(0),e1,e2,e3,e2@e3,-e1@e3,e1@e2,e1@e2@e3]


import cProfile, pstats
profiler = cProfile.Profile()
profiler.enable()
#for y in row:
#    for x in row:
#        x@y
#blades=[blade(i)for i in range(2**10-1)]
blades=[blade(i)for i in range(2**10-1)]
#blades=[blade(random.getrandbits(i))for i in range(1000,1100)]
for b1 in blades:
    for b2 in blades:
        b1@b2


profiler.disable()
stats = pstats.Stats(profiler).sort_stats('ncalls')
stats.print_stats()
#print(b1)
#print(b2)
#print(b1.matmul2(b2))
#print(b1@(b2))
#b1=blade(random.getrandbits(32))
#b2=blade(random.getrandbits(16))
#print(b1)
#print(b2)
#print(b1@b2)
#print(b1.matmul2(b2))
#print(b2@b1)
#print(b2.matmul2(b1))

import pandas as pd
table = [[str(x@y).replace("-1*e13","1*e31").replace("1*e13","-1*e31") for y in row]for x in row]
df = pd.DataFrame(table, columns = row, index=row)
print(df)
print(str(df).split()=="""              +1    +1*e1    +1*e2    +1*e3   +1*e23   -1*e13   +1*e12  +1*e123
+1            +1    +1*e1    +1*e2    +1*e3   +1*e23    1*e31   +1*e12  +1*e123
+1*e1      +1*e1       +1   +1*e12  +-1*e31  +1*e123    -1*e3    +1*e2   +1*e23
+1*e2      +1*e2   -1*e12       +1   +1*e23    +1*e3  +1*e123    -1*e1    1*e31
+1*e3      +1*e3    1*e31   -1*e23       +1    -1*e2    +1*e1  +1*e123   +1*e12
+1*e23    +1*e23  +1*e123    -1*e3    +1*e2       -1   -1*e12    1*e31    -1*e1
-1*e13     1*e31    +1*e3  +1*e123    -1*e1   +1*e12       -1   -1*e23    -1*e2
+1*e12    +1*e12    -1*e2    +1*e1  +1*e123  +-1*e31   +1*e23       -1    -1*e3
+1*e123  +1*e123   +1*e23    1*e31   +1*e12    -1*e1    -1*e2    -1*e3       -1
""".split())

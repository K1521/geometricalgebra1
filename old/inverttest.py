

def inv1(bas1,bas2):

    invert=0
    count=bas1.bit_count()
    for b1,b2 in zip(bin(bas1)[:1:-1],bin(bas2)[:1:-1]):
        #this is also possible with shift operator
        #but by preconverting this should be faster for high dimensions
        #assuming shift has O(n) complexity instead of O(1)
        #this inversion count should have O(n) complexity
        #with n=max(self.basis.bit_length(),othe.basis.bit_length())
        count-=b1=="1"
        if b2=="1":
            invert+=count
    return invert%2

def inv2(bas1,bas2):

    invert=0
    count=bas1.bit_count()&1
    for b1,b2 in zip(bin(bas1)[:1:-1],bin(bas2)[:1:-1]):
        #this is also possible with shift operator
        #but by preconverting this should be faster for high dimensions
        #assuming shift has O(n) complexity instead of O(1)
        #this inversion count should have O(n) complexity
        #with n=max(self.basis.bit_length(),othe.basis.bit_length())
        count^=b1=="1"
        if b2=="1":
            invert^=count
    return invert
import random


for i in range(100):
    a=random.getrandbits(20)
    b=random.getrandbits(20)
    e=inv1(a,b)
    if inv2(a,b)!=e:
        print("bad")
        break
else:
    print("good")

for i in range(10):
    bas1=random.getrandbits(50)
    bas2=random.getrandbits(50)
    #print(bin(bas1))
    invert=0
    count=bas1.bit_count()&1
    count=0
    s=""
    for b1,b2 in zip(bin(bas1),bin(bas2)):
        count^=b1=="1"
        s+=str(count)
        if b2=="1":
            invert^=count
    invert2=0
    count=0

    s2=""
    for b1,b2 in zip(bin(bas1),bin(bas2)):
        count^=b1=="1"
        if (b2=="1") and count:
            s2+="1"
            invert2^=1
        else:
            s2+="0"
    bas1acc=bas1
    i=0
    while (xt:=bas1acc>>(2**i))>0:
        #bas1acc=bas1acc^xt
        bas1acc^=xt
        i+=1
        
    invert3=0
    count=0


    for b1,b2 in zip(bin(bas1),bin(bas2)):
        count=b1=="1"
        if (b2=="1") and count:
            invert3^=1

    



    
    #print(bin(x))
    #print(bin(bas2))

    print(bin(bas1))
    print(s)
    print(bin(bas1acc))
    #print(bin(x&(bas2)))
    #print(s2)
    print(invert,invert2,invert3)
    print()
    #print((x&bas2).bit_count()%2)
def sxor(a,b):
    return "".join("10"[a==b]for a,b in zip(a,b))
def leftShift(text,n):
    return text[n:] + text[:n]
def rightShift(text,n):
    return text[-n:] + text[:-n]
#print(bin(bas1))
#print(leftShift(sxor(basi1str,s),0)[1:])
#print(bin(2**bas1.bit_length()+bas1-1)[:1:-1])
#print(bin(bas1^(bas1<<1))[:1:-1])
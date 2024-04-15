

def blademul6(b1,b2):
    bas1acc=b1^(b1.bit_count()&1)#this is here because b1 is backwards
    i=1
    l=min(b1.bit_length(),b2.bit_length())
    mask=(2<<l)-1
    while i<=l:
        bas1acc^=(bas1acc<<i)&mask
        i<<=1
    invert=(bas1acc&b2).bit_count()&1
    return invert
def blademul6reverse(b1,b2):
    bas1acc=b1>>1
    i=1
    while True:
        shifted=bas1acc>>i
        if shifted==0:
            break
        bas1acc^=shifted
        i<<=1
    invert=((bas1acc&b2).bit_count()&1)
    return invert
def blademul6reverse2(b1,b2):
    bas1acc=b1
    for i in range(b1.bit_length().bit_length()):#lol
        bas1acc^=(bas1acc>>(1<<i))
        #print(bas1acc>>(1<<i))
    #while True:
    #    shifted=bas1acc>>i
    #    if shifted==0:
    #        break
    #    bas1acc^=shifted
    #    i<<=1
    invert=(bas1acc&b2).bit_count()&1
    return invert
e=[1<<i for i in range(10)]

e13=e[1]|e[3]
e24=e[2]|e[4]
print(blademul6(e13,e24))


def binit(x,l=8):
    return [t=="1"for t in f"{x:{l}b}"]
def reverseint(x,l=8):
    return int(f"{x:{l}b}"[::-1],2)
print(binit(5))
def blademulsimple1(b1,b2):
    count=False
    invert=False
    for a,b in zip(binit(b1)[::-1],binit(b2)[::-1]):
        count^=b
        if a:
            invert^=count

    return invert
n=8
import numpy as np 
ans=[["0"]*n for i in range(n)]
ansnum = np.zeros((n,n))

#isorted=sorted(list(range(n)),key=int.bit_count)
#print(isorted)
#isorted=[1,2,4,3,5,6,7]
isorted=list(range(n))
eq = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        #if i & j:continue
        ansnum[i][j]=(blademul6reverse(isorted[i],isorted[j]))+1
        if not blademul6reverse(i,j)==blademul6(i,j):
            if not i&j:
                print(":(")
                exit()
            eq[i][j]=1


import matplotlib.pyplot as plt
plt.imshow(ansnum, interpolation='none')
plt.show()
plt.imshow(eq, interpolation='none')
plt.show()
exit()

import timeit
import random


results = {'blademul6': [], 'blademul6reverse2': [], 'blademul6reverse': []}
for length in range(1, 1001,10):
    tests = []
    for _ in range(1000):
        a = random.getrandbits(length)
        b = random.getrandbits(length)
        b = b ^ (a & b)
        tests.append((a, b))

    for fun in (blademul6, blademul6reverse, blademul6reverse2):
        def f():
            for a, b in tests:
                fun(a, b)
 
        results[fun.__name__].append(timeit.timeit(f, number=100))

# Plot the results
plt.figure(figsize=(10, 6))
for name, times in results.items():
    plt.plot(range(1, 101), times, label=name)

plt.title('Performance of Functions vs Length of Binary Numbers')
plt.xlabel('Length of Binary Numbers')
plt.ylabel('Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()






def basedecode(basis):
    return tuple(i for i,x in enumerate(bin(basis)[:1:-1],0)if x=="1")
print(basedecode(8))


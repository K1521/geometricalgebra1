


b1=int("1010",2)
b2=int("0101",2)
print(f"{b1:4b}")
bas1acc=b1
i=1
while True:
    shifted=bas1acc>>i
    if shifted==0:
        break
    bas1acc^=shifted
    i<<=1
    #print(f"{bas1acc:4b}")
    
print(f"{bas1acc:4b}")
invert=(bas1acc&b2).bit_count()&1
print(invert)
def blademul(b1,b2):
    bas1acc=b1
    i=1
    while True:
        shifted=bas1acc>>i
        if shifted==0:
            break
        bas1acc^=shifted
        i<<=1
    return (bas1acc&b2).bit_count()&1

n=64
import numpy as np 
ansnum = np.zeros((n,n))

for i in range(n):
    for j in range(n):
        if i & j:
            continue
        ansnum[i][j]=(blademul(i,j))+1
    


from matplotlib import pyplot as plt
#import matplotlib.pyplot as plt
plt.imshow(ansnum, interpolation='none')
plt.show()

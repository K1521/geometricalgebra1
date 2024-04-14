import random

bas1=random.getrandbits(50)

bas1acc=bas1

i=0
while (xt:=bas1acc>>(2**i))>0:
    bas1acc=bas1acc^xt
    #bas1acc^=xt
    i+=1
print(bas1acc)

bas1acc=bas1
i=0
while (xt:=bas1acc>>(2**i))>0:
    #bas1acc=bas1acc^xt
    bas1acc^=xt
    i+=1
print(bas1acc)

print(b"a"<<1)
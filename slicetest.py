

print(list(range(10,1,-1)))

def basetotup(basis):
    return tuple(i for i,x in enumerate(bin(basis)[:1:-1],0)if x=="1")

def basetotup2(basis):
    return tuple(i for i in range(basis.bit_length()) if basis&(1<<i))


x=int("1100001",2)
print(basetotup(x))
print(basetotup2(x))
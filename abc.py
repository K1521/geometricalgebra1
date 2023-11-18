


print("Hello World")

c=1
print(1,2)
for i in range(3,1000,2):
    
    for j in range(3,int(i**0.5+2),2):
        if i%j==0:
            break
    else:
        c+=1
        print(c,i)
        
#https://www.youtube.com/watch?v=eCDFA02SiIA

from time import perf_counter,time


def solution(array,n):
    result=[]
    offset=n-1
    while l:
        offsetneu=(offset-len(l))%n
        result.extend(l[offset::n])
        del l[offset::n]
        offset=offsetneu
    return result

k=1000000
n=3
l=list(range(1,k+1))
start=perf_counter()
sol=solution(l,n)
print(perf_counter()-start)
print(sol[-100:])


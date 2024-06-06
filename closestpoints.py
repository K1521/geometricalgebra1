


#(p+a*norm)


#avgpoint=sum(pi+a*ni)/n=(sum(pi)+a*sum(ni))/n

#sum((pi+a*ni-avgpoint)**2)
sum((pi+a*ni-avgpoint)**2)
sum((pi+p_+a*(ni-n_))**2)

#(p1x+a*n2x-avgpointx)**2+(p2x+a*n2x-avgpointx)**2 ...

#2*(p1x+a*n2x-avgpointx)*n2x


#sum{i in 1..n}(2*(p_i+a*n_i-avgpoint)*n_i)=0
avgpoint=sum{i in 1..n}(p_i-a*n_i)

2*sum(pi*ni+a*ni*ni-avgpoint*ni)
2*sum(pi*ni+a*ni*ni-(sum(pj)*ni+a*sum(nj)*ni)*1/n)
2*sum(pi*ni+a*ni*ni-(sum(pi)+a*sum(ni))*ni*1/n)

#sum(2*(pi+a*ni-avgpoint)*ni)=0

sum(2*(pi-p_+a*(ni-n_))*(ni-n_))=0
sum((pi-p_)*(ni-n_)+a*(ni-n_)*(ni-n_))=0
sum((pi-p_)*(ni-n_))+a*sum((ni-n_)*(ni-n_))=0

-sum((pi-p_)*(ni-n_))/sum((ni-n_)*(ni-n_))=a

sum((pi-p_+a*(ni-n_))**2)

sum((pi-p_)**2+a*a*(ni-n_)**2+2*a*(pi-p_)*(ni-n_))
sum((pi-p_)**2)+sum(a*a*(ni-n_)**2)+sum(2*a*(pi-p_)*(ni-n_))
sum((pi-p_)**2)+a*a*sum((ni-n_)**2)+2*a*sum((pi-p_)*(ni-n_))

B=sum((pi-p_)*(ni-n_))
C=sum((ni-n_)*(ni-n_))=sum((ni-n_)**2)
a=-B/C
sum((pi-p_)**2)+a*a*C+2*a*B
sum((pi-p_)**2)+B**2/C-2*B**2/C
sum((pi-p_)**2)-B**2/C




A=sum((pi-p_)**2)
A=sum(pi**2)-n*p_**2

B=sum((pi-p_)*(ni-n_))=sum(pi*ni)-p_*sum(ni)-sum(pi)*n_+n*p_*n_
B=sum((pi-p_)*(ni-n_))=sum(pi*ni)-2*n*p_*n_+n*p_*n_
B=sum((pi-p_)*(ni-n_))=sum(pi*ni)-n*p_*n_

C=sum((ni-n_)**2)
C=sum(ni**2)-2*sum(ni)*n_+n*n_**2
C=sum(ni**2)-2*n*n_*n_+n*n_**2
C=sum(ni**2)-n*n_**2


avgpoint=sum(pi+a*ni)/n=p_+a*n_=p_-B/C*n_
error=sum((pi+a*ni-avgpoint)**2)=A-B**2/C
a=-B/C
A=sum(pi**2)-n*p_**2=sum((pi-p_)**2)
B=sum((pi-p_)*(ni-n_))=sum(pi*ni)-n*p_*n_
C=sum((ni-n_)**2)=sum(ni**2)-n*n_**2
n_=sum(ni)/n
p_=sum(pi)/n




avgpoint=sum(pi+a*nmi)/n=p_+a*nm_
e=sum((pi+a*ni-avgpoint)**2)
=sum(((pi-p_)+a*(ni-nm_))**2)=sum((pi-p_)**2)+a**2*sum((ni-nm_)*(ni-nm_))+2*a*sum((pi-p_)*(ni-nm_))=A+2*a*B+a**2*C=A+2*-B/C*B+B**2/C=A-B**2/C
0=sum(2*((pi-p_)+a*(ni-nm_))*(ni-nm_))
0=sum(((pi-p_)+a*(ni-nm_))*(ni-nm_))
0=sum((pi-p_)*(ni-nm_))+a*sum((ni-nm_)*(ni-nm_))
a=-sum((pi-p_)*(ni-nm_))/sum((ni-nm_)*(ni-nm_))
a=-B/C
A=sum((pi-p_)**2)=sum(pi**2)+n*p_**2-2*sum(pi-p_)=sum(pi**2)+n*p_**2
B=sum((pi-p_)*(ni-nm_))=sum(pi*ni)-sum(pi)*nm_-p_*sum(ni)+p_*nm_*n=sum(pi*ni)-n*p_*(nm_+n_)
C=sum((ni-nm_)*(ni-nm_))=sum(ni**2)+n*nm_**2-2*sum(ni*nm_)=sum(ni**2)+n*nm_**2-2*n*n_*nm_

avgpoint=sum(pi+a*nmi)/n=p_+a*nm_
e=A-B**2/C
a=-B/C
A=sum((pi-p_)**2)=sum(pi**2)+n*p_**2-2*sum(pi-p_)=sum(pi**2)+n*p_**2=sum(pi**2)+sum(pi)**2/n
#B=sum((pi-p_)*(ni-nm_))=sum(pi*ni)-sum(pi)*nm_-p_*sum(ni)+p_*nm_*n#fehler=sum(pi*ni)-n*p_*(nm_+n_)=sum(pi*ni)-sum(pi)*sum(nmi+ni)/n
C=sum((ni-nm_)*(ni-nm_))=sum(ni**2)+n*nm_**2-2*sum(ni*nm_)=sum(ni**2)+n*nm_**2-2*n*n_*nm_=sum(ni**2)+sum(nmi)**2/n-2*sum(nmi)*sum(ni)/n
A=sum(pi**2)+sum(pi)**2/n
#B=sum(pi*ni)-sum(pi)*sum(nmi+ni)/n
B=sum(pi*ni)-sum(pi)*sum(ni)/n
C=sum(ni**2)+sum(nmi)**2/n-2*sum(nmi*ni)/n



avgpoint=sum(pi)/n+a*sum(nmi)/n
e=A-B**2/C
a=-B/C
A=sum(pi**2)+sum(pi)**2/n
#B=sum(pi*ni)-sum(pi)*sum(nmi+ni)/n
B=sum(pi*ni)-sum(pi)*sum(ni)/n
C=sum(ni**2)+sum(nmi)**2/n-2*sum(nmi*ni)/n

e2=sum((pi+a*nmi-avgpoint)**2)
a=-sum((pi-p_)*(nmi-nm_))/sum((nmi-nm_)*(nmi-nm_))
A2=A
B2=sum(pi*nmi)-n*p_*nm_
C2=sum((nmi-nm_)**2)=sum(nmi**2)+sum(nmi)**2/n-2*sum(nmi)**2/n
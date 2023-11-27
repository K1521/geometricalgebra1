from blademul6 import sortgeo,algebra
import sympy

dcga=algebra(8,2)


#algebra declaration
multivec=sortgeo(dcga)
dcga.bladenames="1,2,3,4,6,7,8,9,5,10".split(",")

e1,e2,e3,e4,e6,e7,e8,e9,e5,e10=multivec.monoblades()

eo1=(e5-e4)/2
eo2=(e10-e9)/2
ei1=e4+e5
ei2=e9+e10


def point1(x,y,z):
    return e1*x+e2*y+e3*z+ei1*((x*x+y*y+z*z)/2)+eo1
def point2(x,y,z):
    return e6*x+e7*y+e8*z+ei2*((x*x+y*y+z*z)/2)+eo2

def point(x,y,z):
    return point1(x,y,z).outer(point2(x,y,z))

eo=eo1.outer(eo2)
ei=ei1.outer(ei2)
Txx=e6.outer(e1)
Tyy=e7.outer(e2)

def CGA1_Plane(x,y,z,h):
    vec=(x*e1+y*e2+z*e3)
    #return vec/sympy.sqrt(vec.inner(vec).toscalar())+h*ei1
    return vec+h*sympy.sqrt(vec.inner(vec).toscalar())*ei1
    #return vec+ei1
def CGA2_Plane(x,y,z,h):
    vec=(x*e6+y*e7+z*e8)
    #return vec/sympy.sqrt(vec.inner(vec).toscalar())+h*ei2
    return vec+h*sympy.sqrt(vec.inner(vec).toscalar())*ei2
    #return vec+ei2
#CGA2_Plane = { Normalize(_P(1)*e6 + _P(2)*e7 + _P(3)*e8) + _P(4)*ei2 }
def Plane(x,y,z,h):
    return CGA1_Plane(x,y,z,h)^CGA2_Plane(x,y,z,h)
#Plane = {
# CGA1_Plane(_P(1),_P(2),_P(3),_P(4))^CGA2_Plane(_P(1),_P(2),_P(3),_P(4))
#}

T1=-ei
Tt2=eo2.outer(ei1)+ei2.outer(eo1)
Tt4=-4*eo

def toroid(R,r):
    dSq=R*R-r*r
    return Tt4+2*Tt2*dSq+T1*dSq*dSq-4*R*R*(Txx+Tyy)


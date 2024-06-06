
from algebra.algebrabase import SimpleAlgebraBase
import numpy as np

class xyzderiv(SimpleAlgebraBase):#this class is a auto differentiator for 3 variables
    def convert(self,x):
        if isinstance(x,xyzderiv):
            return x
        return xyzderiv(x,[0,0,0])

    def __init__(self,f,df):
        self.f=f
        self.df=df#df should be al list df=[dx,dy,dz]
    
    def mul(s,o):
        return xyzderiv(s.f*o.f,[s.f*odf+o.f*sdf for sdf,odf in zip(s.df,o.df)])
    def add(s,o):
        return xyzderiv(s.f+o.f,[odf+sdf for sdf,odf in zip(s.df,o.df)])
    def sub(s,o):
        return xyzderiv(s.f-o.f,[odf-sdf for sdf,odf in zip(s.df,o.df)])
    def __neg__(s):
        return xyzderiv(-s.f,[-sdf for sdf in s.df])
    def __pow__(s,e):
        #g(f(x))'=g'(f(x))*f'(x) g(x)=x**e g'(x)=e*x**(e-1) 
        if   e == 0:return s.convert(1)
        elif e == 1:return s  # Any base raised to the power of 1 is itself
        else:
            #g'(f(x))=e*f**(e-1)  f=s.f
            gf=e*s.f**(e-1) 
            return xyzderiv(s.f**e,[sdf*gf for sdf in s.df])
    def __abs__(s):
        sgn=np.sign(s.f)
        return xyzderiv(abs(s.f),[sgn*sdf for sdf in s.df])

if __name__=="__main__":
    import sympy as sy
    x,y,z=(xyzderiv(x,d)for x,d in zip(sy.symbols("x y z"),[[1,0,0],[0,1,0],[0,0,1]]))
        # Define the polynomial
    polynomial = x**2 + 2*y + 3*z**3

    # Compute the derivatives
    df_dx = polynomial.df[0]
    df_dy = polynomial.df[1]
    df_dz = polynomial.df[2]

    print("Derivative with respect to x:", df_dx)
    print("Derivative with respect to y:", df_dy)
    print("Derivative with respect to z:", df_dz)

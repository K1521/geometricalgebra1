class inter3d:
    def __init__(self,coeffs=None):
        self.coeffs=coeffs or dict()
    def __rmul__(s,o):return s*o
    def __mul__(s,o):
        if not isinstance(o,inter3d):
            #if o==0:
            #    return interpoly()
            return s*inter3d({(0,0,0):o})


        coeffs=dict()
        for es,cs in s.coeffs.items():
            for eo,co in o.coeffs.items():
                e=tuple(a+b for a,b in zip(es,eo))
                c=coeffs.get(e,0)
                coeffs[e]=cs*co+ c if c else cs*co
        return inter3d(coeffs)
    def __radd__(s,o):return s*o
    def __add__(s,o):
        if not isinstance(o,inter3d):
            return s+inter3d({(0,0,0):o})
        
        small, large = sorted([s.coeffs, o.coeffs], key=len)
        result = large.copy()
        # Update values from the smaller dictionary
        for key, value in small.items():
            r=result.get(key, 0) 
            result[key] = value+ r if r else value
        
        return inter3d(result)
    def __sub__(s,o):return s+(-1)*o

    def __pow__(s,n):
        r=1
        for i in range(n):
            r*=s
        return r

    def tointer(self):#maybe it would be possible to do something like i**2-i**3 to eliminate i**3 and reduce i**2 because i**2>i**3 if i=[0,1]
        l=0
        h=0
        for es,cs in self.coeffs.items():
            if cs>0:
                h+=cs
            else:
                l+=cs
        cs=self.coeffs.get((0,0,0),0)
        if cs<=0:
            h+=cs
        else:
            l+=cs
        return l,h
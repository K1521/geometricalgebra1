


from algebra.algebrabase import SimpleAlgebraBase

class Termdag(SimpleAlgebraBase):
    def convert(self,x):
        if isinstance(x,Termdag):
            return x
        if x==0:
            pass
        return Termdag(str(x))
    
    
    _next_id = 1
    def __init__(self,display,parents=None) -> None:
        self.id = Termdag._next_id  
        Termdag._next_id += 1

        if self.id==477:
            pass

        self.parents=parents or []
        self.children=[]
        self.display=display
        for p in self.parents:
            p.children.append(self)
    
    def __str__(self) -> str:
        return self.display#+ "id"+str(self.id)
    
    def removechild(self,x):
        self.children.remove(x)
        x.parents.remove(self)
    def removeparent(self,x):
        self.parents.remove(x)
        x.children.remove(self)
    def addchild(self,x):
        self.children.append(x)
        x.parents.append(self)
    def addparent(self,x):
        self.parents.append(x)
        x.children.append(self)
        
    def add(a, b) -> SimpleAlgebraBase:
        if a.display=="0":return b
        return Termdag("+",[a,b])
    def sub(a, b) -> SimpleAlgebraBase:
        return Termdag("-",[a,b])
    def mul(a, b) -> SimpleAlgebraBase:
        if a.display=="1":return b
        return Termdag("*",[a,b])
    def div(a, b) -> SimpleAlgebraBase:
        return Termdag("/",[a,b]) 
    
    def __eq__(self, other):
        return self.id==other.id
    
    def __hash__(self):
        return hash(self.id)



def collectall(root):
    visited=set()
    def collecth(n):
        if n in visited:
            return
        visited.add(n)
        for x in n.parents:#+n.children:
            collecth(x)
    
    collecth(root)
    return visited

def symplifyall(root):
    visited=set()
    seen=dict()
    def collecth(n):
        if n in visited:
            return
        visited.add(n)

    
        for x in n.parents[:]:
            collecth(x)
        
        
        # for p in n.parents[:]:
        #     if p.display==n.display and len(p.children)==1:
        #         n.parents.remove(p)
        if len(n.children)==1 and n.children[0].display==n.display and n.display in ("+","*"):
            child=n.children[0]
            #child.parents.remove(n)
            # n.removechild(child)
            # for p in n.parents:
            #     p.children.remove(n)
            #     p.children.append(child)
            #     child.parents.append(p)

            child=n.children[0]
            n.removechild(child)
            for p in n.parents[:]:
                p.removechild(n)
                p.addchild(child)

        

        #         n.parents.remove(p)

    collecth(root)
    return visited


def removedouble(root):
    visited=set()
    seen=dict()
    def collecth(n):
        if n in visited:
            return
        visited.add(n)

    
        for x in n.parents[:]:
            collecth(x)
        
        
        
        
        k=(n.display,tuple(n.parents))
        if k in seen:
            n2=seen[k]
            for c in n.children[:]:
                n.removechild(c)
                n2.addchild(c)
            print(":")
        else:
            seen[k]=n

        #         n.parents.remove(p)

    collecth(root)
    return visited

def prune(root):
    
    visited=set()
    seen=set()
    def collecth(n):
        if n in visited:
            return
        visited.add(n)
        for x in n.parents:#+n.children:
            collecth(x)
        seen.update(n.children)
    
    collecth(root)
    for c in seen-visited:
        for p in c.parents:
            p.children.remove(c)
    
    

    return visited
def layers(root):
    layer={}
    layer[root]=1
    inf=float("inf")
    def collecth(n):
        if n in layer:
            print("??")
            return
        l=max(layer.get(c,inf)for c in n.children)+1
        if l==inf:
            return
        layer[n]=l
        for p in n.parents:
            collecth(p)

    for p in root.parents:
        collecth(p)

    return layer

import algebra.dcga as dcga
point=dcga.point(Termdag("x"),Termdag("y"),Termdag("z"))

t=dcga.toroid(1,.5)^dcga.Plane(1,1,1,0)


o=t.inner(point)
o=Termdag("",[x.magnitude for x in o.lst])
print(len(collectall(o)))
#x,y,z=Termdag("x"),Termdag("y"),Termdag("z")
#o=x+2*x+3*x+4*x

nodes=symplifyall(o)
#removedouble(o)
print(len(nodes))
prune(o)
nodes=collectall(o)
print(len(nodes))

#x,y,z=Termdag("x"),Termdag("y"),Termdag("z")
#nodes=collectall(x*x+y*0.003)
import networkx as nx
import matplotlib.pyplot as plt


# Create a Directed Acyclic Graph
dag = nx.DiGraph()
l=layers(o)
#print(l)
# Add nodes and edges to the graph
for node in nodes:
    dag.add_node(node,layer=l[node])

for node in nodes:
    for parent in node.parents:
        dag.add_edge(parent, node)

# Draw the graph
pos = nx.spring_layout(dag, seed=42,iterations=200)  # You can use different layouts for visualization
#pos=nx.planar_layout(dag)
pos=nx.multipartite_layout(dag,subset_key="layer")
#labels = nx.get_node_attributes(dag, 'layer')
#nx.draw(dag, pos, labels=labels, node_color='lightblue',  arrows=True)
nx.draw(dag, pos, with_labels=True, node_color='lightblue',  arrows=True)
plt.show()
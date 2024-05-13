import numpy as np
import scipy

def cofactor_matrix(matrix):
    n = matrix.shape[0]
    cofactors = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            minor_matrix = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            sign = (-1) ** (i + j)
            cofactors[i, j] = sign * np.linalg.det(minor_matrix)

    return cofactors
def adjugate(matrix):
    cofactors = cofactor_matrix(matrix)
    adjugate_matrix = cofactors.T
    return adjugate_matrix
def line_point_distance(point, line_origin, line_direction):
    # Calculate the vector from line origin to the point
    vector_to_point = np.array(point) - np.array(line_origin)
    
    # Project the vector onto the line direction to find the component along the line
    projected_length = np.dot(vector_to_point, line_direction)
    
    # Calculate the closest point on the line to the given point
    closest_point_on_line = np.array(line_origin) + projected_length * np.array(line_direction)
    
    # Calculate the vector from the point to the closest point on the line
    vector_to_line = np.array(point) - closest_point_on_line
    
    return vector_to_line
def mineigenvalue(A):
    lambd,eig=np.linalg.eig(A)
    return eig[:,np.argmin(lambd)]
def leastsquaresmod(ata):
    A=ata[:,:-1]+ata[:,-1,None]
    b=ata[:,-1,None]
    x,y=np.linalg.lstsq(A.T@A,A.T@b )[0]
    abc=np.array([x,y,x+y-1])
    abc/=np.linalg.norm(abc)
    return abc.T[0]
# Define the line in 3D space

def randomline():
    line_origin = np.random.random(3)  # Origin point of the line
    line_direction = np.random.random(3)*2-1  # Direction vector of the line
#line_direction=(1,0,0)
    line_direction/=np.linalg.norm(line_direction)
    #print(line_direction)
# Normalize direction vector (optional, to ensure it's a unit vector)
    line_direction = np.array(line_direction) / np.linalg.norm(line_direction)

# List of points to calculate vectors to the line
    points = [(0, 0, 0),(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

# Calculate vectors from each point to the line
    vectors_to_line = np.array([line_point_distance(point, line_origin, line_direction) for point in points])
    vectors_to_line+=np.random.random((8,3))*0.3
    vectors_to_line*=np.random.random((8,1))*2-1
# Print the calculated vectors
    #for point, vector in zip(points, vectors_to_line):
    #    print(f"Point: {point} -> Vector to Line: {vector}")
    return line_direction,vectors_to_line


def normalize(vec):
    norm = np.linalg.norm(vec)  # Calculate the Euclidean norm (magnitude) of the vector
    if norm == 0:
        return vec  # Return the input vector if its norm is zero (to avoid division by zero)
    return vec / norm

m1=1
m2=1
m3=1
m4=1
m5=1
m6=1
m7=1
s1,s2,s3,s4,s5,s6,s7=0,0,0,0,0,0,0
#TODO m7 ->avarage of cross vectors wich point in the same direction of the longest cross vec
for i in range(1000):
    line_direction, vectors_to_line = randomline()
    ata=vectors_to_line.T@vectors_to_line
    #print(line_direction)
    #print(normalize(np.cross(ata[0],ata[2])))
    crossmat=np.array([normalize(np.cross(ata[0],ata[1])),
                       normalize(np.cross(ata[1],ata[2])),
                       normalize(np.cross(ata[2],ata[0]))])
    #crossmat=np.array([np.cross(ata[0],ata[1]),np.cross(ata[1],ata[2]),np.cross(ata[2],ata[0])])
    vec=normalize(crossmat[np.argmax(np.linalg.norm(crossmat,axis=1))])

    e1=abs(mineigenvalue(ata).dot(line_direction))
    e2=abs(leastsquaresmod(ata).dot(line_direction))
    m1=min(e1,m1)
    m2=min(e2,m2)
    #print(leastsquaresmod(ata))
    e3=max(abs(crossmat@line_direction))

    #print(crossmat)
    #print(line_direction)
    #print(abs(crossmat@line_direction))
    crossmat=np.array([np.cross(ata[0],ata[1]),np.cross(ata[1],ata[2]),np.cross(ata[2],ata[0])])
    
    vec=normalize(crossmat[np.argmax(np.linalg.norm(crossmat,axis=1))])
    e4=abs(vec.dot(line_direction))  
    m3=min(e3,m3)
    m4=min(e4,m4)


    lengths=np.linalg.norm(ata,axis=1)
    mat=np.delete(ata, np.argmin(lengths), axis=0)
    e5=abs(normalize(np.cross(mat[0],mat[1])).dot(line_direction))
    s1+=e1
    s2+=e2
    s3+=e3
    s4+=e4
    s5+=e5
    m5=min(e5,m5)
    

    crossmat=np.array([(np.cross(ata[0],ata[1])),
                       (np.cross(ata[1],ata[2])),
                       (np.cross(ata[2],ata[0]))])
    #print(crossmat@ata)
    e6=abs(normalize(crossmat[np.argmax(np.linalg.norm(ata@crossmat,axis=1))]).dot(line_direction))
    #e6=0
    s6+=e6
    m6=min(e6,m6)


    crossmat=np.array([np.cross(ata[0],ata[1]),np.cross(ata[1],ata[2]),np.cross(ata[2],ata[0])])
    


    longvec=crossmat[np.argmax(np.linalg.norm(crossmat,axis=1))]
    #print(crossmat@longvec)
    #print(crossmat)
    #print(np.where(crossmat@longvec>0,1,-1))
    crossmat*=np.where(crossmat@longvec>0,1,-1)[:,None]
    #print(crossmat@longvec)
    #print(crossmat)
    #print()
    #longvec=crossmat@longvec
    e7=abs(normalize(crossmat.sum(axis=0)).dot(line_direction))  
    m7=min(e7,m7)
    s7+=e7

    #print()
print(m1,m2,m3,m4,m5,m6,m7)
print(s1,s2,s3,s4,s5,s6,s7)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(0, 0, 0, line_direction[0], line_direction[1], line_direction[2], color='r', arrow_length_ratio=0.05)

# Plot the vectors
for vector in ata:
    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='b', arrow_length_ratio=0.05)
for vector in crossmat:
    ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='g', arrow_length_ratio=0.05)
# Set plot limits
max_range = np.abs(np.concatenate([ata,crossmat])).max()
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()
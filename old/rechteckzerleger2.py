import requests
import io
from PIL import Image
import numpy as np
from itertools import chain
from random import randrange
from scipy.ndimage import label
from tqdm import tqdm
#x = requests.get('https://cdn.pixabay.com/photo/2012/04/18/00/07/silhouette-of-a-man-36181_960_720.png')
#x = requests.get('https://img.freepik.com/premium-vector/fire-flames-firefighter-silhouette-set-vector-transparent-background_733316-1.jpg?w=1380')
x = requests.get('https://c8.alamy.com/compfr/ba5r97/silhouette-d-un-arbre-ba5r97.jpg')
#x = requests.get('https://cdn.vectorstock.com/i/preview-1x/40/10/tree-silhouette-on-white-background-vector-43684010.webp')
#x = requests.get('https://i.stack.imgur.com/IqpIS.png')
#x = requests.get('https://cdn.vectorstock.com/i/preview-1x/40/10/tree-silhouette-on-white-background-vector-43684010.webp')
#print(x.encoding)
#print(x.text)
stream = io.BytesIO(x.content)
img = Image.open(stream)
#img.show()

data=np.array(img)

from matplotlib import pyplot as plt
#print(data[:,:,0].ptp())
#plt.imshow(data[:,:,1], interpolation='nearest')
#plt.show()

#imagemap=data[:,:,1]>128
#print(data.shape)
#imagemap=data[:,:,1]<128
imagemap=np.sum(data,axis=-1)
imagemap=imagemap<np.mean(imagemap)
#imagemap should now be a bitmap:)
print(imagemap)
#plt.imshow(imagemap, interpolation='nearest')
#plt.show()


import pygame
# Initializing Pygame
pygame.init()
surface = pygame.display.set_mode(imagemap.shape[::-1])

#for imagemap in (imagemap,~ imagemap):


emptyrect=(None,None,0,0,0)
while imagemap.max()!=0:
    for event in pygame.event.get() :
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    distancematrix=np.zeros(imagemap.shape,int)#how far could you go down from this pixel
    distancematrix[-1]=imagemap[-1]
    for i in reversed(range(distancematrix.shape[0]-1)):
        distancematrix[i]=(distancematrix[i+1]+1)*imagemap[i]
    #plt.imshow(distancematrix, interpolation='nearest')
    #plt.show()


    #bestrect=(None,None,0,0,0)#x,y,w,h,area
    imagemaplabeld,num_features=label(imagemap)
    #bestrects=[emptyrect]*num_features
    bestrects=[set() for _ in range(num_features)]
    bestrectsfinal=[]
    for y in tqdm(range(imagemap.shape[0])):
        considering=[]#x,down
        #print(y)
        for rectset in bestrects:
            toremove=[]
            for r in rectset:
                if r[1]+r[3]<y:
                    toremove.append(r)
                    bestrectsfinal.append(r)
            rectset-=set(toremove)

        for x,down in enumerate(chain(distancematrix[y],[0])):
            #down=distancematrix[y,x]
            minx=x
            while considering and considering[-1][1]>=down:
                lastx,lastdown=considering.pop()
                minx=lastx
                arealast=(x-lastx)*lastdown


                rectset=bestrects[imagemaplabeld[y,lastx]-1]
                toremove=[]
                newrect=(lastx,y,(x-lastx),lastdown,arealast)
                for r in rectset:
                    if ((newrect[0]+newrect[2])>=r[0]  and (r[0]+r[2]) >=newrect[0] and (newrect[1]+newrect[3])>= r[1] and( r[1]+r[3]) >=newrect[1]):
                        if newrect[4]>=r[4]:
                            toremove.append(r)
                        else:
                            break
                else:
                    rectset-=set(toremove)
                    rectset.add(newrect)


                #labelindex=imagemaplabeld[y,lastx]-1
                #if arealast>bestrects[labelindex][4]:
                    #bestrects[labelindex]=(lastx,y,(x-lastx),lastdown,arealast)
                #if arealast>bestrect[4]:
                    #bestrect=(lastx,y,(x-lastx),lastdown,arealast)
                

            if down>0:
                considering.append((minx,down))
    
    #for bestrect in (r for r in bestrects if r!= emptyrect):
    for bestrect in chain(chain(*bestrects),bestrectsfinal):
        x,y,w,h,area=bestrect
        #print(imagemap[y:y+h,x:x+w].astype(int).sum())
        imagemap[y:y+h,x:x+w]=0
        print(bestrect)


        pygame.draw.rect(surface, (randrange(255),randrange(255),randrange(255)), pygame.Rect(x, y, w, h))
    pygame.display.flip()
    
    #print(area)


while 1:

    for event in pygame.event.get() :
      if event.type == pygame.QUIT:
        pygame.quit()
        exit()
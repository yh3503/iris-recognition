import numpy as np
import math
import cv2
from PIL import  Image

def normalization(img,two_circle):

    x,y,r1,r2=two_circle
    # cv2.circle(img,(x,y),r1,(255,0,0))
    # cv2.circle(img, (x, y), r2, (255, 0, 0))

    M=10   #   M is the partition of angles
    N=10    #   N is the partition of distances
    output=np.zeros((M,N,3),dtype=np.uint8)
    print(output.shape)
    for yy in range(img.shape[0]):
        for xx in range(img.shape[1]):
            rr=np.sqrt((x-xx)**2+(y-yy)**2)
            tt=np.arctan2(yy-y,xx-x)
            if r1<=rr<=r2:
                x_mapped=int((rr-r1)/(r2-r1)*N)
                y_mapped=int((tt+np.pi)/(2*np.pi)*M)
                output[y_mapped,x_mapped]=img[yy,xx]
    return output




    # print(two_circle)
    # theta=list(range(M))
    # inner=list(range(N))
    # theta=[i/M*2*math.pi for i in theta]
    # inner=[(int(x+r1*math.cos(i)),int(y+r1*math.sin(i))) for i in theta]
    # outer=[(int(x+r2*math.cos(i)),int(y+r2*math.sin(i))) for i in theta]
    # theta=np.array(theta)
    # inner = np.array(inner)
    # outer = np.array(outer)
    # new=[]

    # print("theta",theta)

    for i in range(M):
        tmp=[]
        for j in range(N):
            point=(int(inner[i][0]+(outer[i][0]-inner[i][0])*j/N),int(inner[i][1]+(outer[i][1]-inner[i][1])*j/N))
            tmp.append(point)
            # cv2.circle(img,point,1,(0,255,0))
            # cv2.imshow("h", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        new.append(tmp)
    new=np.array(new)
    new=new.astype(int)
    print("new")
    print(new)
    print("end")
    pic=[]
    img2=img.copy()
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            img2[i][j]=(255,255,255)
    # for i in range(img2)
    cv2.resize(img2,(400,300))
    print(img2.shape)


    for i in range(M):
        for j in range(N):
            img2[i,j]=img[new[i][j][0],new[i][j][1]]
    return img2[0:N,0:M]





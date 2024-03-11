import cv2
import os
import time
import numpy as np
from PIL import Image


start_time = time.time()


def find_pupil_coarse(img):
    gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred_image=cv2.GaussianBlur(gray_image,(5,5),0)

    _,thresholded_image=cv2.threshold(blurred_image,70,255,cv2.THRESH_BINARY)
    # cv2.imshow("loaded2", thresholded_image)

    circles=cv2.HoughCircles(thresholded_image,cv2.HOUGH_GRADIENT,dp=1,minDist=45,param1=200,param2=15,minRadius=0,maxRadius=100)
    #找出最大的圆
    max_radius = 0
    max_circle=None
    if circles is not None:

        circles=np.uint16(np.around(circles))
        for circle in circles[0,:]:
            x,y,r=circle
            if r>max_radius:
                max_radius=r
                max_circle=circle
        circles=max_circle
    pupil=img
    if circles is not None:
        circles=circles.astype(int)
        x,y,r=circles
        cv2.circle(img,(x,y),r,(0,255,0),4)
        pupil=img[y-r:y+r,x-r:x+r]
        img2 = cv2.rectangle(img, (x-r-64, y-32), (x-r, y+32), (255, 0, 0), 2)
        img2 = cv2.rectangle(img, (x + r, y - 32), (x + r+64, y + 32), (255, 0, 0), 2)
        # return img2
        if(y>32 and x-r>64 and y+32<280 and x+r+64<320):
            return img[y-32:y+32,x-r-64:x-r],img[y-32:y+32,x+r:x+r+64]
        else: return img,img
    else: return img,img

def is_between_circles(x,y,circle1,circle2):
    x1,y1,r1=circle1
    x2,y2,r2=circle2
    if((x-x1)**2+(y-y1)**2>r1**2 and (x-x2)**2+(y-y2)**2<r2**2): return True
    return False

def is_between_parabolas(x,y,par1,par2):
    a1,b1,c1=par1
    a2,b2,c2=par2
    if(y>a1*(x**2)+b1*x+c1 and y<a2*(x**2)+b2*x+c2): return True
    return False


def find_pupil_later(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    _, thresholded_image = cv2.threshold(blurred_image, 70, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(thresholded_image, cv2.HOUGH_GRADIENT, dp=0.8, minDist=45, param1=1000, param2=15,
                               minRadius=15, maxRadius=63)
    edges = cv2.Canny(blurred_image, threshold1=15,threshold2=100)
    # cv2.imshow(f"loaded{i}",edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 找出最大的圆
    max_radius = 00
    max_circle = None
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            if r > max_radius:
                max_radius = r
                max_circle = circle
        circles = max_circle





    if circles is not None:
        circles = circles.astype(int)
        small_circle=circles
        x, y, r = circles
        img2=cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        edges = cv2.Canny(blurred_image, threshold1=30, threshold2=90)


        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=5, maxLineGap=10)
        upper_eyelid = []
        lower_eyelid = []
        for line in lines:
            x1,y1,x2,y2=line[0]
            tmp=r
            if y1<y-tmp:
                upper_eyelid.append((x1,y1))
            elif y1>y+tmp:
                lower_eyelid.append((x1, y1))
            if y2<y-tmp:
                upper_eyelid.append((x2, y2))
            elif y2>y+tmp:
                lower_eyelid.append((x2,y2))
        upper_eyelid=np.array(upper_eyelid)
        lower_eyelid=np.array(lower_eyelid)
        upper_coeff=np.polyfit(upper_eyelid[:,0],upper_eyelid[:,1],2)
        upper_para=np.poly1d(upper_coeff)
        lower_coeff=np.polyfit(lower_eyelid[:,0],lower_eyelid[:,1],2)
        lower_para=np.poly1d(lower_coeff)
        for xx in range(img.shape[1]):
            y_upper=int(upper_para(xx))
            y_lower=int(lower_para(xx))
            cv2.circle(img,(xx,y_upper),1,(0,255,0),-1)
            cv2.circle(img, (xx, y_lower), 1, (0, 255, 0), -1)
        tmp=110
        gray_image2=gray_image[y-tmp:y+tmp,x-tmp:x+tmp]
        if gray_image2 is not None:
            circles2 = cv2.HoughCircles(gray_image2, cv2.HOUGH_GRADIENT, dp=0.8, minDist=450, param1=5, param2=5,
                                       minRadius=0, maxRadius=200)
            if circles2 is not None:
                circles2 = np.uint16(np.around(circles2))

                for circle in circles2[0,:]:
                    x2,y2,r2=circle
                    big_circle=x2+x-tmp, y2+y-tmp, r2
                    cv2.circle(img2, (x2+x-tmp, y2+y-tmp), r2, (0, 255, 0), 4)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(big_circle is not None ):
                if not (is_between_circles(x,y,small_circle,big_circle) and is_between_parabolas(x,y,upper_coeff,lower_coeff)):

                    img[y,x]=[0,0,0]

    return img

def cal_quaility_descriptor(img):
    # print(img.shape)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    img_fft=np.fft.fft2(gray_image)
    img_fft_shifted=np.fft.fftshift(gray_image)
    magnitude_spectrum=np.abs(img_fft_shifted)
    low_bin=(0,6)
    mid_bin=(6,22)
    high_bin=(22,32)
    F1=np.sum(magnitude_spectrum[low_bin[0]:low_bin[1],low_bin[0]:low_bin[1]])
    F2 = np.sum(magnitude_spectrum[mid_bin[0]:mid_bin[1], mid_bin[0]:mid_bin[1]])
    F3 = np.sum(magnitude_spectrum[high_bin[0]:high_bin[1], high_bin[0]:high_bin[1]])
    return F1+F2+F3,F2/(F1+F3)


databse_folder = './database/'
bmp=[]

for i in range(1,109):
    subfolder_name=f"{i:03}"
    subfolder_path=os.path.join(databse_folder,subfolder_name,"1")

    if os.path.isdir(subfolder_path):
        for root,dirs,files in os.walk(subfolder_path):
            for file in files:
                if file.endswith("bmp"):
                    bmp_file_path=os.path.join(root,file)
                    bmp.append(bmp_file_path)

score=[]
for i in range(len(bmp)):
    bmp[i]=bmp[i].replace('\\',"/")
    img = cv2.imread(bmp[i])

    iris=find_pupil_coarse(img)
    if(iris is None):
        print("can't find")


    score1=cal_quaility_descriptor(iris[0])
    score2=cal_quaility_descriptor(iris[1])
    mean_score=[(score1[0]+score2[0])/2,(score1[1]+score2[1])/2]
    score.append(mean_score)

    width,height=img.shape[:2]
    # print(width,height)

bmp=np.array(bmp)
bmp=bmp.reshape(108,3)
score=np.array(score)


score=score.reshape(108,3,2)
selected=[]

for i in range(108):
    max=0
    max_index=-1
    for j in range(3):
        if(score[i][j][1]>max):
            max=score[i][j][1]
            max_index=j
    selected.append(bmp[i][max_index])

selected=np.array(selected)
for i in range(108):
    if(i==81 or i==82):
        continue
    img = cv2.imread(selected[i])
    iris=find_pupil_later(img)
    if(iris is None):
        print("can't find")
    #
    # cv2.imshow(f"loaded{i}",iris)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



# print(selected)
end_time = time.time()
# Make it < 30s
print(f"Running time: {end_time - start_time} seconds")

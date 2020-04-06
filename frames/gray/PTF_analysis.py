import os,sys
import numpy as np
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import time

pos = np.array([[70.4,64.6],
[69.4,62.3],
[73.1,63.7],
[70.3,62.0],
[73.2,61.4],
[72.4,61.4],
[70.8,61.9],
[71.7,62.0],
[71.7,61.9],
[71.8,62.0],
[72.0,61.8]])

pos1 = np.array([[22.2,47.3],
 [20.6,48.0],
 [24.6,47.9],
 [19.5,48.0],
 [23.0,48.0],
 [21.5,48.0],
 [21.1,48.0],
 [22.1,48.0],
 [22.0,48.0],
 [21.8,48.0],
 [22.0,48.0]])

xsep_dist = 500+pos[0,0] - (120 + pos1[0,0])
ysep_dist = 870+pos[0,1] - (716 + pos1[0,1])
frames = [146,148,149,150,151,152,153,154,155,156,157]

for i in range(0,len(frames)):
    ind = frames[i]
    fn = "frames/"+"frame%03d.png"%ind
    print(fn)
    orig = cv.imread(fn, 0)
    leftview = orig[:1078,1920:]
    topview = orig[:1078,:1920]
    irview = orig[1078:,:1920]
    roi1 = leftview[870:990,500:640]
    roi2 = leftview[716:815,120:180]

    im = leftview
    # create a CLAHE object (Arguments are optional).
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2 = clahe.apply(im)
    cv.imwrite("frames/gray/"+"frame%03d_gray.png"%ind,im)

    # ROI 1 (circular washer)
    roi1=cv.GaussianBlur(roi1, (5, 5), 0)
    img2 = clahe.apply(roi1)
    img2 = cv.equalizeHist(img2)
    cv.circle(img2,(int(pos[i][0]),int(pos[i][1])),25,255,1)
    cv.imwrite("ROI_1_%03d_gray.png"%ind,img2)

##    plt.imshow(img2)
##    plt.plot(pos[i][0],pos[i][1],'ro')
##    plt.show()

    # ROI 2 (foil)
    img2 = clahe.apply(roi2)
    cv.circle(img2,(int(pos1[i][0]),int(pos1[i][1])),5,0,1)
    cv.imwrite("ROI_2_%03d_gray.png"%ind,img2)

##    plt.imshow(img2)
##    plt.plot(pos1[i][0]-pos1[0][0],pos1[i][1]-pos1[0][1],'ro')
##    plt.show()


##plt.figure(0)
##plt.plot( pos[:,0]- pos[0,0], -pos[:,1]+ pos[0,1],'bo--')
##plt.plot(pos1[:,0]-pos1[0,0],-pos1[:,1]+pos1[0,1],'ro--')

plt.figure(1)
time = np.arange(0.,11.)/30
plt.plot(time, pos[:,0]- pos[0,0],'bo--',label="Washer feature")
plt.plot(time, pos1[:,0]-pos1[0,0],'ro--',label="Tape feature")
plt.xlabel("Time (s)")
plt.ylabel("Horizontal motion (px)")
plt.legend(loc=0)

plt.figure(2)
time = np.arange(0.,11.)/30
plt.plot(time, pos[:,1]- pos[0,1],'bo--',label="Washer feature")
plt.plot(time, pos1[:,1]-pos1[0,1],'ro--',label="Tape feature")
plt.xlabel("Time (s)")
plt.ylabel("Vertical motion (px)")
plt.legend(loc=0)
plt.show()


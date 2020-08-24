import numpy as np
import cv2
from matplotlib import pyplot as plt

bgr = cv2.imread("pika.png")

lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Convert BGR to HSV

#hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
#hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # convert the image in gray
blurred = cv2.GaussianBlur(gray, (11, 11), 0) # smoothing (blurring) it to reduce high frequency noise
thresh = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY)[1] # threshold the image to reveal light regions in the
        # perform a series of erosions and dilations to remove
        # any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)




gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
#edges = cv2.Canny(hsv,700,700,apertureSize = 3)

minLineLength = 10
maxLineGap = 10
#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#for x1,y1,x2,y2 in lines[0]:
    #cv2.line(edges,(x1,y1),(x2,y2),(0,255,0),2)


cv2.imwrite('pika_hsv_bgright.png', thresh)

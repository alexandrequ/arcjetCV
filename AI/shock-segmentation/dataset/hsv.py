import numpy as np
import cv2
from matplotlib import pyplot as plt

bgr = cv2.imread("img011.png")
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Convert BGR to HSV

hsv1 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(hsv1, cv2.COLOR_BGR2HSV)

gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(hsv2,700,700,apertureSize = 3)

minLineLength = 10
maxLineGap = 10
#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#for x1,y1,x2,y2 in lines[0]:
    #cv2.line(edges,(x1,y1),(x2,y2),(0,255,0),2)


cv2.imwrite('img811.png',hsv2)
cv2.imwrite('img711.png',hsv1)
#cv2.imwrite('img607.png',bgr)

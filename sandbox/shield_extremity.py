import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

img = cv2.imread('shock_detection/tests/frame_0012_out.png')   # you can read in images with opencv
#img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_color1 = np.asarray([0, 230, 0])
hsv_color2 = np.asarray([1, 255, 1])

mask = cv2.inRange(img, hsv_color1, hsv_color2)

#shield = np.where(img == [0, 255, 0])
coordinates =cv2.findNonZero(mask)
x = [p[0][0] for p in coordinates]
y = [p[0][1] for p in coordinates]
centroid = (int(sum(x) / len(coordinates)), int(sum(y) / len(coordinates)))

print(centroid)



contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#cv2.drawContours(img,contours,-1,(255, 0, 0),5)
cv2.circle(img, centroid, 5, (255, 0, 0), 2)


contours = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(contours)
c = max(cnts, key=cv2.contourArea)
print(c[c[:, :, 0].argmax()])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extLeft = tuple(c[c[:, :, 0].argmin()][0])
cv2.circle(img, extLeft, 8, (0, 255, 0), -1)

print(extLeft)
#print(c[1, 0, 0])
dist_shield_CG = abs(centroid[0]-extLeft[0])
#print(dist_shield_CG)
#indices = np.where(mask!= [0])
#coordinates = zip(indices[0], indices[1])

# Going through every contours found in the image.

# cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# c = max(cnts, key=cv2.contourArea)
#
# extLeft = tuple(c[c[:, :, 0].argmin()][0])
# extRight = tuple(c[c[:, :, 0].argmax()][0])
#
# print(c[:, :, 0].argmin())
# print(c[:,0])
# for idx, i in enumerate(c[:, :, 0]):
#     if i == centroid[0]:
#            print(i)
#            print(tuple(c[c[idx, :, 0].argmin()][0]))

#extLeft_CG = tuple(c[extLocLeft][0])
#extRight_CG = tuple(c[c[:, :, 0].argmax()][0])
#
#
# cv2.drawContours(img, [c], -1, (0, 255, 255), 2)
cv2.circle(img, extLeft, 8, (255, 0, 0), -1)
# cv2.circle(img, extRight, 8, ( 255, 0, 0), -1)

cv2.imshow("Robust", img)
cv2.waitKey(0)
cv2.destroyAllWindows("Robust")

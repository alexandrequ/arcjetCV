import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Frame import getModelProps
from Functions import *

orig = cv.imread('sample11.png',1)

##img = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
##histr = cv.calcHist( [img], None, None, [256], (0, 256));
##imgsize = img.size
##plt.figure(1)
##plt.plot(histr/imgsize,'b-')
##plt.xlim([0,256])
##plt.ylim([0,.005])

# get ROI
##try:
##(c,stingc), ROI, orientation, flowRight,flags = getModelProps(orig,plot=True)
##print(flowRight)

gray = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
flags = classifyImageHist(gray)
c,stingc = contoursHSV(orig,flags,plot=True,draw=False)

cn = getConvexHull(c,600)
stingc = getConvexHull(stingc,1000)
cv.drawContours(orig, cn, -1, (0,255,255), 1)

edges, ROI, orientation, flowRight = getEdgesFromContours(orig,c,stingc,flags,draw=True,plot=False)
cnn = combineEdges(edges[0],edges[1],flowRight)
cv.drawContours(orig, cnn, -1, (0,0,255), 1)

plt.figure(0)
rgb = orig[...,::-1].copy()
plt.subplot(1,1,1),plt.imshow(rgb)
plt.show()

##print(magnus)
##
####cv2.imshow('image',img)
####cv2.waitKey(0)
####cv2.destroyAllWindows()
##
##laplacian = cv2.Laplacian(img[y1:y2,x1:x2],cv2.CV_64F)
##sobelx = cv2.Sobel(img[y1:y2,x1:x2],cv2.CV_64F,1,0,ksize=5)
##abs_sobel64f = np.absolute(sobelx)
##sobel_8u = np.uint8(abs_sobel64f)
##
##sobely = cv2.Sobel(img[y1:y2,x1:x2],cv2.CV_64F,0,1,ksize=5)
##
##plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
##plt.title('Original'), plt.xticks([]), plt.yticks([])
##plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
##plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
##plt.subplot(2,2,3),plt.imshow(sobel_8u,cmap = 'gray')
##plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
##plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
##plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
##
##plt.show()
##
##edges = cv2.Canny(img[y1:y2,x1:x2],20,100)
##cv2.drawContours(orig, [c], -1, 255, 1)
##cv2.rectangle(orig,(x-int(w/4),y-int(h/4)),(x+int(5*w/4),y+int(5*h/4)),(255,255,255),1)
##plt.subplot(121),plt.imshow(orig,cmap = 'jet')
##plt.title('Original Image'), plt.xticks([]), plt.yticks([])
##
##cv2.drawContours(orig, [c], -1, 255, 1)
##plt.subplot(122),plt.imshow(orig[y1:y2,x1:x2],cmap = 'gray')
##plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
##
##plt.show()
##
##mask = cv2.inRange(img, 127, 255)
##output = cv2.bitwise_and(img, img, mask=mask)
##
##ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
##th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
##            cv2.THRESH_BINARY,11,2)
##contours,hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##cnt = cv2.drawContours(th2, contours, -1, (0,255,0), 3)
##
##th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
##            cv2.THRESH_BINARY,11,2)
##
##
##
##titles = ['Original Image', 'Global Thresholding (v = 127)',
##            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
##images = [img, th1, th2, th3]
##
##plt.figure(1)
##for i in range(4):
##    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
##    plt.title(titles[i])
##    plt.xticks([]),plt.yticks([])
##plt.show()

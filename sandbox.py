import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Frame import getModelProps
from Functions import combineEdges

orig = cv.imread('sample4.png',1)
##hsv_ = cv.cvtColor(orig, cv.COLOR_BGR2HSV)
##hsv_=cv.GaussianBlur(hsv_, (5, 5), 0)
##hsv = cv.cvtColor(hsv_, cv.COLOR_RGB2HSV)

##hsv_ = cv.cvtColor(orig, cv.COLOR_BGR2HSV)
##print(np.shape(orig),np.shape(hsv_))
##plt.figure(0)
##plt.imshow(hsv_)
##plt.figure(1)
##plt.imshow(orig)
##plt.show()

img = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
histr = cv.calcHist( [img], None, None, [256], (0, 256));
imgsize = img.size
##sv= histr[50:100].sum()/imgsize
##print("Sting visible",sv,sv>.05)
##print("overexposed", histr[250:].sum()/imgsize > 0.005)
##print("underexposed", histr[150:].sum()/imgsize < 0.005)
##print("saturated",histr[254:].sum()/imgsize > 0.005)
plt.figure(1)
plt.plot(histr/imgsize,'b-')
plt.xlim([0,256])
plt.ylim([0,.005])
##
##plt.figure(0)
##rgb = orig[...,::-1].copy()
##plt.subplot(1,2,1),plt.imshow(rgb)
##plt.subplot(1,2,2),plt.imshow(img)
##plt.show()

# get ROI
##try:
(c,stingc), ROI, orientation, flowRight = getModelProps(orig,plot=False)
print(flowRight)
    
cv.drawContours(orig, c, -1, (0,255,255), 1)

##### Bottom corner
##if abs(diff_top[0]) > 20:
##    pc = c[20,:,:]
##    ind = np.where(stingc[:,0,0] == pc[0,0])[0][0]
##    mt=stingc[:ind+1,0,:]
##    
##    ### linear offset correction
##    delta = stingc[ind,:,:] - pc
##    s = np.linspace(0,1,ind+1)
##    ds = (delta*s[:, np.newaxis]).astype(np.int32)
##    mt -= ds
##else:
##    mt = splineFit()

plt.figure(0)
rgb = orig[...,::-1].copy()
plt.subplot(1,1,1),plt.imshow(rgb)
plt.show()


##except TypeError:
##    print("ROI not found... :(")

##### Centerline
##if flowRight:        
##    centerline = hsv[int(cy)-5:int(cy)+5,x-dx:int(cx),:].sum(axis=0)/10.
##else:
##    centerline = hsv[int(cy)-5:int(cy)+5,int(cx):x+w+dx,:].sum(axis=0)/10.
##    centerline = centerline[::-1]
##
##### edgemetric = V**2 + (256-S)**2 + (180-H)**2
##ind = analyzeCenterlineHSV(centerline)
##hsv *= mask
##H,S,V = hsv[y-dx:y+h+dx,x-dx:x+w+dx,0],hsv[y-dx:y+h+dx,x-dx:x+w+dx,1],hsv[y-dx:y+h+dx,x-dx:x+w+dx,2]
##edgemetric = (V**2 + (256-S)**2 + (180-H)**2)
##
##for i in range(0,len(edgemetric),10):
##    row = edgemetric[i,:]
##    plt.plot([i],[row.argmax()-1],'ro')
##plt.show()



print(magnus)

##cv2.imshow('image',img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()

laplacian = cv2.Laplacian(img[y1:y2,x1:x2],cv2.CV_64F)
sobelx = cv2.Sobel(img[y1:y2,x1:x2],cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx)
sobel_8u = np.uint8(abs_sobel64f)

sobely = cv2.Sobel(img[y1:y2,x1:x2],cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()

edges = cv2.Canny(img[y1:y2,x1:x2],20,100)
cv2.drawContours(orig, [c], -1, 255, 1)
cv2.rectangle(orig,(x-int(w/4),y-int(h/4)),(x+int(5*w/4),y+int(5*h/4)),(255,255,255),1)
plt.subplot(121),plt.imshow(orig,cmap = 'jet')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

cv2.drawContours(orig, [c], -1, 255, 1)
plt.subplot(122),plt.imshow(orig[y1:y2,x1:x2],cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

mask = cv2.inRange(img, 127, 255)
output = cv2.bitwise_and(img, img, mask=mask)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
contours,hierarchy = cv2.findContours(th2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = cv2.drawContours(th2, contours, -1, (0,255,0), 3)

th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)



titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

plt.figure(1)
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

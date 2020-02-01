import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Frame import getModelROI

# Load an color image in grayscale
orig = cv.imread('sample4.png',1)

# get ROI
ROI, boxes, orientation, flowRight = getModelROI(orig,plot=True)
print(flowRight)

### Centerline
if flowRight:        
    centerline = hsv[int(cy)-5:int(cy)+5,x-dx:int(cx),:].sum(axis=0)/10.
else:
    centerline = hsv[int(cy)-5:int(cy)+5,int(cx):x+w+dx,:].sum(axis=0)/10.
    centerline = centerline[::-1]

### edgemetric = V**2 + (256-S)**2 + (180-H)**2
ind = analyzeCenterlineHSV(centerline)
hsv *= mask
H,S,V = hsv[y-dx:y+h+dx,x-dx:x+w+dx,0],hsv[y-dx:y+h+dx,x-dx:x+w+dx,1],hsv[y-dx:y+h+dx,x-dx:x+w+dx,2]
edgemetric = (V**2 + (256-S)**2 + (180-H)**2)

for i in range(0,len(edgemetric),10):
    row = edgemetric[i,:]
    plt.plot([i],[row.argmax()-1],'ro')
plt.show()



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


##def find_blobs(img):
##    # Setup SimpleBlobDetector parameters.
##    params = cv2.SimpleBlobDetector_Params()
##     
##    # Change thresholds
##    params.minThreshold = 100;
##    params.maxThreshold = 5000;
##     
##    # Filter by Area.
##    params.filterByArea = True
##    params.minArea = 200
##     
##    # Filter by Circularity
##    params.filterByCircularity = False
##    params.minCircularity = 0.785
##     
##    # Filter by Convexity
##    params.filterByConvexity = False
##    params.minConvexity = 0.87
##     
##    # Filter by Inertia
##    #params.filterByInertia = True
##    #params.minInertiaRatio = 0.01
##
##    # Set up the detector with default parameters.
##    detector = cv2.SimpleBlobDetector(params)
##     
##    # Detect blobs.
##    keypoints = detector.detect(img)
##    print keypoints
##      
##    # Draw detected blobs as red circles.
##    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
##    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]),
##            (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
##    cv2.imwrite("blobs.jpg", im_with_keypoints); 

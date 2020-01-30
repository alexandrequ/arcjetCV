import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load an color image in grayscale
img = cv2.imread('sample2.png',0)
orig = cv2.imread('sample2.png',1)
ret,th1 = cv2.threshold(img,210,255,cv2.THRESH_BINARY)

contours,hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    # draw in blue the contours that were found
    

    # find the biggest contour (c) by the area
    c = max(contours, key = cv2.contourArea)
    print(c)
    cv2.drawContours(img, [c], -1, 255, 1)
    x,y,w,h = cv2.boundingRect(c)

    # draw the biggest contour (c) in green
    #cv2.rectangle(img,(x-int(w/4),y-int(h/4)),(x+int(5*w/4),y+int(5*h/4)),(0,255,0),2)
    x1,y1 = (x-int(w/4),y-int(h/4))
    x2,y2 = (x+int(5*w/4),y+int(5*h/4))
    mu = cv2.moments(c)
    th = 0.5*np.arctan2(-2*mu['mu11'],mu['mu20']-mu['mu02'])
    cx,cy = mu['m10']/mu['m00'],mu['m01']/mu['m00']
    mean,eigenvectors,eigenvalues = cv2.PCACompute2(np.float32(c)[:,0,:],np.empty((0)))
    cntr = (int(mean[0,0]), int(mean[0,1]))
    angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0]) 
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    plt.plot([cntr[0],p1[0]],[cntr[1],p1[1]],'r-')
    plt.plot([cntr[0],p2[0]],[cntr[1],p2[1]],'b-')
    flow = x+w/2. - cx 
    print(flow)
# show the images
plt.imshow(img)
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

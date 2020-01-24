import os,sys
import numpy as np
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def findChessCorners(img,pattern_shape):
    found, corners = cv.findChessboardCorners(img, pattern_shape)
    if found:
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        return corners.reshape(-1, 2)
    else:
        return None

def getBlobs(fn):
    # Read image
    frame = cv.imread(fn, cv.IMREAD_GRAYSCALE)
    im=cv.GaussianBlur(frame, (3, 3), 0)

    # Set up the SimpleBlobdetector with default parameters.
    params = cv.SimpleBlobDetector_Params()
     
    # Change thresholds
    params.minThreshold = 50;
    params.maxThreshold = 127;
     
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 2000
     
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
     
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.5
     
    # Filter by Inertia
    params.filterByInertia =True
    params.minInertiaRatio = 0.5
     
    detector = cv.SimpleBlobDetector_create(params)
     
    # Detect blobs.
    keypoints = detector.detect(im)
     
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    [isFound, centers] = cv.findCirclesGrid(im, (2,9), flags = cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING,  blobDetector=detector)
    cv.drawChessboardCorners(im_with_keypoints, (2,9), centers, isFound)

    return im_with_keypoints, isFound, centers

# Show keypoints
im_with_keypoints, isFound, centers= getBlobs("2x9.png")
print(isFound)
plt.imshow(im_with_keypoints)
plt.show()
    



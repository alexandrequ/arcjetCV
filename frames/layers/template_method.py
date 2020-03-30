import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from glob import glob

folder = "./"
fnames = folder + "ILPreTest-75um*.JPG"  # default
r1,r2 = 25,143

paths = glob(fnames)

folder = "./templates/"
toes = glob(folder + "*toe*.jpg")
warps = glob(folder + "*warp*.jpg")

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def multiTemplateCorr(src, temps, srcThresh=50, iters=1):
    out = np.ones(np.shape(src))
    for temp in temps:
        #edge padding sizes
        h,w = np.shape(temp)
        top,left = int((h-1)/2),int((w-1)/2)
        bottom,right = h-1-top,w-1-left
        mt = cv.matchTemplate(src.astype(np.float32), temp.astype(np.float32), cv.TM_CCORR_NORMED)

        out[top:-bottom,left:-right] *= mt
    out[src<srcThresh] = 0
    out = cv.normalize(out,None,0,255,cv.NORM_MINMAX)
    out[out>=250] = 0

    return out.astype(np.uint8)

def postProcessCorr(src,iterations=5):
    # renormalize histogram
    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i in range(0,iterations):    
        out = (src.astype(np.float32))**2
        out = cv.normalize(out,None,0,255,cv.NORM_MINMAX)
        out = cv.equalizeHist(out.astype(np.uint8))
        out[out==255] =0
    return out

toe_temps=[]
for toe in toes:
    toe_temps.append(cv.imread(toe,0))

warp_temps=[]
for warp in warps:
    warp_temps.append(cv.imread(warp,0))

for path in paths:
    rawimage = cv.imread(path,0)
##    plt.imshow(rawimage)
##    plt.show()

    ### Region of interest
    roi0 = rawimage.copy()[r1:r2,:]

    ### Smoothing
    roi = cv.medianBlur(roi0, 5)
    
    ### Adaptive contrast filter
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    roi_eq = clahe.apply(roi)

##    plt.figure()
##    plt.imshow(toe_temps[1])
##    plt.show()

    ### Apply template correlation 
    dst = multiTemplateCorr(roi_eq, toe_temps)
    dst = postProcessCorr(dst)

    ### Apply thresholding
    ret,thresh = cv.threshold(dst,200,255,cv.THRESH_BINARY)

    ### Plot results
    plt.figure(1)
    plt.imshow(roi_eq)
    plt.figure(2)
    plt.imshow(dst)
    plt.figure(3)
    plt.imshow(thresh)

    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()
    params.blobColor =255
    
    # Change thresholds
    params.minThreshold = 200;
    params.maxThreshold = 255;
    params.thresholdStep = 5;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 140

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.25

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.2

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.001

    # Min distance
    params.minDistBetweenBlobs = 10

    # Detect blobs
    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(dst)
    print(len(keypoints))
    im_with_keypoints = cv.drawKeypoints(roi0, keypoints, roi0, (0,150,0),
                                          cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Write to file
    folder,name,ext = splitfn(path)
    cv.imwrite(folder+'/'+name+'_marked'+'.png',im_with_keypoints)
    print(folder+'/'+name+'_marked'+'.png')

    # Plot marked locations
    plt.figure(9)
    plt.imshow(im_with_keypoints)
    plt.show()

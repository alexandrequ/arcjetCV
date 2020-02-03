import os,sys
import numpy as np
import cv2 as cv
from glob import glob
from scipy.interpolate import interp1d
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

def getContourAxes(contour):
    sz = len(contour)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = contour[i,0,0]
        data_pts[i,1] = contour[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
    angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    
    return mean[0],eigenvectors, eigenvalues,angle

def getOrientation(c):
    mu = cv.moments(c)
    x,y,w,h = cv.boundingRect(c)
    th = 0.5*np.arctan2(2*mu['mu11'],mu['mu20']-mu['mu02'])
    cx,cy = mu['m10']/mu['m00'],mu['m01']/mu['m00']
    flowRight = x+w/2. - c[c[:,0,1].argmin(),0,0] <0
    
    return th,cx,cy,(x,y,w,h),flowRight

def classifyImageHist(gray,verbose=False,stingpercent=.05,modelpercent=.005):    
    histr = cv.calcHist( [gray], None, None, [256], (0, 256));
    imgsize = gray.size
    modelvis = (histr[12:250]/imgsize > 0.15).sum() != 1
    modelvis *= histr[50:250].sum()/imgsize > modelpercent
    stingvis= histr[50:100].sum()/imgsize > stingpercent
    overexp = histr[250:].sum()/imgsize > modelpercent
    underexp= histr[150:].sum()/imgsize < modelpercent
    saturated = histr[254:].sum()/imgsize > modelpercent
    if verbose:
        print("Model visible",modelvis)
        print("Sting visible",stingvis)
        print("overexposed", overexp)
        print("underexposed", underexp)
        print("saturated",saturated)
    return {'underexp':underexp,
            'overexp':overexp,
            'saturated':saturated,
            'stingvis':stingvis,
            'modelvis':modelvis}

def getBlobDetector():
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
    return detector

def getBlobGrid(fn,detector):
    # Read image
    frame = cv.imread(fn, cv.IMREAD_GRAYSCALE)
    im=cv.GaussianBlur(frame, (3, 3), 0)
     
    # Detect blobs.
    keypoints = detector.detect(im)
     
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    [isFound, centers] = cv.findCirclesGrid(im, (2,9), flags = cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING,  blobDetector=detector)
    cv.drawChessboardCorners(im_with_keypoints, (2,9), centers, isFound)

    return im_with_keypoints, isFound, centers

def analyzeCenterlineHSV(cl):
    H,S,V = cl[:,0],cl[:,1],cl[:,2]
    edgemetric = V**2 + (256-S)**2 + (180-H)**2
    ind = edgemetric.argmax()-1
    return ind

def mask3(img,c):
    ### Create mask
    mask = np.zeros(img.shape,np.uint8)
    cv.drawContours(mask,[c],0,1,-1)
    mask[:,:,1],mask[:,:,2] = mask[:,:,0],mask[:,:,0]
    return mask

def contoursGRAY(orig,thresh,draw=False):
    ### take channel with least saturation
    b,g,r = cv.split(orig)
    ind = np.argmin([b.sum(),g.sum(),r.sum()])
    gray = orig[:,:,ind]
    
    ### Global grayscale threshold
    gray=cv.GaussianBlur(gray, (5, 5), 0)
    ret1,th1 = cv.threshold(gray,thresh,256,cv.THRESH_BINARY)
    contours,hierarchy = cv.findContours(th1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        # find the biggest contour (c) by the area
        c = max(contours, key = cv.contourArea)
        if draw:
            cv.drawContours(orig, c, -1, (255,0,0), 1)
    else:
        return None

    ret2,th2 = cv.threshold(gray,200,256,cv.THRESH_BINARY)
    contours,hierarchy = cv.findContours(th2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        # find the biggest contour (c) by the area
        stingc = max(contours, key = cv.contourArea)
        if draw:
            cv.drawContours(orig, stingc, -1, (0,255,0), 1)
    else:
        return None

    return c,stingc

def contoursHSV(orig,flags,draw=False,plot=False,
                minHSV=(90,0,100),maxHSV=(128,255,255),
                stingMinHSV=(60,200,40),stingMaxHSV=(110,250,255),
                modelpercent=.005):

    # Load an color image in HSV, apply HSV transform again
    hsv_ = cv.cvtColor(orig, cv.COLOR_BGR2HSV)
    hsv_=cv.GaussianBlur(hsv_, (5, 5), 0)
    maxpx = hsv_[:,:,2].max()+1
    hsv_[:,:,2] = (hsv_[:,:,2]*(255/maxpx)).astype(np.uint8)
    hsv = cv.cvtColor(hsv_, cv.COLOR_RGB2HSV)
    npx = hsv[:,:,2].size

    # retrieve original hsv intensity
    hsv[:,:,2] = hsv_[:,:,2]

    ### Find model contours
    maskHSV = cv.inRange(hsv, minHSV, maxHSV)
    contours,hierarchy = cv.findContours(maskHSV, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    ### Sting adaptor contours
    stinghsv=cv.GaussianBlur(hsv, (15, 15), 0)
    stingMaskHSV = cv.inRange(stinghsv, stingMinHSV, stingMaxHSV)
    stingContours,stingHierarchy = cv.findContours(stingMaskHSV, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # find the biggest contour (c) by the area
        c = max(contours, key = cv.contourArea)

    if len(stingContours) !=0:
        stingc = max(stingContours, key = cv.contourArea)
        if cv.contourArea(stingc) < npx*modelpercent:
            return None
        if len(contours) == 0 or cv.contourArea(c) < npx*modelpercent:
            c = stingc
    else:
        return None
    if draw:
        cv.drawContours(orig, [c], -1, (255,0,0), 1)
        cv.drawContours(orig, stingc, -1, (0,255,0), 1)

    return c,stingc

def getEdgesFromContours(orig,c,stingc,draw=False,plot=False):
    ### Bounding box
    th,cx,cy,(x,y,w,h),flowRight = getOrientation(c)
    dx = int(w/4.)
    xs,ys,ws,hs = cv.boundingRect(stingc)

    x1,x2 = min(x,xs),max(x+w,xs+ws)
    y1,y2 = min(y,ys),max(y+h,ys+hs)

    ROI = (x1,y1,x2-x1,y2-y1)
    boxes = ((x,y,w,h),(xs,ys,ws,hs))
    orientation = (th,cx,cy)

    cEdge = getFrontEdgeFromContour(c,flowRight)
    stingEdge= getFrontEdgeFromContour(stingc,flowRight)

    edges = (cEdge,stingEdge)
    
    ### Draw features
    if draw:
        cv.rectangle(orig,(x,y,w,h),(0,0,255))
        cv.rectangle(orig,(xs,ys,ws,hs),(0,255,255))
        cv.rectangle(orig,ROI,(255,255,255))
        cv.circle(orig,(int(cx),int(cy)),4,(0,255,0))
        cv.drawContours(orig, cEdge, -1, (255,0,0), 1)
        cv.drawContours(orig, stingEdge, -1, (0,255,0), 1)
    
    if plot:
        # show the images
        plt.figure(0)
        rgb = orig[...,::-1].copy()
        plt.subplot(1,1,1),plt.imshow(rgb)
        plt.show()
    return edges, ROI, orientation, flowRight

def getFrontEdgeFromContour(c,flowRight):
    ### contours are oriented counterclockwise from top left
    #print('entered edge')
    if flowRight:
        ymax_ind = c[:,0,1].argmax()
        frontedge = c[:ymax_ind,:,:]
    else:
        ymin_ind = c[:,0,1].argmax()
        frontedge = c[ymin_ind:,:,:]
    #print(np.shape(frontedge))
    return frontedge

def combineEdges(c,stingc,flowRight,cutoff=20):
    ### top corner
    pc = c[cutoff,:,:]
##    print(pc,np.where(stingc[:,0,0] == pc[0,0]))
##    print(c[:20,0,0],stingc[:20,0,0])
    ind = np.where(stingc[:,0,0] == pc[0,0])[0][0]
    mt=stingc[:ind+1,0,:]
    
    #linear offset correction
    delta = stingc[ind,:,:] - pc
    s = np.linspace(0,1,ind+1)
    ds = (delta*s[:, np.newaxis]).astype(np.int32)
    mt -= ds

    ### Bottom corner
    pc = c[-(cutoff+1),:,:]
    ind = np.where(stingc[:,0,0] == pc[0,0])[0][-1]
    mb=stingc[ind:,0,:]
    
    #linear offset correction
    delta = stingc[ind,:,:] - pc
    s = np.linspace(1,0,len(mb))
    ds = (delta*s[:, np.newaxis]).astype(np.int32)
    mb -= ds

    cn = np.append(mt[:,np.newaxis,:],c[cutoff:-cutoff,:,:],axis=0)
    cn = np.append(cn,mb[:,np.newaxis,:],axis=0)
    return cn

def interpolateCorners(c,stingc,flowRight,npx=5,kind='linear'):
    ### top corner
    pts = np.append(stingc[0:npx,0,:], c[0:npx,0,:],axis=0)
    pts = pts[pts[:,0].argsort()]
    ft = interp1d(pts[:,0], pts[:,1], kind=kind)
    if flowRight:
        x = np.arange(stingc[0,0,0],c[0,0,0],-1)
    else:
        x = np.arange(stingc[0,0,0],c[0,0,0])
    y = ft(x).astype(np.int32)
    ct = np.hstack((x[:,np.newaxis], y[:,np.newaxis]))

    ### bottom corner
    pts = np.append(stingc[-npx:,0,:], c[-npx:,0,:],axis=0)
    pts = pts[pts[:,0].argsort()]
    fb = interp1d(pts[:,0], pts[:,1], kind=kind)
    if flowRight:
        x = np.arange(stingc[-1,0,0],c[-1,0,0],-1)
    else:
        x = np.arange(stingc[-1,0,0],c[-1,0,0])  
    cb = np.hstack((x[:,np.newaxis], fb(x).astype(np.int32)[:,np.newaxis]))
    
    cn = np.append(ct[:,np.newaxis,:],c,axis=0)
    cn = np.append(cn,cb[:,np.newaxis,:],axis=0)
    return cn
 
# Show keypoints
if __name__ == "__main__":
    detector = getBlobDetector()
    im_with_keypoints, isFound, centers= getBlobGrid("2x9.png",detector)
    print(isFound)
    plt.imshow(im_with_keypoints)
    plt.show()
    



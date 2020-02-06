import numpy as np
import cv2 as cv
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def interpolateContour(contour,ninterp,kind='linear'):
    """
    Interpolates given opencv contour using length parameterization
    with ninterp equally spaced points

    :param contour: opencv contour, shape(n,1,n)
    :param ninterp: integer, number of interpolation points
    :returns: output, interpolated contour positions
    """
    x,y= contour[:,0,0],contour[:,0,1]
    # identify duplicate points
    okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
    xp,yp = x[okay],y[okay]

    # find normalized length parameterization
    dl = np.sqrt(np.diff(xp)**2 + np.diff(yp)**2)
    lp = np.append(np.array([0.]),np.cumsum(dl))
    lp /= lp[-1]

    # interpolate using length
    fx = interp1d(lp, xp, kind=kind)
    fy = interp1d(lp, yp, kind=kind)
    ln = np.linspace(0,1,ninterp)
    xi,yi = fx(ln),fy(ln)

    output = np.zeros((ninterp,1,2))
    output[:,0,0],output[:,0,1] = xi,yi

    return output
    
def getConvexHull(contour,ninterp):
    """
    Returns interpolated convexHull contour

    :param contour: opencv contour, shape(n,1,n)
    :param ninterp: integer, number of interpolation points
    :returns: c, interpolated pixel positions
    """
    c= cv.convexHull(contour,clockwise=True)
    c = interpolateContour(c,ninterp,kind='linear')
    c = np.append(c[-1:,:,:],c,axis=0)
    
    return c.astype(np.int32)

def getContourAxes(contour):
    """
    Uses priciple component analysis to acquire
    axes, center, and rotation of given contour

    :param contour: opencv contour, shape(n,1,n)
    :returns: center position, eigenvectors, eigenvalues,angle 
    """
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

def getOrientation(contour):
    """
    Uses moments of inertia to acquire
    axes, center, and rotation of given contour

    :param contour: opencv contour, shape(n,1,n)
    :returns: angle,center position, bounding box,flow direction boolean
    """
    mu = cv.moments(contour)
    x,y,w,h = cv.boundingRect(contour)
    th = 0.5*np.arctan2(2*mu['mu11'],mu['mu20']-mu['mu02'])
    cx,cy = mu['m10']/mu['m00'],mu['m01']/mu['m00']
    flowRight = x+w/2. - contour[contour[:,0,1].argmin(),0,0] <0
    
    return th,cx,cy,(x,y,w,h),flowRight

def classifyImageHist(gray,verbose=False,stingpercent=.05,modelpercent=.005):
    """
    Uses histogram of 8bit grayscale image (0,255) to classify image type

    :param gray: opencv image
    :param verbose: boolean
    :param stingpercent: minimum percent area of sting arm
    :param modelpercent: minimum percent area of model
    :returns: dictionary of flags
    """
    ### grayscale histogram
    histr = cv.calcHist( [gray], None, None, [256], (0, 256));
    imgsize = gray.size

    ### classification criteria
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

def contoursGRAY(orig,thresh,log=None,draw=False,plot=False):
    """
    Find contours for overexposed images

    :param orig: opencv 8bit BGR image
    :param thresh: integer threshold
    :param draw: boolean, if True draws on orig image
    :param plot: boolean, if True plots orig image
    :returns: success boolean, model contour, sting contour
    """
    
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
        if log is not None:
            log.write('no GRAY model contours found at thresh==%i'%thresh)
        return None

    ret2,th2 = cv.threshold(gray,200,256,cv.THRESH_BINARY)
    contours,hierarchy = cv.findContours(th2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        # find the biggest contour (c) by the area
        stingc = max(contours, key = cv.contourArea)
        if draw:
            cv.drawContours(orig, stingc, -1, (0,255,0), 1)
    else:
        if log is not None:
            log.write('no GRAY sting contours found at thresh==200')
        return None

    return c,stingc

def contoursHSV(orig,draw=False,plot=False,log=None,
                minHSV=(90,0,100),maxHSV=(128,255,255),
                stingMinHSV=(60,200,40),stingMaxHSV=(110,250,255),
                modelpercent=.005):
    """
    Find contours for good images and underexposed images.
    Uses the BGR-HSV transformation twice to increase edge contrast.
    Value channel of first HSV transform is retained as well.

    :param orig: opencv 8bit BGR image
    :param minHSV: minimum tuple for HSV-sq transform
    :param maxHSV: maximum tuple for HSV-sq transform
    :param stingMinHSV: minimum tuple for HSV-sq transform
    :param stingMaxHSV: maximum tuple for HSV-sq transform
    :param modelpercent: minimum percent area for model id
    :param draw: boolean, if True draws on orig image
    :param plot: boolean, if True plots orig image & HSV-sq image
    :returns: success boolean, model contour, sting contour
    """
    # Load an color image in HSV, apply HSV transform again
    hsv_ = cv.cvtColor(orig, cv.COLOR_BGR2HSV)
    hsv_=cv.GaussianBlur(hsv_, (5, 5), 0)
    maxpx = hsv_[:,:,2].max()+1
    hsv_[:,:,2] = (hsv_[:,:,2]*(255/maxpx)).astype(np.uint8)
    hsv = cv.cvtColor(hsv_, cv.COLOR_RGB2HSV)
    npx = hsv[:,:,2].size

    # retrieve original hsv intensity
    hsv[:,:,2] = hsv_[:,:,2]

    # Plot colorspaces
    if plot:
        plt.figure(figsize=(8, 16))
        rgb = orig[...,::-1].copy()
        plt.subplot(2,1,1),plt.imshow(rgb)
        plt.title('RGB colorspace')
        plt.subplot(2,1,2),plt.imshow(hsv)
        plt.title('HSV-sq colorspace')
        plt.tight_layout()
        plt.show()

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
        # find the biggest contour (c) by the area
        stingc = max(stingContours, key = cv.contourArea)
        if cv.contourArea(stingc) < npx*modelpercent:
            if log is not None:
                log.write('no model, HSV-sq contours area < %f'%modelpercent)
            return None
        if len(contours) == 0 or cv.contourArea(c) < npx*modelpercent:
            c = stingc
    else:
        if log is not None:
            log.write('no HSV-sq contours found')
        return None
    
    if draw:
        cv.drawContours(orig, [c], -1, (255,0,0), 1)
        cv.drawContours(orig, stingc, -1, (0,255,0), 1)
    
    return c,stingc

def getEdgesFromContours(orig,c,stingc,flags,
                         draw=False,plot=False):
    """
    Finds front edge for two contours and flag dictionary.

    :param orig: opencv 8bit BGR image
    :param c: opencv contour for model, shape(n,1,n)
    :param stingc: opencv contour for sting, shape(n,1,n)
    :param flags: dictionary of flags classifying image
    :param draw: boolean, if True draws on orig image
    :param plot: boolean, if True plots orig image & HSV-sq image
    :returns: edges, ROI, orientation, boolean flowRight
    """
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
##        cv.rectangle(orig,(x,y,w,h),(0,0,255))
##        cv.rectangle(orig,(xs,ys,ws,hs),(0,255,255))
        cv.rectangle(orig,ROI,(255,255,255))
        cv.circle(orig,(int(cx),int(cy)),4,(0,255,0))
        cv.drawContours(orig, cEdge, -1, (255,0,0), 1)
        cv.drawContours(orig, stingEdge, -1, (0,255,0), 1)
    
    if plot:
        # show the images
        plt.figure()
        plt.title("Plotting getEdgesFromContours")
        rgb = orig[...,::-1].copy()
        plt.subplot(1,1,1)
        plt.imshow(rgb)
        plt.show()
    return edges, ROI, orientation, flowRight

def getFrontEdgeFromContour(c,flowRight):
    """
    Find front edge of contour given flow direction

    :param c: opencv contour, shape(n,1,n)
    :param flowRight: boolean for flow direction
    :returns: frontedge, contour of front edge
    """
    ### contours are oriented counterclockwise from top left
    if flowRight:
        ymax_ind = c[:,0,1].argmax()
        frontedge = c[:ymax_ind,:,:]
    else:
        ymin_ind = c[:,0,1].argmax()
        frontedge = c[ymin_ind:,:,:]
    return frontedge

def combineEdges(c,stingc,flowRight,cutoff=50):
    """
    Combine model and sting contours to capture
    front corners of models
    
    :param c: opencv contour for model, shape(n,1,n)
    :param stingc: opencv contour for sting, shape(n,1,n)
    :param flowRight: boolean for flow direction
    :param cutoff: integer, # of elements at edge of model contour to clip
    :returns: merged contour
    """
    ### top corner
    pc = c[cutoff,:,:]
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

    # merge edge corners with center edge
    cn = np.append(mt[:,np.newaxis,:],c[cutoff:-cutoff,:,:],axis=0)
    cn = np.append(cn,mb[:,np.newaxis,:],axis=0)

    return cn

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)] 
 

    

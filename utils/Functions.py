import numpy as np
import cv2 as cv
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from cnn import cnn_apply

def splitfn(fn):
    fn = os.path.abspath(fn)
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def filter_hsv_ranges(hsv,ranges):
    ''' Get union of masks for multiple hsv ranges '''
    maskHSV = np.zeros((hsv.shape[0],hsv.shape[1]),dtype=np.uint8)
    for i in range(0,len(ranges[0])):
        mask = cv.inRange(hsv, ranges[0][i],ranges[1][i])
        maskHSV = cv.bitwise_or(mask,maskHSV)
        
    return maskHSV

def interpolateContour(contour, ninterp, kind='linear'):
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
    xi, yi = fx(ln), fy(ln)

    output = np.zeros((ninterp, 1, 2))
    output[:,0,0],output[:,0,1] = xi,yi

    return output

def getConvexHull(contour, ninterp, plot=False):
    """
    Returns interpolated convexHull contour

    :param contour: opencv contour, shape(n,1,n)
    :param ninterp: integer, number of interpolation points
    :returns: c, interpolated pixel positions
    """
    c= cv.convexHull(contour, clockwise=True)
    c = np.append(c[-1:, :, :], c,axis=0)

    ### cycle contour indices such that
    ### index==0 is at top left (min row)
    ind = c[:,0,1].argmin(); c= np.roll(c,-ind+1,axis=0)

    # Interpolate positions
    c = interpolateContour(c,ninterp)

    if plot:
        plt.plot(c[:,:,0],c[:,:,1],'kx')
        plt.plot(c[0,:,0],c[0,:,1],'ro')
        plt.plot(c[5,:,0],c[5,:,1],'bo')
        plt.show()

    return c.astype(np.int32)

def getPoints(c, flow_direction="right", r=[-.75,-.25,0,.25,.75],prefix="MODEL"):
    """
    Returns interpolated points at relative vertical positions to center
       -assumes only front edge is passed into function (not a closed contour)
       -assumes dense contour, every vertical pixel is occupied 

    :param c: opencv contour, shape(n,1,n)
    :param r: list, interpolation points relative to radius
    :returns: interpolated pixel positions
    """
    ### Extract min/max ypos
    low_ind = c[:,0,1].argmin(); ymin = c[low_ind,0,1]
    high_ind= c[:,0,1].argmax(); ymax = c[high_ind,0,1]
    center = int((ymin+ymax)/2)
    radius = int((ymax-ymin)/2)

    ### Setup output dictionary
    output = {prefix+'_R':r,prefix+'_YCENTER':center,prefix+'_YLOW':ymin,prefix+'_YMAX':ymax,prefix+'_RADIUS':radius}

    # Interpolate corresponding horizontal positions
    xpos = np.zeros(len(r))
    ypos = [int(y*radius+center) for y in r]

    li = min(low_ind,high_ind)
    hi = max(low_ind,high_ind)
    inds = np.arange(li,hi)
    for ind in inds:
        for j in range(0,len(ypos)):
            if abs(c[ind,0,1] - ypos[j]) < 3:
                xpos[j] = c[ind,0,0]
    output[prefix+'_INTERP_XPOS'] = xpos

    return output.copy()

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

def classifyImageHist(img,verbose=False,stingpercent=.05,modelpercent=.005):
    """
    Uses histogram of 8bit grayscale image (0,255) to classify image type

    :param img: opencv image
    :param verbose: boolean
    :param stingpercent: minimum percent area of sting arm
    :param modelpercent: minimum percent area of model
    :returns: dictionary of flags
    """
    ### HSV brightness value histogram
    hsv_ = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv_=cv.GaussianBlur(hsv_, (5, 5), 0)
    gray = hsv_[:,:,2]

    ### grayscale histogram
    histr = cv.calcHist( [gray], None, None, [256], (0, 256))
    imgsize = gray.size

    ### classification criteria
    modelvis = (histr[12:250]/imgsize > 0.00).sum() != 1
    modelvis *= histr[50:250].sum()/imgsize > modelpercent
    stingvis= histr[50:100].sum()/imgsize > stingpercent
    overexp = histr[243:].sum()/imgsize > modelpercent
    underexp= histr[150:].sum()/imgsize < modelpercent

    ### Determine saturation limit
    slimit,peaki=253,240
    if overexp:
        peaki = histr[240:].argmax() +240
        if histr[peaki:].sum()/imgsize > modelpercent:
            slimit = peaki-1
    saturated = histr[slimit:].sum()/imgsize > modelpercent

    ### Extract intensity threshold
    try:
        total = histr[30:].sum()+1
        exp_val = (histr[30:].ravel()*np.arange(30,256)).sum()/total
        avg = int(max(exp_val,55))
        peaki = histr[avg:].argmax() +avg
        if abs(peaki-avg) < 10:
            thresh=peaki-15
        else:
            thresh = max(histr[avg:peaki].argmin() +avg, peaki-15)
    except:
        thresh = 150
        
    if verbose:
        print('peaki, slimit, thresh', peaki,slimit,thresh)
        print("Model visible",modelvis)
        print("Sting visible",stingvis)
        print("overexposed", overexp)
        print("underexposed", underexp)
        print("saturated",saturated)
        plt.title("Grayscale Histogram")
        plt.plot(histr/imgsize,'-')
        plt.ylim([0,histr[12:].max()/imgsize])
        plt.show()
    return {'underexp':underexp,
            'overexp':overexp,
            'saturated':saturated,
            'stingvis':stingvis,
            'modelvis':modelvis,
            }, thresh

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

def contoursGRAY(orig,thresh=150,log=None):
    """
    Find contours for overexposed images

    :param orig: opencv 8bit BGR image
    :param thresh: integer threshold
    :returns: model contour
    """
    flags={'SHOCK_CONTOUR_FAILED':True}
    ### take channel with least saturation
    gray_ = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)

    ### Global grayscale threshold
    gray=cv.GaussianBlur(gray_, (5, 5), 0)
    ret1,th1 = cv.threshold(gray,thresh,255,cv.THRESH_BINARY)
    contours,hierarchy = cv.findContours(th1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        # find the biggest contour (c) by the area
        modelC = max(contours, key = cv.contourArea)
    else:
        if log is not None:
            log.write('no GRAY model contours found at thresh==%i'%thresh)
        modelC = None
        flags['MODEL_CONTOUR_FAILED']=True

    contour_dict ={'MODEL':modelC,'SHOCK':None}
    
    return contour_dict,flags

def contoursAutoHSV(orig,log=None,flags={'UNDEREXPOSED':False}):
    """
    Find contours using default union of multiple HSV ranges.
    Uses the BGR-HSV transformation to increase contrast.

    :param orig: opencv 8bit BGR image
    :param flags: dictionary with flags
    :param log: log object
    :returns: model contour, shock contour, flags
    """

    img = cv.cvtColor(orig, cv.COLOR_BGR2HSV)

    ### HSV pixel ranges for models taken from sample frames
    model_ranges  = np.array([[(0,0,208),   (155,0,155),  (13,20,101), (0,190,100),  (12,150,130)], 
                              [(180,70,255),(165,125,255),(33,165,255),(13,245,160),(25,200,250)]])
    dim_model =np.array([[(7,0,8)],[(20,185,101)]])

    ### HSV pixel ranges for shocks taken from sample frames
    shock_ranges = np.array([[(125,78,115)], 
                            [(145,190,230)]])
    dim_shocks = np.array([[(125,100,35), (140,30,20), (118,135,30)], 
                           [(165,165,150),(156,90,220),(128,194,125)]])
    
    # Append additional ranges for underexposed images
    if flags['UNDEREXPOSED']:
        model_ranges = np.hstack((model_ranges,dim_model))

    # Apply shock filter and extract shock contour
    shockfilter = filter_hsv_ranges(img,shock_ranges)
    if shockfilter.sum() < 500:
        flags['DIM_SHOCK'] = True
        shock_ranges = np.hstack((shock_ranges,dim_shocks))
        shockfilter = filter_hsv_ranges(img,shock_ranges)
    shockcontours,hierarchy = cv.findContours(shockfilter, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # plt.imshow(shockfilter)
    # plt.show()

    # find the biggest shock contour (shockC) by area
    if len(shockcontours) == 0:
        shockC = None
        flags['SHOCK_CONTOUR_FAILED']=True
        if log is not None:
            log.write('no shock contours found')
    else:
        shockC = max(shockcontours, key = cv.contourArea)

    # Apply model filter and extract model contour
    modelfilter = filter_hsv_ranges(img,model_ranges)
    if flags['UNDEREXPOSED']:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
        modelfilter = cv.morphologyEx(modelfilter, cv.MORPH_OPEN, kernel)
    modelcontours,hierarchy = cv.findContours(modelfilter, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # plt.imshow(modelfilter)
    # plt.show()
    # find the biggest model contour (modelC) by area
    if len(modelcontours) == 0:
        modelC = None
        flags['MODEL_CONTOUR_FAILED'] = True
        if log is not None:
            log.write('no model contours found')
    else:
        modelC = max(modelcontours, key = cv.contourArea)
    
    contour_dict ={'MODEL':modelC,'SHOCK':shockC}

    return contour_dict,flags

def contoursHSV(orig,log=None,
                minHSVModel=(0,0,150),maxHSVModel=(181,125,256),
                minHSVShock=(125,78,115),maxHSVShock=(145,190,230) ):
    """
    Find contours using HSV ranges image.
    Uses the BGR-HSV transformation to increase contrast.

    :param orig: opencv 8bit BGR image
    :param minHSVModel: minimum tuple for HSV range
    :param maxHSVModel: maximum tuple for HSV range
    :param minHSVShock: minimum tuple for HSV range
    :param maxHSVShock: maximum tuple for HSV range
    :returns: model contour, shock contour
    """

    flags={'MODEL_CONTOUR_FAILED':False,'SHOCK_CONTOUR_FAILED':False}
    # Load an color image in HSV, apply HSV transform again
    hsv_ = cv.cvtColor(orig, cv.COLOR_BGR2HSV)
    hsv  = cv.GaussianBlur(hsv_, (5, 5), 0)
    
    ### Model contours
    modelmask = cv.inRange(hsv, minHSVModel,maxHSVModel)
    modelcontours,hierarchy = cv.findContours(modelmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    # find the biggest model contour (modelC) by area
    if len(modelcontours) == 0:
        modelC = None
        flags['MODEL_CONTOUR_FAILED'] = True
        if log is not None:
            log.write('no shock contours found')
    else:
        modelC = max(modelcontours, key = cv.contourArea)

    ### Shock contours
    shockmask = cv.inRange(hsv, minHSVShock, maxHSVShock)
    shockcontours,hierarchy = cv.findContours(shockmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # plt.imshow(hsv)
    # plt.show()
    # plt.imshow(shockmask)
    # plt.show()

    # find the biggest shock contour (shockC) by area
    if len(shockcontours) == 0:
        shockC = None
        flags['SHOCK_CONTOUR_FAILED'] = True
        if log is not None:
            log.write('no shock contours found')
    else:
        shockC = max(shockcontours, key = cv.contourArea)
    
    contour_dict ={'MODEL':modelC,'SHOCK':shockC}

    return contour_dict,flags

def contoursCNN(orig,model, log=None):
    """
    Find contours using HSV ranges image.
    Uses the BGR-HSV transformation to increase contrast.

    :param orig: opencv 8bit BGR image
    :param model: compiled CNN model
    :returns: model contour, shock contour
    """

    flags={'MODEL_CONTOUR_FAILED':False,'SHOCK_CONTOUR_FAILED':False}
    
    ### Apply CNN
    cnnmask = cnn_apply(orig,model)

    ### Model contours
    modelmask = ((cnnmask==1)*255).astype(np.uint8)
    modelcontours,hierarchy = cv.findContours(modelmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    # find the biggest model contour (modelC) by area
    if len(modelcontours) == 0:
        modelC = None
        flags['MODEL_CONTOUR_FAILED'] = True
        if log is not None:
            log.write('no shock contours found')
    else:
        modelC = max(modelcontours, key = cv.contourArea)

    ### Shock contours
    shockmask = ((cnnmask==2)*255).astype(np.uint8)
    shockcontours,hierarchy = cv.findContours(shockmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # plt.imshow(hsv)
    # plt.show()
    # plt.imshow(shockmask)
    # plt.show()

    # find the biggest shock contour (shockC) by area
    if len(shockcontours) == 0:
        shockC = None
        flags['SHOCK_CONTOUR_FAILED'] = True
        if log is not None:
            log.write('no shock contours found')
    else:
        shockC = max(shockcontours, key = cv.contourArea)
    
    contour_dict ={'MODEL':modelC,'SHOCK':shockC}

    return contour_dict,flags

def getROI(c1,c2):
    """
    Finds ROI for two contours

    :param c1: opencv contour for model, shape(n,1,n)
    :param c2: opencv contour for sting, shape(n,1,n)
    :returns: ROI
    """
    ### Bounding box
    x,y,w,h = cv.boundingRect(c1)
    xs,ys,ws,hs = cv.boundingRect(c2)

    x1,x2 = min(x,xs),max(x+w,xs+ws)
    y1,y2 = min(y,ys),max(y+h,ys+hs)

    ROI = (x1,y1,x2-x1,y2-y1)
    
    return ROI

def getEdgeFromContour(c,flow_direction, offset = None):
    """
    Find front edge of contour given flow direction

    :param c: opencv contour, shape(n,1,n)
    :param flow_direction: 'left' or 'right'
    :returns: frontedge, contour of front edge
    """
    ### contours are oriented counterclockwise from top left
    ymin_ind = c[:,0,1].argmin()
    ymax_ind = c[:,0,1].argmax()
    if flow_direction=='right':
        frontedge = c[ymin_ind:ymax_ind,:,:]
    else:
        frontedge = c[ymax_ind:,:,:]
    frontedge[:,0,1] += offset[0]
    frontedge[:,0,0] += offset[1]
    return frontedge

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
    return y[int(window_len/2)-1:-int(window_len/2)-1] 

def convert_mask_BGR_to_gray(img):
    mask = np.zeros(img.shape[0:2],np.uint8)
    B,G,R = img[:,:,0], img[:,:,1], img[:,:,2]
    
    # Red -> 0, Green->1, Blue -> 2
    mask[R>2] = 0
    mask[G>2] = 1
    mask[B>2] = 2

    return mask

def convert_mask_gray_to_BGR(img):
    mask = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # Red -> 0, Green->1, Blue -> 2
    mask[:,:,0]= (img==2)*255
    mask[:,:,1]= (img==1)*255
    mask[:,:,2]= (img==0)*255

    return mask

def annotateImage(orig,flags,top=True,left=True):
    y,x,c = np.shape(orig)

    if top:
        yp = int(y*.025)
    else:
        yp = int(y*.85)
    if left:
        xp = int(x*.025)
    else:
        xp = int(y*.85)

    offset=0
    for key in flags.keys():
        
        if flags[key] and key != 'modelvis':
            cv.putText(
             orig, #numpy array on which text is written
             "{0}: {1}".format(key,True), #text
             (xp,yp+offset), #position at which writing has to start
             cv.FONT_HERSHEY_SIMPLEX, #font family
             1, #font size
             (0, 0, 255, 255), #font color
             3) #font stroke
        offset += int(y*.035)

def cropBGR(img, CROP):
    return img[CROP[0][0]:CROP[0][1], CROP[1][0]:CROP[1][1], :]

def cropGRAY(img, CROP):
    return img[CROP[0][0]:CROP[0][1], CROP[1][0]:CROP[1][1]]

def getOutlierMask(metric,threshold=2,method="stdev"):
    """Returns mask for outliers in time series

    Args:
        metric (array): 1D numpy array, presumably time series
        threshold (int, optional): Threshold value. Default is 2 for stdev.
        method (str, optional): "stdev" or "percent". Defaults to "stdev".

    Returns:
        mask: numpy masked array
    """
    mask = np.zeros_like(metric)
    if len(metric) > 25:
        smetric = smooth(metric)
        diff = abs(smetric - metric)

        if method=="percent":
            rdiff = abs(diff/(metric + 1e-3))
            mask += rdiff > threshold

        if method=="stdev":
            dmean = np.mean(diff)
            dstd = np.std(diff)
            mask += diff > dmean + threshold*dstd

    return mask

def cleanEdge(edge, ind=1, ninterp=None):
    """Return interpolation of edge with non-function points removed
        Only designed for eliminating adjacent non-function points,
        ***will not work on contours like closed circles

    Args:
        edge (array): non-closed opencv contour, numpy ndarray shape->(n,1,n)
        ninterp (int): number of interpolation points, Defaults to n/10
        ind (int): 0 or 1 for function of X or function of Y

    Returns:
        cf: opencv contour array (ni,1,ni)
    """
    if ninterp is None:
        ninterp = int(edge.shape[0]*2)
    c1 = interpolateContour(edge,ninterp)
    
    nx,n,ny = c1.shape
    x0, y0 = c1[0,0,0], c1[0,0,1]
    if ind == 1: # Y(X)
        for i in range(1,nx-1):
            if c1[i,0,0] == x0:
                c1[i,0,0] = np.nan
                c1[i,0,1] = np.nan
            else:
                x0 = c1[i,0,0]
    else: # X(Y)
        for i in range(1,ny-1):
            if c1[i,0,1] == y0:
                c1[i,0,0] = np.nan
                c1[i,0,1] = np.nan
            else:
                y0 = c1[i,0,1]

    x, y = c1[:,0,0], c1[:,0,1]
    xo, yo = x[~(np.isnan(x))], y[~(np.isnan(x))]

    retc = np.zeros((len(xo),1,2))
    retc[:,0,0] = xo
    retc[:,0,1] = yo

    return retc

def getEdgeDifference(edge1, edge2, dim_ind=1,ninterp=100):
    """ Function to subtract two edges via interpolation to measure shape change

    Args:
        edge1 (array): opencv contour format, numpy ndarray (n,1,n)
        edge2 (array): opencv contour format, numpy ndarray (n,1,n)
        ninterp (int, optional): number of interpolation points.
    
    Returns:
        snew: ndarray of interpolated indep variable
        vdiff: interpolated differences of dependent variable
        vi1: interpolated dependent variable of edge1
        vi2: interpolated dependent variable of edge2
    """
    # get interpolatable set of edge points (e.g. single valued function)
    e1 = cleanEdge(edge1,dim_ind)
    e2 = cleanEdge(edge2,dim_ind)

    # extract independent variable
    s1, s2 = e1[:,0,dim_ind], e2[:,0,dim_ind]
    s1 -= (s1.max()+s1.min())/2
    s2 -= (s2.max()+s2.min())/2
    smin = min(s1.min(), s2.min())
    smax = max(s1.max(), s2.max())

    # extract dependent variable
    vdim_ind = (dim_ind + 1)%2
    v1, v2 = e1[:,0,vdim_ind], e2[:,0,vdim_ind]

    # interpolate edges as function of independent variable
    f1, f2 = interp1d(s1, v1, bounds_error=False), interp1d(s2, v2, bounds_error=False)
    snew = np.linspace(smin,smax,ninterp)
    vi1,vi2 = f1(snew), f2(snew)
    vdiff = vi2-vi1

    return snew, vdiff, vi1,vi2
 

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from Functions import getOrientation,classifyImageHist,getEdgesFromContours
from Functions import contoursGRAY,contoursHSV,combineEdges
from Functions import getConvexHull

def normImage():
    return

def getCameraCalib(img_mask,pattern_shape,square_size=1.0,nthreads=4,folder='./calib/'):
    ### Retrieve filenames from img_mask (i.e. "./camera/frame??.jpg")
    from glob import glob
    from Functions import findChessCorners,splitfn

    img_names = glob(img_mask)
    
    ### Generate 3D (x,y,z) obj points associated with pattern, assuming z=0
    pattern_points = np.zeros((np.prod(pattern_shape), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_shape).T.reshape(-1, 2)
    pattern_points *= square_size

    ### Create arrays to store img and obj points, check image size
    obj_points = []
    img_points = []
    h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]

    ### wrapper with single arg- needed for multiproc
    def g(fn): 
        img = cv.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            return None
        hi, wi = img.shape[:2]
        assert hi==h and wi==w,("%s has different shape: %d x %d ... " % (fn,hi,wi))
        corners= findChessCorners(img, pattern_shape)
        
        ### Output imgs with annotated corners
        if folder and not os.path.isdir(folder):
            os.mkdir(folder)
        vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(vis, pattern_shape, corners, True)
        _path, name, _ext = splitfn(fn)
        outfile = os.path.join(folder, name + '_chess.png')
        cv.imwrite(outfile, vis)
        
        return corners

    ### Multiprocessing for speed
    if nthreads <= 1:
        chessboards = [g(fn) for fn in img_names]
    else:
        print("Run with %d threads..." % nthreads)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(nthreads)
        chessboards = pool.map(g, img_names)

    ### Only keep frames with recognized pattern
    chessboards = [x for x in chessboards if x is not None]
    for corners in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())
    return camera_matrix, dist_coefs

def getModelProps(orig,plot=False,draw=False,verbose=False,):
    ### Classify image
    gray = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    flags = classifyImageHist(gray)
    if verbose:
        print(flags)

    if flags['modelvis'] == False:
        return None
    
    if flags['stingvis']:
        if flags['saturated']:
            thresh = 252
        else:
            thresh = 230
        try:
            c,stingc = contoursGRAY(orig,thresh)
            edges, ROI, orientation, flowRight = getEdgesFromContours(orig,c,stingc,flags,draw=draw,plot=plot)
        except TypeError:
            return None
    else:
        
        ### HSV contours
        try:
            c,stingc = contoursHSV(orig,flags,plot=plot,draw=True)
            if flags['underexp']:
                print('hull executed')
                c = getConvexHull(c,1000)
            stingc = getConvexHull(stingc,10000)
            edges, ROI, orientation, flowRight = getEdgesFromContours(orig,c,stingc,flags,draw=draw,plot=plot)

            diff_top = edges[1][0,0,:]- edges[0][0,0,:]
            diff_bottom = edges[1][-1,0,:]- edges[0][-1,0,:]
            if len(c) < 50 or (len(c)==len(stingc) and (c==stingc).all()):
                print(len(c),'using stingc only')
                
            elif flags['saturated']:
                pass
            else:
                try:
                    cn = combineEdges(edges[0],edges[1],flowRight)
                    edges = (cn,stingc)
                except:
                    print('Corner correction failed')
        except TypeError:
            return None

    return edges, ROI, orientation, flowRight,flags

def getPose():
    return

def getModelEdge():
    return



import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

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



def getModelROI(orig,low=140,high=255,plot=False):
    # Load an color image in grayscale
    img = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    hist_full = cv.calcHist([img],[0],None,[256],[0,256])
##    plt.plot(hist_full)
##    plt.show()
    
    px,py = img.shape
    total = np.sum(img)
    avg = total/(px*py)
    lowbar = max(avg*10,low)
    lowbar = min(lowbar,252)
    print(lowbar)
    ret,th1 = cv.threshold(img,lowbar,high,cv.THRESH_BINARY)
    contours,hierarchy = cv.findContours(th1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # find the biggest contour (c) by the area
        c = max(contours, key = cv.contourArea)
        cv.drawContours(orig, [c], -1, 255, 1)

        ### Bounding box
        x,y,w,h = cv.boundingRect(c)
        x1,y1 = (x-int(w/4),y-int(h/4))
        x2,y2 = (x+int(5*w/4),y+int(5*h/4))
        cv.rectangle(orig,(x1,y1),(x2,y2),(255,255,255),1)

        ### Centerline
        centerline = orig[y+int(h/2),x1:x2,:]
        
        if plot:
            # show the images
            plt.figure(0)
            plt.subplot(1,2,1),plt.imshow(orig)
            plt.subplot(1,2,2),plt.plot(centerline)
            plt.show()
        return x,y,w,h, centerline
    else:
        return None
    


def getPose():
    return

def getModelEdge():
    return



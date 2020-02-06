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
    """
    Finds specified chessboard grid pattern in image.
    Used for camera calibration

    :param img: opencv image, 2D numpy ndarray
    :param pattern_shape: tuple; shape of the grid (i.e. 9,6)
    :returns: boolean, corners- array of refined corner pixel positions
    """
    ### Search for corners
    found, corners = cv.findChessboardCorners(img, pattern_shape)
    if found:
        ### Refine corner positions for subpixel accuracy
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        return corners.reshape(-1, 2)
    else:
        return False,None

def getBlobDetector():
    """
    Initialize blob detector

    :returns: SimpleBlobDetector object
    """
    
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

def getBlobGrid(frame,detector,pattern_shape):
    """
    Find grid of blobs similar to chessboard
    Used for camera calibration

    :param frame: opencv 8bit grayscale image
    :param detector: SimpleBlobDetector object
    :param pattern_shape: tuple; shape of the grid (i.e. 9,6)
    :returns: annotated image, success boolean, blob coordinates 
    """
    
    # Denoise image
    im=cv.GaussianBlur(frame, (3, 3), 0)
     
    # Detect blobs.
    keypoints = detector.detect(im)
     
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    [isFound, centers] = cv.findCirclesGrid(im, pattern_shape, blobDetector=detector,
                                            flags = cv.CALIB_CB_ASYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING)
    cv.drawChessboardCorners(im_with_keypoints, pattern_shape, centers, isFound)

    return im_with_keypoints, isFound, centers

def getCameraCalib(img_mask,pattern_shape,square_size=1.0,nthreads=4,folder='./calib/'):
    ### Retrieve filenames from img_mask (i.e. "./camera/frame??.jpg")

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

# Show keypoints
if __name__ == "__main__":
    detector = getBlobDetector()
    frame = cv.imread("2x9.png",1)
    im_with_keypoints, isFound, centers= getBlobGrid(frame,detector,(2,9))
    print(isFound)
    plt.imshow(im_with_keypoints)
    plt.show()

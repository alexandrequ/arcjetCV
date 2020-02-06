import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Functions import getOrientation,classifyImageHist,getEdgesFromContours
from Functions import contoursGRAY,contoursHSV,combineEdges
from Functions import getConvexHull

class Logger(object):
    def __init__(self,filename,prefix=''):
        self.filename=filename
        self.prefix = prefix
        self.print = True
        self.fileio = False
        
    def write(self,line):
        if self.print:
            print(self.prefix+line.__str__())
        if self.fileio:
            fh = open(self.filename,'a')
            fh.write(self.prefix+line.__str__()+'\n')
            fh.close()

def getModelProps(orig,frameID,log=None,plot=False,draw=True,verbose=False,annotate=False):
    ### Initialize logfile
    if log is None:
        log = Logger('getModelProps.log')
    log.prefix = "%s: "%str(frameID)

    ### Classify image
    gray = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    flags = classifyImageHist(gray)
    if verbose:
        log.write(flags)

    if flags['modelvis'] == False:
        return None
    
    if flags['stingvis']:
        log.write('sting visible')
        if flags['saturated']:
            log.write('saturated')
            thresh = 252
        else:
            thresh = 230
        try:
            c,stingc = contoursGRAY(orig,thresh,log=log)
            ret = getEdgesFromContours(orig,c,stingc,flags,
                                       draw=draw,plot=False)
            edges, ROI, orientation, flowRight = ret
        except:
            log.write('failed GRAY edge detection')
            return None
    else:
        ### HSV contours
        try:
            c,stingc = contoursHSV(orig,plot=False,draw=draw,log=log)
            if flags['underexp']:
                log.write('underexposed, imposed convex hull')
                c = getConvexHull(c,1000)
            stingc = getConvexHull(stingc,10000)
            edges, ROI, orientation, flowRight = getEdgesFromContours(orig,c,stingc,flags,
                                                                      draw=draw,plot=False)
            ### determine if edges need to be merged
            if len(c) < 50 or (len(c)==len(stingc) and (c==stingc).all()):
                log.write(len(c),'no model contour, using stingc only')
            elif flags['saturated']:
                log.write('saturated frame, no model contour')
                pass
            else:
                try:
                    cn = combineEdges(edges[0],edges[1],flowRight)
                    edges = (cn,stingc)
                    if draw:
                        cv.drawContours(orig, cn, -1, (0,0,255), 1)
                except:
                    log.write('corner correction failed')
        except TypeError: # catch when return type is None
            log.write('failed HSV edge detection')
            return None

    if annotate:
        annotateImage(orig,flags)
    if plot:
        plt.figure()
        plt.title("Plotting getModelProps")
        rgb = orig[...,::-1].copy()
        plt.imshow(rgb)
        plt.show()

    return edges, ROI, orientation, flowRight,flags

def annotateImage(orig,flags,top=True,left=True):
    y,x,c = np.shape(orig)

    if top:
        yp = -10
    else:
        yp = y-100
    if left:
        xp = 10
    else:
        xp = x-100

    offset=0
    for key in flags.keys():
        offset += 35
        if flags[key] and key != 'modelvis':
            cv.putText(
             orig, #numpy array on which text is written
             "{0}: {1}".format(key,True), #text
             (xp,yp+offset), #position at which writing has to start
             cv.FONT_HERSHEY_SIMPLEX, #font family
             1, #font size
             (0, 0, 255, 255), #font color
             3) #font stroke
    


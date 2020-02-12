import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Functions import getOrientation,classifyImageHist,getROI
from Functions import contoursGRAY,contoursHSV,combineEdges
from Functions import getConvexHull,getEdgeFromContour

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

def getModelProps(orig,frameID,log=None,plot=False,draw=True,
                  verbose=False,annotate=True,modelpercent=.005):
    ### Initialize logfile
    if log is None:
        log = Logger('getModelProps.log')
    log.prefix = "%s: "%str(frameID)

    ### Classify image
    gray = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    flags,slimit = classifyImageHist(gray,verbose=verbose,modelpercent=modelpercent)
    if verbose:
        log.write(flags)

    ### Exit if no model visible
    if flags['modelvis'] == False:
        return None

    ### Sting is visible, image is bright, use grayscale countours

    if flags['saturated'] or flags['stingvis']:
        log.write('overexposed')
        if flags['saturated']:
            log.write('saturated')
            thresh = slimit-5
        else:
            thresh = 230
        try:
            ### Extract grayscale contours
            c,stingc = contoursGRAY(orig,thresh,log=log,plot=plot)

            ### Estimate orientation, center of mass
            th,cx,cy,(x,y,w,h),flowRight = getOrientation(c)
            if draw:
                cv.circle(orig,(int(cx),int(cy)),4,(0,255,0))
            
            ### get leading edges
            cEdge = getEdgeFromContour(c,flowRight)
            stingEdge= getEdgeFromContour(stingc,flowRight)
            edges = (cEdge,stingEdge)

            ### get ROI 
            ROI = getROI(orig,cEdge,stingEdge,draw=draw,plot=plot)

            ### try to correct corners
            try:
                cutoff = 15

                if flowRight:
                    if edges[1][-1,0,0]>edges[0][-1,0,0]:
                        cn = combineEdges(edges[0],edges[1],cutoff=cutoff)
                        edges = (cn,stingc)
                else:
                    if edges[1][-1,0,0]<edges[0][-1,0,0]:
                        cn = combineEdges(edges[0],edges[1],cutoff=cutoff)
                        edges = (cn,stingc)
                if draw:
                    cv.drawContours(orig, cn, -1, (0,0,255), 1)
                flags["cornerFailed"] = False
            except:
                flags["cornerFailed"] = True
                log.write('corner correction failed')
        except:
            log.write('failed GRAY edge detection')
            raise
            return None
    else:
        ### If sting not visible, model is isolated, use HSV
        try:
            ### Extract HSV contours
            if flags['overexp']:
                minHue=95;maxHue=140
            elif flags['underexp']:
                minHue=75;maxHue=170
            else:
                minHue=110;maxHue=140
            
            c,stingc = contoursHSV(orig,plot=plot,draw=True,log=log,
                                   minHue=minHue,maxHue=maxHue,
                                   modelpercent=modelpercent,flags=flags)

            ### Estimate orientation, center of mass
            th,cx,cy,(x,y,w,h),flowRight = getOrientation(stingc)
            if draw:
                cv.circle(orig,(int(cx),int(cy)),4,(0,255,0))

            ### Apply convex hull if underexposed
            if not flags['overexp']:
                #log.write('underexposed, imposed convex hull')
                c = getConvexHull(c,min(len(c),1000),flowRight)
                stingc = getConvexHull(stingc,len(stingc),flowRight)
            ### get leading edges
            cEdge = getEdgeFromContour(c,flowRight)
            stingEdge= getEdgeFromContour(stingc,flowRight)
            edges = (cEdge,stingEdge)

            ### get ROI 
            ROI = getROI(orig,cEdge,stingEdge,draw=draw,plot=False)
            
            ### try to correct corners
            if not flags['saturated']:
                try:
                    if flags['underexp']:
                        cutoff = 25
                    else:
                        cutoff = 35

                    if flowRight:
                        if edges[1][-1,0,0]>edges[0][-1,0,0]:
                            cn = combineEdges(edges[0],edges[1],cutoff=cutoff)
                            edges = (cn,stingc)
                    else:
                        if edges[1][-1,0,0]<edges[0][-1,0,0]:
                            cn = combineEdges(edges[0],edges[1],cutoff=cutoff)
                            edges = (cn,stingc)
                    if draw:
                        cv.drawContours(orig, cn, -1, (0,0,255), 1)
                    flags["cornerFailed"] = False
                except:
                    flags["cornerFailed"] = True
                    log.write('corner correction failed')
        except TypeError: # catch when return type is None
            log.write('failed HSV edge detection')
            return None

    if annotate:
        annotateImage(orig,flags)
    if plot:
        plt.figure()
        plt.title("Plotting getModelProps %s"%str(frameID))
        rgb = orig[...,::-1].copy()
        plt.imshow(rgb)
        plt.show()

    return edges, ROI, (th,cx,cy), flowRight,flags

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
    


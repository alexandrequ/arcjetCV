import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from classes.Functions import getOrientation,classifyImageHist,getROI
from classes.Functions import contoursGRAY,contoursHSV,combineEdges
from classes.Functions import getConvexHull,getEdgeFromContour

class Logger(object):
    def __init__(self,filename,PRINT=True,FILEIO=False,prefix=''):
        self.filename=filename
        self.prefix = prefix
        self.print = PRINT
        self.fileio = FILEIO
        
    def write(self,line):
        if self.print:
            print(self.prefix+line.__str__())
        if self.fileio:
            fh = open(self.filename,'a')
            fh.write(self.prefix+line.__str__()+'\n')
            fh.close()
##    modelpercent
##    stingpercent
##    contourChoice('default','GRAY','HSV')
##    flowRight ('left','right')
##    hueMin (75,85,95), hueMax (121,140,170)
##    intensityMin, intensityMax 
##    cornerCutoff (15,25,35)

def getModelProps(orig,frameID,log=None,plot=False,draw=True,
                  verbose=False,annotate=True,modelpercent=.005,
                  stingpercent=.05,contourChoice='default',flowDirection=None,
                  minHue=None,maxHue=None,intensityMin=None,intensityMax=None,
                  cornerCutoff=None):
    ### Initialize logfile
    if log is None:
        log = Logger('getModelProps.log',prefix="%s: "%str(frameID))

    ### Classify image
    flags,thresh = classifyImageHist(orig,verbose=verbose,
                                     modelpercent=modelpercent,
                                     stingpercent=stingpercent)
    if verbose:
        log.write(flags)

    ### Exit if no model visible
    if flags['modelvis'] == False:
        return None

    ### Set contour type
    if contourChoice=='default':
        ### Sting is visible, image is bright, use grayscale countours
        if flags['saturated'] or flags['stingvis']:
            contourChoice = 'GRAY'
        else:
            contourChoice = 'HSV'

    ### Set intensity limits
    if intensityMin == None:
        intensityMin = thresh
    if flags['overexp']:
        intensityMin = 230
    if flags['saturated']:
        intensityMin = 243
    if flags['underexp']:
        intensityMin = max(80,min(thresh,200))
    if intensityMax == None:
        intensityMax = 255

    #print(slimit, "%f intensityMin"%intensityMin)
    ### Set hue limits
    if minHue == None or maxHue == None:
        ### HSV default params
        if flags['overexp']:
            minHue=95;maxHue=140
        elif flags['underexp']:
            minHue=75;maxHue=170
        else:
            minHue=85;maxHue=140

    ### Set corner cutoff limits
    if cornerCutoff == None:
        if flags['underexp']:
            cornerCutoff = 25
        elif flags['overexp']:
            cornerCutoff = 15
        else:
            cornerCutoff = 35
        
    ### Extract contours
    try:
        if contourChoice == 'GRAY' or contourChoice == 'GREY':
            c,stingc = contoursGRAY(orig,intensityMin,log=log,
                                    draw=True,plot=plot)
        else:
            c,stingc = contoursHSV(orig,plot=plot,draw=True,log=log,
                                   minHue=minHue,maxHue=maxHue,intensityMin=intensityMin,
                                   modelpercent=modelpercent,flags=flags)
    except:
        log.write('failed contour%s edge detection'%contourChoice)
        return None        

    ### Get contour moments
    try:
        ### Estimate orientation, center of mass
        th,cx,cy,(x,y,w,h),flowRight = getOrientation(c)
        if draw:
            cv.circle(orig,(int(cx),int(cy)),4,(0,255,0))
    except:
        log.write('failed contour moment calculation')
        return None

    ### Determine flow direction
    if flowDirection == 'left':
        flowRight = False
    elif flowDirection == 'right':
        flowRight = True

    ### Get leading edges & ROI
    try:
        cEdge = getEdgeFromContour(c,flowRight)
        stingEdge= getEdgeFromContour(stingc,flowRight)
        edges = (cEdge,stingEdge)

        ROI = getROI(orig,cEdge,stingEdge,draw=draw,plot=plot)
        flags["EdgeExtractionFailed"] = False
    except:
        flags["EdgeExtractionFailed"] = True
        log.write('front edge extraction failed')
    ### try to correct corners
    try:
        if flowRight:
            if edges[1][-1,0,0]>edges[0][-1,0,0]+5:
                cn = combineEdges(edges[0],edges[1],cutoff=cornerCutoff)
                edges = (cn,stingc)
        else:
            if edges[1][-1,0,0]<edges[0][-1,0,0]-5:
                cn = combineEdges(edges[0],edges[1],cutoff=cornerCutoff)
                edges = (cn,stingc)
        if draw:
            cv.drawContours(orig, cn, -1, (0,0,255), 1)
        flags["cornerFailed"] = False
    except:
        flags["cornerFailed"] = True
        log.write('corner correction failed')
        
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
    


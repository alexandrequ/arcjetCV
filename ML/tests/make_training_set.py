# Make training set for convolutional neural net 
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from glob import glob
from utils.Frame import getModelProps
from utils.Models import Video, FrameMeta
from utils.Functions import splitfn,contoursHSV,contoursGRAY,getROI
from utils.Functions import getEdgeFromContour,combineEdges
from utils.grabcut import GrabCut

##fname = "AHF335Run001_EastView_1.mp4"
##fname = "IHF360-005_EastView_3_HighSpeed.mp4"
##fname = "IHF360-003_EastView_3_HighSpeed.mp4"

folder = "../data/video/"
filemask = folder+ "*.mp4"
paths = glob(filemask)
SELECT_FRAMES = True
MANUAL_ADJUST = False
MAKE_MASKS = True
MAKE_SHOCK_MASKS = True

#### Select 8 frames per video & create pngs & meta files
if SELECT_FRAMES:
    counter =0
    for path in paths:
        folder, name, ext = splitfn(path)
        fname = name+ext;print("### "+ name)
        
        meta = FrameMeta(os.path.join(folder,name+'.meta'))
        #print(meta.WIDTH,meta.HEIGHT)

        cap = cv.VideoCapture(path)
        
        for fnumber in np.linspace(meta.FIRST_GOOD_FRAME,meta.LAST_GOOD_FRAME,8):
            # Capture frame-by-frame
            
            cap.set(cv.CAP_PROP_POS_FRAMES,int(fnumber));
            ret, frame = cap.read()

            fd = "./train_frames/"
            fname = "frame_%04d"%counter
            cv.imwrite(fd+ fname+".png",frame)

            ### Add meta files
            metapath = fd+fname+'.meta'
            meta.path = metapath
            meta.folder = fd
            meta.name = fname
            meta.CALFRAME_INDEX = int(fnumber)
            print(meta)
            meta.write()
            
            counter += 1

        # When everything done, release the capture
        cap.release()

# Create metadata for all test frames
if MANUAL_ADJUST:
    folder = "./train_frames/"
    fname = "frame_0080"
    a = FrameMeta(folder+fname+".meta")
    frame = cv.imread(folder+fname+".png", flags=cv.IMREAD_COLOR)
    a.processFrame(frame,hueMin=95,hueMax=140,iMin=130,iMax=256,
                  GRAY=False,HSVSQ=True,fD='right')

if MAKE_MASKS:
    maskfolder = "./train_masks/"
    framefolder = "./train_frames/"
    filemask = framefolder+ "*.png"
    paths = glob(filemask)
    n=4
    for p in [79]:#range(0+8*n,8*(n+1)):
        path = framefolder + 'frame_%04d.png'%p
        folder, name, ext = splitfn(path)
        fname = name+ext;print("### "+ name)
        frame = cv.imread(os.path.join(folder,fname), flags=cv.IMREAD_COLOR)
        
        meta = FrameMeta(os.path.join(folder,name+'.meta'))
        # Operations on the frame
        draw = True
        plot=True
        verbose=False

        print(meta)
        meta.hueMin = 80
        meta.iMin = 70
        if meta.USE_GRAY:
             c,stingc = contoursGRAY(frame,meta.iMin,log=None,draw=False,plot=False)
             ROI = getROI(frame,c,c,draw=draw,plot=plot)

        if meta.USE_HSVSQ:
             c,stingc = contoursHSV(frame,minHue=meta.hueMin,maxHue=meta.hueMax,flags=None,
                                    modelpercent=.005,intensityMin=meta.iMin,draw=False,plot=True)
             ROI = getROI(frame,c,stingc,draw=draw,plot=plot)

        flowRight = True
        if meta.FLOW_DIRECTION == 'left':
            flowRight = False
        print(meta.FLOW_DIRECTION)

        cEdge = getEdgeFromContour(c,flowRight)
        stingEdge= getEdgeFromContour(stingc,flowRight)
        backEdge= getEdgeFromContour(stingc,not flowRight)
        edges = (cEdge,stingEdge)

        toDelete = []
        for i in range(0,len(backEdge)):
            if flowRight:
                if backEdge[i,0,0] <= max(edges[0][-1,0,0],edges[0][0,0,0]) and backEdge[i,0,1]<200:
                    toDelete.append(i)
            else:
                if backEdge[i,0,0] >= edges[0][-1,0,0]:
                    toDelete.append(i)
        backEdge = np.delete(backEdge,toDelete,axis=0)
        
        cn = None
        cornerCutoff=15
        
        if flowRight:
            if edges[1][-1,0,0]>edges[0][-1,0,0]+5:
                cn = combineEdges(edges[0],edges[1],cutoff=cornerCutoff)
                edges = (cn,stingc)
        else:
            if edges[1][-1,0,0]<edges[0][-1,0,0]-5:
                cn = combineEdges(edges[0],edges[1],cutoff=cornerCutoff)
                edges = (cn,stingc)
        if cn is None:
            frontc = c
        else:
            frontc = np.append(cn,backEdge,axis=0)        

        frontc = np.append(cEdge,backEdge,axis=0)
        cv.drawContours(frame, [frontc], -1, (255,255,0), 2)

        maskframe = np.zeros((meta.HEIGHT,meta.WIDTH,3),np.uint8) 
        cv.drawContours(maskframe, [frontc], -1, (1,1,1), cv.FILLED)
        #cv.imwrite(maskfolder+name+ext,maskframe)
        ax0 = plt.subplot(211)
        plt.imshow(frame)
        ax1 = plt.subplot(212)
        plt.imshow(maskframe[:,:,0])
        
        plt.title('maskframe %i'%p)
        plt.show()
        
if MAKE_SHOCK_MASKS:
    ### Load existing images
    maskfolder = "./train_masks/"
    framefolder = "./train_frames/"
    fpaths = sorted(glob(framefolder+ "*.png"))
    mpaths = sorted(glob(maskfolder+ "*.png"))

    for i in range(56,59):#range(0,len(fpaths)):
        frame = cv.imread(fpaths[i],1)
        modelmask = cv.imread(mpaths[i],1)
        modelmask[modelmask != 1]=0
        element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7), (3, 3))
        dilatation_dst = cv.dilate(modelmask, element)
        sub = frame.copy()
        sub[dilatation_dst>0] =0
        
        plt.imshow(sub)
        plt.show()
        uin = input("Shock extract? (y/n): ")

        if uin == 'y':
            outname = 'shockout.png'
            GrabCut().run(fn=sub,outname=outname,maskval=2)
            cv.destroyAllWindows()

            shockmask = cv.imread(outname,1)
            finalmask = modelmask + shockmask
            cv.imwrite(mpaths[i],finalmask)
        











    


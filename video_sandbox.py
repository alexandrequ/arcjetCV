import numpy as np
import cv2 as cv
from classes.Frame import getModelProps
from classes.Calibrate import splitfn
import matplotlib.pyplot as plt
from glob import glob


##fname = "AHF335Run001_EastView_1.mp4"
##fname = "IHF360-005_EastView_3_HighSpeed.mp4"
##fname = "IHF360-003_EastView_3_HighSpeed.mp4"

#folder = "video/AHF335/"

folder = "video/IHF360/"
mask = folder + "IHF360-005_EastView_3_HighSpeed.mp4"

folder = "video/IHF338/"
mask = folder + "*006_EastView_1.mp4"  # default

folder = "video/HyMETS/"
mask = folder + "PS12*.mp4"  # default

paths = glob(mask)

WRITE_VIDEO = False
WRITE_PICKLE = False
SHOW_CV = True
FIRST_FRAME = 0#+303
LAST_FRAME = 2706

MODELPERCENT = 0.012
STINGPERCENT = 0.5
CC = 'HSV'
fD = 'left'
iMin = 150
iMax = None#255
hueMin = 60#95
hueMax = 170#140

for path in paths:    
    pth, name, ext = splitfn(path)
    fname = name+ext;print("### "+ name)

    cap = cv.VideoCapture(path)
    ret, frame = cap.read(); h,w,chan = np.shape(frame)

    if WRITE_VIDEO:
        vid_cod = cv.VideoWriter_fourcc('m','p','4','v')
        output = cv.VideoWriter(folder+"edit_"+fname[0:-4]+'.avi', vid_cod, 100.0,(w,h))
        
    nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)
    cap.set(cv.CAP_PROP_POS_FRAMES,FIRST_FRAME);
    counter=FIRST_FRAME
    myc=[]
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret==False:
            print("No more frames")
            break
        
        # Operations on the frame
        if SHOW_CV:
            draw = False
            plot=False
            verbose=False
        else:
            draw = True
            plot=True
            verbose=True
            
        ret = getModelProps(frame,counter,draw=draw,plot=plot,verbose=verbose,
                            modelpercent=MODELPERCENT,stingpercent=STINGPERCENT,
                            contourChoice=CC,flowDirection=fD,
                            intensityMin=iMin,intensityMax=iMax,
                            minHue=hueMin,maxHue=hueMax)

        if ret != None:
            (c,stingc), ROI, (th,cx,cy), flowRight,flags = ret
            (xb,yb,wb,hb) = cv.boundingRect(c)
            area = cv.contourArea(c)
            cv.rectangle(frame,(xb,yb,wb,hb),(255,255,255),3)
            cv.drawContours(frame, c, -1, (0,255,255), 3)

            ### Save contours and useful parameters
            myc.append([counter,cy,hb,area,c,flags])
        if WRITE_VIDEO:
            output.write(frame)

        if SHOW_CV:
            magnus =1
            cv.imshow(name,frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        counter +=1
      
    # When everything done, release the capture
    cap.release()
    if WRITE_VIDEO:
        output.release()
    cv.destroyAllWindows()

    if WRITE_PICKLE:
        import pickle
        fout = open(folder+fname[0:-4] +'_edges.pkl','wb')
        pickle.dump(myc,fout)
        fout.close()

    #input("Next video?")

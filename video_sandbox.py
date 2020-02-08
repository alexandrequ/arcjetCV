import numpy as np
import cv2 as cv
from Frame import getModelProps
from Functions import classifyImageHist
import matplotlib.pyplot as plt

folder = "video/IHF338/"
fname = "AHF335Run001_EastView_1.mp4"
fname = "IHF338Run001_WestView_2.mp4"

##folder = "video/"
##fname = "IHF360-005_EastView_3_HighSpeed.mp4"
#fname = "IHF360-003_EastView_3_HighSpeed.mp4"
cap = cv.VideoCapture(folder+fname)
ret, frame = cap.read(); h,w,chan = np.shape(frame)
WRITE_VIDEO = True
WRITE_PICKLE = True
SHOW_CV = True
FIRST_FRAME = 550

if WRITE_VIDEO:
    vid_cod = cv.VideoWriter_fourcc('m','p','4','v')
    output = cv.VideoWriter(folder+"edit_"+fname[0:-4]+'.avi', vid_cod, 100.0,(w,h))
    
nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv.CAP_PROP_FPS)
cap.set(cv.CAP_PROP_POS_FRAMES,FIRST_FRAME);
counter=0
myc,xpts=[],[]
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret==False:
        break
    
    counter +=1
##    if counter%10==0:
##        print(counter)
    
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
                        modelpercent=.001)

    if ret != None:
        (c,stingc), ROI, orientation, flowRight,flags = ret
        box = cv.boundingRect(c)
        cv.rectangle(frame,box,(255,255,255),3)
        cv.drawContours(frame, c, -1, (0,255,255), 3)
        myc.append([c,flags,counter])
    if WRITE_VIDEO:
        output.write(frame)

    if SHOW_CV:
        # Display the resulting frame
        #cv.imwrite('frame%03d.png'%counter,frame)
##        src = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
##        dd = cv.GaussianBlur(src, (3, 3), 0)
##        ret1,th1 = cv.threshold(dd,2,256,cv.THRESH_BINARY)
####        ax2.imshow(frame[...,::-1])
##        # Find contours
##        contours, hierarchy = cv.findContours(th1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
##        c = max(contours, key = cv.contourArea)
##        if draw:
##            cv.drawContours(frame, c, -1, (255,0,0), 1)
##
##        ind = np.sqrt(c[:,0,0]**2 + c[:,0,1]**2).argmin()
##        print(c[ind,0,0],c[ind,0,1])
##        xpts.append(c[ind,0,0])
##        cv.circle(frame, (c[ind,0,0],c[ind,0,1]), 5, (255,0,255), 2)
        cv.imshow('img2',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
        
        
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

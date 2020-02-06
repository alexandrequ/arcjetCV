import numpy as np
import cv2 as cv
from Frame import getModelProps
from Functions import classifyImageHist
import matplotlib.pyplot as plt

folder = "video/"
fname = "AHF335Run001_EastView_1.mp4"
fname = "IHF360-005_EastView_3_HighSpeed.mp4"
#fname = "IHF360-003_EastView_3_HighSpeed.mp4"
cap = cv.VideoCapture(folder+fname)
ret, frame = cap.read(); h,w,c = np.shape(frame)
WRITE_VIDEO = False
WRITE_PICKLE = False
SHOW_CV = True
SHOW_MATPLOTLIB = True
FIRST_FRAME =0

if WRITE_VIDEO:
    vid_cod = cv.VideoWriter_fourcc('m','p','4','v')
    output = cv.VideoWriter(folder+"edit_"+fname[0:-4]+'.avi', vid_cod, 100.0,(w,h))

nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv.CAP_PROP_FPS)
cap.set(cv.CAP_PROP_POS_FRAMES,FIRST_FRAME);
counter=0
myc=[]
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
    else:
        draw = True
        
    ret = getModelProps(frame,counter,draw=draw,annotate=True)
    if ret != None:
        (c,stingc), ROI, orientation, flowRight,flags = ret
        box = cv.boundingRect(c)
        cv.rectangle(frame,box,(255,255,255),2)
        cv.drawContours(frame, c, -1, (0,255,255), 2)
        myc.append([c,flags,counter])
    if WRITE_VIDEO:
        output.write(frame)

    if SHOW_CV:
        # Display the resulting frame
        #cv.imwrite('frame%03d.png'%counter,frame)
        cv.imshow('img2',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    if SHOW_MATPLOTLIB and not SHOW_CV:
        ax2= plt.subplot(1,1,1)
        ax2.imshow(frame[...,::-1])
        plt.show()
        
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

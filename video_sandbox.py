import numpy as np
import cv2 as cv
from Frame import getModelProps
from Functions import classifyImageHist
import matplotlib.pyplot as plt

folder = "video/"
fname = "AHF335Run001_EastView_1.mp4"
fname = "IHF360-005_EastView_3_HighSpeed.mp4"
fname = "IHF360-003_EastView_3_HighSpeed.mp4"
cap = cv.VideoCapture(folder+fname)
ret, frame = cap.read(); h,w,c = np.shape(frame)

vid_cod = cv.VideoWriter_fourcc('M','J','P','G')
output = cv.VideoWriter(folder+"edit_"+fname, vid_cod, 60.0,(w,h))

nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv.CAP_PROP_FPS)
cap.set(cv.CAP_PROP_POS_FRAMES,361);
counter=0
myc=[]
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret==False:
        break
    
    counter +=1
    print(counter)
    
    # Our operations on the frame come here
    ret = getModelProps(frame,plot=False)
    if ret != None:
        (c,stingc), ROI, orientation, flowRight = ret
        box = cv.boundingRect(c)
        cv.rectangle(frame,box,(255,255,255))
        cv.drawContours(frame, c, -1, (0,255,255), 3)
        myc.append([c[:,0,:],counter])
    output.write(frame)
        
##    color = ('b','g','r')
##    ax1= plt.subplot(2,1,1)
##    plt.xlim([10,256])
##    plt.ylim([0,0.2])
##    ax2= plt.subplot(2,1,2)
##    
##    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
##    flags = classifyImageHist(gray)
##    for i,col in enumerate(color):
##        histr = cv.calcHist([gray],None,None,[256],[0,256])
##        npx = frame[:,:,0].size
##        ax1.plot(histr/npx,color = col)
##    ax2.imshow(frame[...,::-1])
##    plt.show()
        
    # Display the resulting frame
##    cv.imwrite('frame%03d.png'%counter,frame)
    cv.imshow('img2',frame)
    #print(np.shape(frame))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
output.release()
cv.destroyAllWindows()

import pickle
fout = open(fname[0:-4] +'_edges.pkl','wb')
pickle.dump(myc,fout)
fout.close()

import numpy as np
import cv2 as cv
from Frame import getModelROI

fname = "video/AHF335Run001_EastView_1.mp4"
#fname = "video/trim.mov"
cap = cv.VideoCapture(fname)
counter=0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret==False:
        break

    # Our operations on the frame come here
    ret = getModelROI(frame)
    if ret != None:
        ROI, boxes, orientation, flowRight = ret
        
    # Display the resulting frame
##    cv.imwrite('frame%03d.png'%counter,frame)
    counter +=1
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

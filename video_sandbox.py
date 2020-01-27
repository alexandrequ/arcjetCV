import numpy as np
import cv2 as cv
from Frame import getModelROI

fname = "video/AHF335Run001_EastView_1.mp4"
cap = cv.VideoCapture(fname)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret==False:
        break

    # Our operations on the frame come here
    ret = getModelROI(frame,low=140,high=255)
    if ret != None:
        x,y,w,h, centerline = ret
        
    # Display the resulting frame
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

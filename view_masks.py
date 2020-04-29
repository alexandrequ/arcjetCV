# Make training set for convolutional neural net 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob

mfolder = "./train_masks/"
ffolder = "./train_frames/"
mmask = mfolder+ "*.png"
paths = glob(mmask)

for p in sorted(paths):
    mask = cv.imread(p, flags=cv.IMREAD_COLOR)
    mask *= 100
    mask[:,:,0:1]= 50
    frame = cv.imread(p.replace('mask','frame'), flags=cv.IMREAD_COLOR)
    added_image = cv.addWeighted(frame,0.4,mask,0.6,0)

    cv.imshow(p,added_image)
 
    cv.waitKey(0) # waits until a key is pressed
    cv.destroyAllWindows() # destroys the window showing image

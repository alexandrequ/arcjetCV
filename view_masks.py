# Make training set for convolutional neural net 
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from glob import glob

mfolder = "./train_masks/"
ffolder = "./train_frames/"
vfolder = "./validation/"
ffolder = vfolder
mmask = mfolder+ "*.png"
paths = glob(vfolder+"*.png")

n=128

for p in sorted(paths)[0:]:
##    mask = cv.imread(p, flags=cv.IMREAD_COLOR)
##    origmask = mask.copy()
##    mask[:,:,2] *= (mask[:,:,2] == 1).astype(np.uint8)*int(255)
##    mask[:,:,1] *= (mask[:,:,1] == 2).astype(np.uint8)*int(255)
    frame = cv.imread(p.replace('mask','frame'), flags=cv.IMREAD_COLOR)
##    added_image = cv.addWeighted(frame,0.4,mask,0.5,0)
##    cv.imshow(p,added_image)
## 
##    cv.waitKey(0) # waits until a key is pressed
##    cv.destroyAllWindows() # destroys the window showing image

    fn =int(p[-6:-4])
    
    if fn <16:
        cv.imwrite(ffolder + 'cropped/frame_%04d.png'%fn,cv.resize(frame[:,:720,:],(n,n) ))
        #cv.imwrite(mfolder + 'cropped/frame_%04d.png'%fn,cv.resize(origmask[:,:720,:], (n,n) ))
    if fn >=16 and fn <24:
        cv.imwrite(ffolder + 'cropped/frame_%04d.png'%fn,cv.resize(frame[:,360:360+1080,:],(n,n)))
        #cv.imwrite(mfolder + 'cropped/frame_%04d.png'%fn,cv.resize(origmask[:,360:360+1080,:],(n,n)))
    if fn >=24 and fn <40:
        cv.imwrite(ffolder + 'cropped/frame_%04d.png'%fn,cv.resize(frame[:,:720,:],(n,n) ))
        #cv.imwrite(mfolder + 'cropped/frame_%04d.png'%fn,cv.resize(origmask[:,:720,:], (n,n) ))
    if fn >=40 and fn <48:
        cv.imwrite(ffolder + 'cropped/frame_%04d.png'%fn,cv.resize(frame[:,360:360+1080,:],(n,n)))
        #cv.imwrite(mfolder + 'cropped/frame_%04d.png'%fn,cv.resize(origmask[:,360:360+1080,:],(n,n)))
    if fn >=48 and fn <72:
        cv.imwrite(ffolder + 'cropped/frame_%04d.png'%fn,cv.resize(frame[:,:720,:],(n,n) ))
        #cv.imwrite(mfolder + 'cropped/frame_%04d.png'%fn,cv.resize(origmask[:,:720,:], (n,n) ))
    if fn >=72:
        cv.imwrite(ffolder + 'cropped/frame_%04d.png'%fn,cv.resize(frame[:,200:920,:],(n,n) ))
        #cv.imwrite(mfolder + 'cropped/frame_%04d.png'%fn,cv.resize(origmask[:,200:920,:], (n,n) ))        
    

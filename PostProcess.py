import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
from Functions import smooth,interpolateContour
from Calibrate import splitfn
from scipy.interpolate import splev, splprep, interp1d
from glob import glob

folder = "video/IHF338/"

mask = folder + 'IHF338Run005_WestView*_edges.pkl'  # default
paths = glob(mask)

### Units
rnorms = [-.75,-.5,0,.5,0.75]
labels = ['75% radius','50% radius','Apex']
fps = 240
sample_radius = 4 #inches
x0 = 500
### Visualization
PLOTXY  = True
PLOTTIME= True
VERBOSE = True

for path in paths:    
    pth, name, ext = splitfn(path)
    fname = name+ext;print("### "+ name)

    ### Load pickle file
    fin = open(path,'rb');myc = np.array(pickle.load(fin));fin.close()

    ### Parse file
    skip = 4
    t0,tf = myc[0,0],myc[-1,0]
    time= myc[skip:-skip,0].astype(np.int16)
    cy = myc[skip:-skip,1].astype(np.float)
    dy = myc[skip:-skip,2].astype(np.float)
    area = myc[skip:-skip,3].astype(np.float)
    cntrs = myc[skip:-skip,4]
    flagList = myc[skip:-skip,5]

    R_px = dy.max()/2.
    
    ### Filter by area,time
    finite_area = area > 1200
    ddt_area = np.append(np.zeros(1),abs(np.diff(area)) < 1200)
    
    goodinds = np.nonzero(finite_area*ddt_area)[0]
    
    ### Loop through pickle file
    recess_pos =[]
    for ind in goodinds:
        c = cntrs[ind]
        con = cv.convexHull(c)
        # offset vertical motion
        xi,yi = con[:,0,0],con[:,0,1]
        x,y = xi-x0,yi-cy[ind]

        # interpolate desired radial positions
        f = interp1d(y[2:-1], x[2:-1], kind='cubic')
        xi = f(np.array(rnorms)*R_px)
        recess_pos.append(xi)
        
##        if PLOTXY and (ind==goodinds[0] or ind==goodinds[-1]):
        plt.plot(y*sample_radius/R_px,x*sample_radius/R_px,'o-')            

    rp = np.array(recess_pos)
    if PLOTXY:
        for i in range(0,len(rnorms)):
            ys = rnorms[i]*R_px*np.ones(len(rp))
            plt.plot(ys*sample_radius/R_px,rp[:,i]*sample_radius/R_px,'x')
        plt.show()
    if PLOTTIME:
        t = time[goodinds]
        for i in range(0,len(rnorms)):
            plt.plot(t,rp[:,i]*sample_radius/R_px,'-',label="%f"%rnorms[i])
        plt.legend(loc=0)
        plt.show()

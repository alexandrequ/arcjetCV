import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
from classes.Functions import smooth,interpolateContour,getOrientation
from classes.Calibrate import splitfn
from scipy.interpolate import splev, splprep, interp1d
from glob import glob

folder = "video/IHF338/"
mask = folder + 'IHF338Run004_WestView_3_edges.pkl'  # default
flowDirection = 'right'

folder = "video/HyMETS/"
mask = folder + 'PS07*.pkl'  # default
flowDirection = 'left'

paths = glob(mask)

### Units
rnorms = [-.75,-.5,0,.5,0.75]
#rnorms = [0]
labels = ['75% radius','50% radius','Apex','75% radius','50% radius']
fps = 240
minArea = 1200
sample_radius = 23.2156 #mm
skip=1
fitindex = 0
PLOTXY=True;PLOTTIME=True;VERBOSE=True
WRITETOFILE = True

for path in paths:
    pth, name, ext = splitfn(path)
    fname = name+ext;print("### "+ name)

    ### Load pickle file
    fin = open(path,'rb');myc = np.array(pickle.load(fin));fin.close()

    ### Parse file
    t0,tf = myc[0,0],myc[-1,0]
    time= myc[skip:-skip,0].astype(np.int16)
    cy = myc[skip:-skip,1].astype(np.float)
    dy = myc[skip:-skip,2].astype(np.float)
    area = myc[skip:-skip,3].astype(np.float)
    cntrs = myc[skip:-skip,4]
    flagList = myc[skip:-skip,5]

    R_px = dy.max()/2.

    ### Filter by area,time
    finite_area = area > minArea
    ddt_area = np.append(np.zeros(1),abs(np.diff(area)) < 1200)

    goodinds = np.nonzero(finite_area*ddt_area)[0]

    ### Loop through pickle file
    recess_pos,t =[],[]
    for ind in goodinds:
        c = cntrs[ind]
        con = cv.convexHull(c)

        ### Evaluate center/angle/direction of contour
        th,cx,cyy,(x,y,w,h),fR = getOrientation(con)
        if flowDirection == None:
            if fR:
                flowDirection = 'right'
            else:
                flowDirection = 'left'

        # offset vertical motion
        xi,yi = con[:,0,0],con[:,0,1]
        x,y = xi,yi-cyy#[ind]

        # adjust starting location
        if flowDirection == 'left':
            imin = y.argmin()
            y,x = np.roll(y,-imin),np.roll(x,-imin)

            # remove non-function corner points
            dyGreaterThanZero = np.append(np.zeros(1),np.diff(y) > 0)
            okay = np.nonzero(dyGreaterThanZero)[0]
        else:
            # remove non-function corner points
            dyLessThanZero = np.append(np.zeros(1),np.diff(y) < 0)
            okay = np.nonzero(dyLessThanZero)[0]

        # interpolate desired radial positions
        f = interp1d(y[okay], x[okay], kind='cubic')
        try:
            xi = f(np.array(rnorms)*R_px)
            recess_pos.append(xi)
            t.append(time[ind])
        except:
            xi = np.array(rnorms)*np.nan

##        # interpolate full curve 
##        try:
##            yi = np.linspace(min(y[okay]),max(y[okay]),200)
##            xis = f(yi)
##        except:
##            xis = yi*np.nan
            
        if PLOTXY:
            plt.plot(y*sample_radius/R_px,x*sample_radius/R_px,'-')            

    ### Cast list to array
    rp,t = np.array(recess_pos),np.array(t)
    sec = (t-t0)/fps
    xpos = rp*sample_radius/R_px

    ### remove >2 pixel jumps
    diff = abs(np.diff(rp,axis=0))>2
    isDiffGT2 = np.vstack((diff[0,:]*0,diff))
    rp[np.nonzero(isDiffGT2)[0]] = np.nan    
        
    if PLOTXY:
        for i in range(0,len(rnorms)):
            ys = rnorms[i]*R_px*np.ones(len(rp))
            plt.plot(ys*sample_radius/R_px,rp[:,i]*sample_radius/R_px,'x')
        plt.show()
    if PLOTTIME:
        err= 2*np.ones(len(sec))*sample_radius/R_px
        inds = np.arange(0,len(sec))
        err[inds>fitindex] /= 10000.
        for i in range(0,len(rnorms)):
            plt.plot(sec,xpos[:,i],'o',label="%f"%rnorms[i])
            coeff,cov = np.polyfit(sec, xpos[:,i], 1,cov=True,w=1/err)
            dm = np.sqrt(cov[0,0])
            #plt.plot(sec,sec*coeff[0]+coeff[1],'-')
            dsec = (tf-t0)/fps
            print("slope: %f +- %f, %f R"%(coeff[0],dm*6,rnorms[i]))
            print("recession: %f +- %f, %f R"%(dsec*coeff[0],2*sample_radius/R_px,rnorms[i]))
        plt.legend(loc=0)
        plt.show()
    if WRITETOFILE:
        np.savetxt(name+'.csv',xpos,delimiter=',',header = str(rnorms)[1:-1])
        np.savetxt(name+'_cy.csv',cy,delimiter=',',header = "vertical pos (px)")
        np.savetxt(name+'_time.csv',t-t0,delimiter=',',header = "frame number")


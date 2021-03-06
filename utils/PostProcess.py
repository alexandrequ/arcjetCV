import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
from Functions import smooth,interpolateContour
from Calibrate import splitfn
from scipy.interpolate import splev, splprep, interp1d
from glob import glob

    
def processModelProps(myc,rnorms, minArea=7000, sample_radius=1.,fps=1.,
                      fitindex=4,skip=4,
                      PLOTXY=False,PLOTTIME=False,VERBOSE=False):
    """
    Processes output of getModelProps to extract axial (horizontal) position,
    of 

    :param contour: opencv contour, shape(n,1,n)
    :param ninterp: integer, number of interpolation points
    :returns: output, interpolated contour positions
    """

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
        # offset vertical motion
        xi,yi = con[:,0,0],con[:,0,1]
        x,y = xi,yi-cy[ind]

        # interpolate desired radial positions
        f = interp1d(y[2:-1], x[2:-1], kind='cubic')
        try:
            xi = f(np.array(rnorms)*R_px)
            recess_pos.append(xi)
            t.append(time[ind])
        except:
            xi = np.array(rnorms)*np.nan
        recess_pos.append(xi)
        
        if PLOTXY:
            plt.plot(y*sample_radius/R_px,x*sample_radius/R_px,'-')            

    ### Cast list to array
    rp,t = np.array(recess_pos),np.array(t)
    t = time[goodinds]
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
            plt.plot((t-t0)/fps,rp[:,i]*sample_radius/R_px,'-',label="%f"%rnorms[i])
            coeff,cov = np.polyfit(sec, xpos[:,i], 1,cov=True,w=1/err)
            dm = np.sqrt(cov[0,0])
            plt.plot(sec,sec*coeff[0]+coeff[1],'-')
            dsec = (tf-t0)/fps
            print("slope: %f +- %f, %f R"%(coeff[0],dm*6,rnorms[i]))
            print("recession: %f +- %f, %f R"%(dsec*coeff[0],2*sample_radius/R_px,rnorms[i]))
            plt.legend(loc=0)
        plt.show()

    return sec,xpos

if __name__ == "__main__":
    folder = "video/IHF338/"

    mask = folder + 'IHF338Run004_WestView_?_edges.pkl'  # default
    paths = glob(mask)

    ### PostProcess inputs
    rnorms = [0]
    fps = 30
    minArea = 7000
    sample_radius = 4 #inches
    skip = 4
    fitindex = 150
    
    for path in paths:
        pth, name, ext = splitfn(path)
        fname = name+ext;print("### "+ name)

        ### Load pickle file
        fin = open(path,'rb');myc = np.array(pickle.load(fin));fin.close()
        t,rp = processModelProps(myc,rnorms,minArea=minArea,fps=fps,
                                 sample_radius=4.,fitindex=fitindex,
                                PLOTXY=True,PLOTTIME=True,VERBOSE=True)

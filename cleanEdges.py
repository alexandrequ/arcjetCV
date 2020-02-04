import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
from Functions import smooth
from scipy.interpolate import splev, splprep, interp1d


folder = "video/"
fname = "AHF335Run001_EastView_1.mp4"
fname = "IHF360-005_EastView_3_HighSpeed.mp4"
fname = "IHF360-003_EastView_3_HighSpeed.mp4"
ninterp = 200

x0 = [885-60]
dypx = [852,861]

fin = open(folder+fname[0:-4] +'_edges.pkl','rb')
myc = pickle.load(fin)
fin.close()

lowx,highx=[],[]
lowy,highy=[],[]
time=[]

dfx = np.zeros((len(myc),ninterp+1))
dfy = np.zeros((len(myc),ninterp+1))

for i in range(0,len(myc),1):
    # clip bad corners
    c, ind = myc[i]
    start= c[0:50,0].argmin()
    end = c[-50:,0].argmin() + len(c)-50 + 1
    c = c[start:end,:]

    # ROI -> find centerline
    time.append(ind)
    lowx.append(c[-1,0])
    lowy.append(c[-1,1])
    highx.append(c[0,0])
    highy.append(c[0,1])
    y0 = (c[0,1]+c[-1,1])/2.

    # offset vertical motion
    x,y = c[:,0]-x0,c[:,1]-y0

##    # add horiz tails
##    tx,ty = np.linspace(0,x[0],6),np.ones(6)*y[0]
##    bx,by = np.linspace(0,x[-1],6),np.ones(6)*y[-1]

    #cn = np.append(mt[:,np.newaxis,:],c[cutoff:-cutoff,:,:],axis=0)

    # identify duplicate points
    okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)

    # spline interpolation
    tck,u = splprep([x[okay],y[okay]],s=0)
    nu = np.linspace(0,1,ninterp)
    xi,yi = splev(nu,tck)
    dl = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    arclen = dl.sum()
    
    # save data into matrix format
    dfx[i,1:]=xi
    dfy[i,1:]=yi

    # frame number is first column
    dfx[i,0] = ind
    dfy[i,0] = ind
    
    plt.plot(y,x,'-')
plt.show()

### save interpolated edges
np.savetxt(folder+fname[0:-4] +'_X.csv', dfx,delimiter=',')
np.savetxt(folder+fname[0:-4] +'_Y.csv', dfy,delimiter=',')

dy = np.array(highy)-np.array(lowy)
plt.subplot(211)
plt.title('ypos')
plt.plot(smooth(lowy),'r-')
plt.plot(smooth(highy),'b-')
plt.subplot(212)
plt.title('xpos')
#plt.plot(smooth(lowx),'m-')
plt.plot(smooth(highx,window_len=20),'c-')
plt.show()
    

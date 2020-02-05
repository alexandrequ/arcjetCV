import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
from Functions import smooth,interpolateContour
from scipy.interpolate import splev, splprep, interp1d


folder = "video/"
fname = "AHF335Run001_EastView_1.mp4"
fname = "IHF360-005_EastView_3_HighSpeed.mp4"
#100fname = "IHF360-003_EastView_3_HighSpeed.mp4"
ninterp = 1000

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
    c, flags, ind = myc[i]
##    start= c[0:50,0].argmin()
##    end = c[-50:,0].argmin() + len(c)-50 + 1
##    c = c[start:end,:]

    # ROI -> find centerline
    time.append(ind)
    lowx.append(c[-1,0])
    lowy.append(c[-1,1])
    highx.append(c[0,0])
    highy.append(c[0,1])
    y0 = (c[0,1]+c[-1,1])/2.

    # offset vertical motion
    cout = interpolateContour(c[:,np.newaxis,:],ninterp)
    xi,yi = cout[:,0,0],cout[:,0,1]
    x,y = xi-x0,yi-y0
   
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
    

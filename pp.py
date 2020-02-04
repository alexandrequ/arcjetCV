import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
from Functions import smooth
from scipy.interpolate import splev, splprep, interp1d

folder = "video/"
fname = "IHF360-003_EastView_3_HighSpeed.mp4"

##folder= "video/"
##fname = "IHF360-005_EastView_3_HighSpeed.mp4"

dfx = np.loadtxt(folder+fname[0:-4]+"_X.csv",delimiter=',')
dfy = np.loadtxt(folder+fname[0:-4]+"_Y.csv",delimiter=',')

time = dfx[:,0]
cx = dfx[:,1:]
cy = dfy[:,1:]
clen = []

y75 = (cy[35,-1])*.75
y50 = (cy[35,-1])*.5
y00 = 0

c = np.zeros((200,1,2),dtype=np.int32)
for row in range(35,2000,10):
    c[:,0,0],c[:,0,1] = cy[row,:],cx[row,:]
    hull = cv.convexHull(c)
    y,x = hull[:,0,0],hull[:,0,1]
    plt.plot(x,y,'-')
    dl = np.sqrt(np.diff(cy[row,:])**2 + np.diff(cx[row,:])**2)
    arclen = dl.sum()
    print(time[row], cx[row,:].max())
    clen.append(arclen)
plt.show()

plt.plot(clen)
##plt.plot(time,cx[:,250],'r-')
##plt.plot(time,cx[:,500],'g-')
##plt.plot(time,cx[:,750],'b-')
plt.show()

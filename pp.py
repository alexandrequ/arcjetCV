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

x75,x50,x00 = [],[],[]

for row in range(0,len(cy)):
    i75 = np.where(cy[row,:]>y75)[0][0]
    x75.append(cx[row,i75])

    i50 = np.where(cy[row,:]>y50)[0][0]
    x50.append(cx[row,i50])

    x00.append(cx[row,:].max())

plt.plot(time,x75,'r-')
plt.plot(time,x50,'g-')
#plt.plot(time,x00,'b-')

plt.plot(time,cx[:,125],'r--')
plt.plot(time,cx[:,250],'g--')
plt.plot(time,cx[:,500],'b--')

##plt.plot(time,smooth(cx[:,125],window_len=48),'r-')
##plt.plot(time,smooth(cx[:,126],window_len=48),'g-')
##plt.plot(time,smooth(cx[:,500],window_len=48),'b-')
plt.show()

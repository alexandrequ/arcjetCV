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
okay = np.where(time != 0)
cx = dfx[:,1:]
cy = dfy[:,1:]
clen = []

##y75 = (cy[35,-1])*.75
##y50 = (cy[35,-1])*.5
##y00 = 0
##
##x75,x50,x00 = [],[],[]
##
##for row in range(0,len(cy)):
##    i75 = np.where(cy[row,:]>y75)[0][0]
##    x75.append(cx[row,i75])
##
##    i50 = np.where(cy[row,:]>y50)[0][0]
##    x50.append(cx[row,i50])
##
##    x00.append(cx[row,:].max())

##plt.plot(time,x75,'r-')
##plt.plot(time,x50,'g-')
##plt.plot(time,x00,'b-')

##plt.plot(time,cx[:,25],'r--')
##plt.plot(time,cx[:,50],'g--')
##plt.plot(time,cx[:,100],'b--')
pxpi = 214
inds = [25,50,100]
labels = ['75% radius','50% radius','Apex']

out = np.zeros((2312,6))
for i in range(0,len(inds)):
    
    j,label = inds[i],labels[i]
    x = cx[okay,j]
    a,b = ( (time[okay]-time[okay][0])/240)[150:-35],(x[0,150:-35]-x[0,150])/pxpi

    b=smooth(b,window_len=48)
##    plt.plot(x[0,:],'-',label=label)
    plt.fill_between(a,-b-1/pxpi,-b+1/pxpi,alpha=0.5,label=label)
    out[:,2*i] = a
    out[:,2*i+1]=b
    
plt.legend(loc=0)
plt.title(fname[0:-4].replace('_',' '))
#plt.xlim([0,a.max()])
plt.ylabel('Recession (in)')
plt.xlabel('Insertion Time (s)')
plt.show()

np.savetxt(fname[0:-4]+'_recess.csv',out, delimiter=',')


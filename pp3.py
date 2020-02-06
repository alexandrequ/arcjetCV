import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
from Functions import smooth
from scipy.interpolate import splev, splprep, interp1d

folder = "video/"
fname = "IHF360-003_EastView_3_HighSpeed.mp4"
pxpi = 214
rnorms = [0.75,.5,0]
labels = ['75% radius','50% radius','Apex']
ns,ne = 230,35
fps = 240

folder= "video/"
fname = "IHF360-005_EastView_3_HighSpeed.mp4"
ns,ne = 310,35

dfx = np.loadtxt(folder+fname[0:-4]+"_X.csv",delimiter=',')
dfy = np.loadtxt(folder+fname[0:-4]+"_Y.csv",delimiter=',')

time = dfx[:,0]-dfx[0,0]
cx = dfx[:,1:]
cy = dfy[:,1:]
clen = []

def getRadialPoint(cx,cy, rnorm):
    y = (cy[0])*rnorm
    ind = np.where(cy<y)[0][0]
    print(ind)
    xp,yp = cx[ind],cy[ind]
    return (xp,yp),ind


cpts = []
inds = []
for rnorm in rnorms:
    p,ind = getRadialPoint(cx[ns,:],cy[ns,:],rnorm)
    cpts.append(p); inds.append(ind)

out = np.zeros((len(cy),len(rnorms)))

for i in range(0,len(cpts)):
    xp,yp=[],[]
    for row in range(0,len(cy)):
        p = cpts[i]
        lp = np.sqrt( (cx[row,:]-p[0])**2 + (cy[row,:]-p[1])**2 )
        dl,ind = lp.min(),lp.argmin()
        #print(ind)
        x,y = cx[row,ind],cy[row,ind]
        xp.append(x); yp.append(y)
        out[row,i] = dl
    #plt.plot(time[ns:-ne]/fps,smooth(-out[ns:-ne,i]/pxpi,window_len=48),'-')
    plt.plot(yp[ns:-ne],xp[ns:-ne],'-')
#plt.show()



out = np.zeros((len(time[ns:-ne]),6))
for i in range(0,len(inds)):
    
    j,label = inds[i],labels[i]
    x,y = cx[:,j],cy[:,j]
    x0,y0 = x[ns:-ne]-cx[ns,j],y[ns:-ne]-cy[ns,j]
    a,b,c = ( (time[ns:-ne])/fps),np.sqrt(x0**2 )/pxpi,np.sqrt(y0**2 )/pxpi

    b = smooth(x[ns:-ne],window_len=48)
    c = smooth(y[ns:-ne],window_len=48)
    plt.plot(c,b,'-',label=label)
##    plt.fill_between(a,-b-1/pxpi,-b+1/pxpi,alpha=0.5,label=label)
    out[:,2*i] = a
    out[:,2*i+1]=b

plt.plot(cy[ns,:],cx[ns,:],'k-',label="initial")
plt.plot(cy[-ne,:],cx[-ne,:],'k--',label="final")
ax1=plt.gca()
ax1.set_aspect('equal')

plt.legend(loc=0)
plt.title(fname[0:-4].replace('_',' '))
##plt.xlim([0,a.max()])
##plt.ylabel('Recession (in)')
##plt.xlabel('Insertion Time (s)')

plt.ylabel('X (pixels)')
plt.xlabel('Y (pixels)')
plt.grid(True)
plt.tight_layout()
plt.show()
##
##np.savetxt(fname[0:-4]+'_recess.csv',out, delimiter=',')
##

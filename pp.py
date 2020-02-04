import numpy as np
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
from Functions import smooth
from scipy.interpolate import splev, splprep, interp1d

folder = "video/IHF360-003/"
fname = "IHF360-003_EastView_3_HighSpeed.mp4"

folder= "video/IHF360-005/"
fname = "IHF360-005_EastView_3_HighSpeed.mp4"

dfx = np.loadtxt(folder+fname[0:-4]+"_X.csv",delimiter=',')
dfy = np.loadtxt(folder+fname[0:-4]+"_Y.csv",delimiter=',')

time = dfx[:,0]
cx = dfx[:,1:]
cy = dfy[:,1:]

#for row in range(0,len(cx),100):

plt.plot(time,cx[:,250],'r-')
plt.plot(time,cx[:,500],'g-')
plt.plot(time,cx[:,750],'b-')
plt.show()

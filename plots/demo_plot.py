import numpy as np
import cv2 as cv
from classes.Frame import getModelProps
from classes.Calibrate import splitfn
import matplotlib.pyplot as plt
from glob import glob

folder = "../video/IHF338/"
mask = folder + "*004_WestView_3.mp4"  # default

paths = glob(mask)

myr = np.loadtxt('IHF338Run004_WestView_3_edges.csv',delimiter=',')
mcy = np.loadtxt('IHF338Run004_WestView_3_edges_cy.csv',delimiter=',')
myt = np.loadtxt('IHF338Run004_WestView_3_edges_time.csv',delimiter=',')
t0= 310

plt.plot((myt)/30,myr-1.153,'r^')

plt.fill([0,1.3,1.3,0], [-0.02,-0.02,.05,.05], 'c', alpha=0.2)
plt.text(.15, .022, 'sting\nmotion', fontsize=12)
plt.fill([1.3,4.4,4.4,1.3], [-0.02,-0.02,.05,.05], 'm', alpha=0.2)
plt.text(2.2, .022, 'Region 1', fontsize=12)
plt.fill([4.4,8,8,4.4], [-0.02,-0.02,.05,.05], 'r', alpha=0.2)
plt.text(5.2, .022, 'Region 2', fontsize=12)
plt.fill([8,10,10,8], [-0.02,-0.02,.05,.05], 'g', alpha=0.2)
plt.text(8.4, .022, 'Region 3', fontsize=12)


plt.xlim([0,10])
plt.ylim([-.018,.04])
plt.title('HEEET recession tracking (IHF338_04_3)')
plt.xlabel('Time (s)')
plt.ylabel('Recession (in)')
plt.tight_layout()
plt.show()

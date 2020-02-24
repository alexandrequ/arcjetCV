import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from glob import glob

from skimage.filters.rank import entropy
from skimage.morphology import disk

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def build_filters(ksize=9,nangles=12,sigma=3,lmbda=3.5,gamma=.46,psi=0):
    filters = []
    for theta in np.linspace(0, np.pi, nangles):
        kern = cv.getGaborKernel((ksize, ksize), sigma, theta, lmbda, gamma, psi,
                             ktype=cv.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters
 
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv.filter2D(img, cv.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def textureFilter(img,params=None):
    if params is None:
        x= [9,12,2.906,3.485,.47]
    else:
        x = params
    filters = build_filters(ksize=int(x[0]),nangles=int(x[1]),sigma=x[2],
                            lmbda=x[3],gamma=x[4],psi=0)
    res1 = process(img, filters)
    return res1

def fitTextureParams(filemask):
    paths = glob(filemask)
    fits = []
    for path in paths:

        pth, name, ext = splitfn(path)

        ### Read in images
        img = cv.imread(name+'.tif',0)
        mask = cv.imread(name+'.png')

        ### Compare with mask
        b,g,r = cv.split(mask)
        bpx,gpx,rpx = cv.countNonZero(b),cv.countNonZero(g),cv.countNonZero(r)

        def getContrast(x):
            ### Apply Gabor filters
            res1 = textureFilter(img,x)
            
            bres = cv.bitwise_and(res1,res1,mask=b)
            gres = cv.bitwise_and(res1,res1,mask = g)
            rres = cv.bitwise_and(res1,res1,mask = r)

            bavg,gavg = bres.sum()/bpx, gres.sum()/gpx
            return 1/(bavg - gavg)

        x0 = np.array([9,12,2.906,3.485,.47])
        minmodel = minimize(getContrast, x0, method='nelder-mead',
                            options={'xatol': 1e-8, 'disp': True})
        print(minmodel.x)
        fits.append(minmodel.x)
    return fits

if __name__ == "__main__":

    filemask = "ms_8ply_000?.tif"
    paths = glob(filemask)

    for path in paths:
        pth, name, ext = splitfn(path)
        print(name)

        ### Read in images
        img = cv.imread(name+'.tif',0)
        mask = cv.imread(name+'.png')
        
        x = np.array([9,12,2.9,3.48,.47])
        texture = textureFilter(img,x)
        texture = cv.medianBlur(texture, 5, 0)

        ### noise removal
        stexture = cv.GaussianBlur(texture, (3, 3), 0)
        stexture = cv.GaussianBlur(stexture, (3, 3), 0)

        ### Threshold
        th1 = cv.inRange(texture,205,255,cv.THRESH_BINARY)
        th1 = cv.medianBlur(th1, 3, 0)
##        th1 = cv.GaussianBlur(th1, (3, 3), 0)
##        th1 = cv.inRange(th1,150,255,cv.THRESH_BINARY)
##        th1 = cv.medianBlur(th1, 7, 0)

        ### Erode & dialate to remove fluff
        kernel1 = np.ones((5,5), dtype=np.uint8)
        eroded = cv.erode(th1, kernel1)

        kernel2 = np.ones((2,2), dtype=np.uint8)
        dist = cv.dilate(eroded, kernel2)
        
        #res1 = entropy(img, disk(4))
        plt.imshow(dist)
        plt.show()


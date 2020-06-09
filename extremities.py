import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splprep, interp1d

img = cv2.imread('shock_detection/tests/frame_0012_out.png')   # you can read in images with opencv

def makeDyPositive(x,y):
    newx,newy =[x[0]],[y[0]]
    yval = y[0]
    for i in range(0,len(y)):
        # look for duplicate y vals
        if y[i] > yval:
            newy.append(y[i])
            newx.append(x[i])
            yval = y[i]
    return np.array(newx),np.array(newy)

def getEdge(mask,flowDirection):
    ### Get shield contour
    contours, hierarchy= cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cShield = max(contours, key=cv2.contourArea)

    ### Calculate contour centroid
    M = cv2.moments(cShield)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = (cX,cY)

    ### Bounding box
    (xb,yb,wb,dy) = cv2.boundingRect(cShield)
    c = np.vstack(cShield).squeeze()

    ### Contour params
    xi = c[:,0]
    yi = c[:,1]
    R_px = dy/2.
    yc  = (yi.max()+yi.min())/2.

    ### Get leading edge from contour
    if flowDirection == 'left':
        imin = yi.argmax()
        okay = range(imin,len(yi))
        okay = okay[::-1]
    else:
        imax = yi.argmax()
        okay = np.append([-1],range(0,imax+2))
    x,y = xi[okay],yi[okay]-yc
    x,y = makeDyPositive(x,y)

    return x,y,centroid,yc,R_px

def extremity(img_mask, flowDirection, rnorms=[-.75,-.5,0,.5,0.75]):

    ###### SHIELD ######
    sumMask = np.sum(img_mask)
    mask_shield = cv2.inRange(img_mask, 0.5, 1.5)
    hasShield = np.sum(mask_shield) > 0.002*sumMask

    if hasShield:
        # Get leading edge
        x,y,ShieldCen,ShieldY,ShieldR = getEdge(mask_shield,flowDirection)
        
        # interpolate desired radial positions
        fShield = interp1d(y, x, kind='linear')
        try:
            xShield = fShield(np.array(rnorms)*ShieldR)
        except:
            xShield = np.empty(5)
        yShield = np.array(rnorms)*ShieldR

    ###### SHOCK ######
    mask_shock = cv2.inRange(img_mask,  1.5, 2.5)
    hasShock = np.sum(mask_shock)> 0.002*sumMask

    if hasShock:
        # Get leading edge
        xs,ys,ShockCen,ShockY,ShockR = getEdge(mask_shock,flowDirection)
        
        # interpolate desired radial positions
        fShock = interp1d(ys, xs, kind='linear')
        xShock,yShock =[],[]
        for i in range(0,len(rnorms)):
            # check if shock-standoff can be measured
            if abs(rnorms[i])*ShockR < ShieldR:
                try:
                    xShock.append( fShock(rnorms[i]*ShieldR) )
                    yShock.append(rnorms[i]*ShieldR)
                except:
                    xShock.append(None)
                    yShock.append(None)
            else:
                xShock.append(None)
                yShock.append(None)

    return [x,y, xShield, yShield, ShieldY, ShieldR], [xs,ys, xShock, yShock, ShockY, ShockR]


if __name__ == "__main__":
    img_mask = cv2.imread('shock_detection/tests/frame_0012_out.png')
    img = cv2.imread('shock_detection/tests/frame_0012.png')
    flowDirection = "right"

##    img_mask = cv2.imread('shock_detection/tests/pika_large_out.png')
##    img = cv2.imread('frames/sample12.png')
##    flowDirection = "left"
    a,b= extremity(img_mask, flowDirection)

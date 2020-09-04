import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splprep, interp1d


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
    contours, hierarchy= cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    mask_shield = img_mask==1
    mask_bg = img_mask==0
    hasShield = np.sum(mask_shield) > .002*np.sum(mask_bg)

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
    else:
        x,y,ShieldCen,ShieldY,ShieldR = [None],[None],(None,None),None,None
        xShield,yShield = np.empty(5),np.empty(5)

    ###### SHOCK ######
    mask_shock = img_mask==2
    hasShock = np.sum(mask_shock)> .002*np.sum(mask_bg)

    if hasShock and hasShield:
        # Remove any trailing segments
        if flowDirection == 'right':
            xcutoff= ShieldCen[0] + int(ShieldR/4)
            mask_shock[:,xcutoff:] = 0
        else:
            xcutoff= ShieldCen[0] - int(ShieldR/4)
            mask_shock[:,:xcutoff] = 0

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
    else:
        xs,ys,ShockCen,ShockY,ShockR = [None],[None],(None,None),None,None
        xShock,yShock = np.empty(5),np.empty(5)

    return [x,y, xShield, yShield, ShieldY, ShieldR], [xs,ys, xShock, yShock, ShockY, ShockR]

### Convert integer segmentation to colored RGB image
def get_colored_segmentation_image(seg_arr, n_classes, colors=[(0,0,255),(0,255,0),(255,0,0)]):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')
    return seg_img

### Convert colored RGB image to integer segmentation 
def get_segmentation_from_colored_image(path):
    seg_arr = cv2.imread(path,1)
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width))    
    seg_img[:, :] += seg_arr[:,:,1] > 0 
    seg_img[:, :] += 2*(seg_arr[:,:,0] > 0)
    return seg_img 

if __name__ == "__main__":
##    img_mask = get_segmentation_from_colored_image('shock_detection/tests/frame_0012_out.png')
##    img = cv2.imread('shock_detection/tests/frame_0012.png')
##    flowDirection = "right"

    img_mask = get_segmentation_from_colored_image('shock_detection/tests/pika_large_out.png')
    img = cv2.imread('frames/sample12.png')
    flowDirection = "left"
    a,b= extremity(img_mask, flowDirection)

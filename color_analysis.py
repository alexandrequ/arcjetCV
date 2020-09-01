import cv2 as cv
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from mayavi import mlab

from models import FrameMeta, VideoMeta
from utils.Functions import convert_mask_gray_to_BGR, convert_mask_BGR_to_gray, splitfn

FRAME_FOLDER = "/home/magnus/Desktop/NASA/arcjetCV/data/sample_frames/"
MASK_FOLDER = "/home/magnus/Desktop/NASA/arcjetCV/data/sample_masks/"
regex = FRAME_FOLDER + "*.png"
framepaths = sorted(glob(regex))

def get_RGB_hist(R,G,B):
    mat = np.zeros((256,256,256))
    for i in range(0,len(R)):
        mat[R[i], G[i], B[i]] += 1
    
    X,Y,Z = np.mgrid[0:256,0:256,0:256]
    x,y,z,s = X[mat >0], Y[mat >0], Z[mat >0], mat[mat >0]
    return x,y,z, s

def box(lows,highs,color=(0,1,0)):
    x,y,z = lows
    xx,yy,zz = highs
    [xp,yp] = np.mgrid[x:xx, y:yy]
    zp = xp*0 + z
    mlab.mesh(xp, yp, zp, color=color)
    mlab.mesh(xp, yp, zp+zz-z, color=color)

    [xp,zp] = np.mgrid[x:xx, z:zz]
    yp = xp*0 + y
    # mlab.mesh(xp, yp, zp, color=color)
    # mlab.mesh(xp, yp+yy-y, zp, color=color)

    [yp,zp] = np.mgrid[y:yy, z:zz]
    xp = yp*0 + x
    mlab.mesh(xp, yp, zp, color=color)
    # mlab.mesh(xp+xx-x, yp, zp, color=color)

def filter_hsv_ranges(hsv,ranges,show_range=False):
    ### Evaluate multiple ranges
    maskHSV = np.zeros((hsv.shape[0],hsv.shape[1]),dtype=np.uint8)
    for i in range(0,len(ranges[0])):
        mask = cv.inRange(hsv, ranges[0][i],ranges[1][i])
        maskHSV = cv.bitwise_or(mask,maskHSV)
        
        if show_range:
            box(ranges[0][i],ranges[1][i])
    return maskHSV
    #contours,hierarchy = cv.findContours(maskHSV, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    

model_ranges  = np.array([[(0,0,208),   (155,0,155),  (13,20,101), (0,190,100),  (12,150,130)], 
                          [(180,70,255),(165,125,255),(33,165,255),(13,245,160),(25,200,250)]])
underexp_model =np.array([[(7,0,8)],[(20,185,101)]])

shock_ranges = np.array([[(125,78,115)], 
                         [(145,190,230)]])
dim_shocks = np.array([[(125,100,35), (140,30,20), (118,135,30)], 
                       [(165,165,150),(156,90,220),(128,194,125)]])

rm,gm,bm = np.zeros(1,np.uint8),np.zeros(1,np.uint8),np.zeros(1,np.uint8)
rs,gs,bs = np.zeros(1,np.uint8),np.zeros(1,np.uint8),np.zeros(1,np.uint8)
rb,gb,bb = np.zeros(1,np.uint8),np.zeros(1,np.uint8),np.zeros(1,np.uint8)
for path in framepaths[0:8]:
    folder, name, ext = splitfn(path)
    frame = cv.imread(path,1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # hsvsq = cv.cvtColor(hsv, cv.COLOR_BGR2HSV)
    # hsvsq[:,:,2] = hsv[:,:,2]
    frame = hsv
    framemeta = FrameMeta(folder+'/'+name+".meta")

    img = frame[framemeta.YMIN:framemeta.YMAX,framemeta.XMIN:framemeta.XMAX,:]
    # HSV section
    
    if 0:
        model_ranges = np.hstack((model_ranges,underexp_model))
    modelfilter = filter_hsv_ranges(img,model_ranges,show_range=False)
    if 1:
        shock_ranges = np.hstack((shock_ranges,dim_shocks))
        print("dim shock")
    shockfilter = filter_hsv_ranges(img,shock_ranges,show_range=False)
    
    npx = img.shape[0]*img.shape[1]
    print(modelfilter.sum()/npx/255)
    alpha = .5
    beta = (1.0 - alpha)
    shockmask = convert_mask_gray_to_BGR(shockfilter)
    dst = cv.addWeighted(img, alpha, shockmask, beta, 0.0)
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    plt.imshow(shockfilter)
    plt.subplot(133)
    plt.imshow(modelfilter)
    plt.show()

    R,G,B = img[:,:,0],img[:,:,1],img[:,:,2]

    mask = cv.imread(MASK_FOLDER+name+ext, 0)
    mask = mask[framemeta.YMIN:framemeta.YMAX,framemeta.XMIN:framemeta.XMAX]
    modelpx = mask==1
    shockpx = mask==2
    bgpx    = mask==0

    rm,gm,bm = np.append(rm,R[modelpx]), np.append(gm,G[modelpx]), np.append(bm,B[modelpx])
    rs,gs,bs = np.append(rs,R[shockpx]), np.append(gs,G[shockpx]), np.append(bs,B[shockpx])
    rb,gb,bb = np.append(rb,R[bgpx]), np.append(gb,G[bgpx]), np.append(bb,B[bgpx])
# shockfilter = filter_hsv_ranges(img,model_ranges,show_range=True)


### Plot histogram

# model histogram
x,y,z,s = get_RGB_hist(rm,gm,bm)
scale = 250./max(s)
mlab.points3d(x,y,z,s, color=(0,1,0), scale_factor=scale, mode = '2dcross')

# shock histogram
x,y,z,s = get_RGB_hist(rs,gs,bs)
scale = 50./max(s)
mlab.points3d(x,y,z,s, color=(0,0,1), scale_factor=scale, mode = '2dsquare')

# background histogram
x,y,z,s = get_RGB_hist(rb,gb,bb)
scale = 500./max(s)
if len(x) > 1e6:
    mlab.points3d(x,y,z,s, color=(1,0,0), scale_factor=scale, mask_points = 4, mode = '2dcircle')
else:
    mlab.points3d(x,y,z,s, color=(1,0,0), scale_factor=scale, mode = '2dcircle')

# model ranges


axes = mlab.axes(extent = [0, 255, 0, 255, 0, 255] )
axes.title_text_property.shadow_offset = np.array([ 1, -1])
axes.title_text_property.color = (1,1,1)
axes.label_text_property.color = (1,1,1)
axes.property.color = (1,1,1)

e = mlab.get_engine()
scene = e.current_scene
scene.scene.background = (0,0,0)
scene.scene.camera.position = [-247.99318777985366, -519.2248020944824, 253.13809152835512]
scene.scene.camera.focal_point = [151.86200786183366, 200.09580355449995, 69.33460549510208]
scene.scene.camera.view_angle = 30.0
scene.scene.camera.view_up = [0.08754419768742545, 0.20066108813464886, 0.9757413290211109]
scene.scene.camera.clipping_range = [356.3799974999291, 1262.8807023322072]
scene.scene.camera.compute_view_plane_normal()
mlab.show()
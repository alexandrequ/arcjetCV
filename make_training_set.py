# Make training set for convolutional neural net 
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from glob import glob
from models import Video, FrameMeta, VideoMeta
from utils.Functions import splitfn,contoursHSV,contoursGRAY,convert_mask_gray_to_BGR
from utils.Functions import getEdgeFromContour,convert_mask_BGR_to_gray, cropBGR, cropGRAY
from utils.Grabcut import GrabCut

##fname = "AHF335Run001_EastView_1.mp4"
##fname = "IHF360-005_EastView_3_HighSpeed.mp4"
##fname = "IHF360-003_EastView_3_HighSpeed.mp4"

ARCJETCV = "/home/magnus/Desktop/NASA/arcjetCV/"
VIDEO_FOLDER = ARCJETCV+"data/video/"
FRAME_FOLDER = ARCJETCV+"data/sample_frames/"
MASK_FOLDER = ARCJETCV+"data/sample_masks/"
MOSAIC_FRAME_FOLDER = ARCJETCV+"data/mosaic_frames/"
MOSAIC_MASK_FOLDER = ARCJETCV+"data/mosaic_masks/"

filemask = VIDEO_FOLDER + "HyMETS*.mp4"
videopaths = glob(filemask)

SELECT_FRAMES = False
MAKE_MASKS = False
EDIT_FRAMES = False
edit_frame_list = [94]
ADD_FRAMES = False
add_frame_list = [1,10,25,50,100,150,200,300]
GET_MOSAIC = True

#### Select frames per video, create pngs & meta files
def select_frames(videopaths, fd =FRAME_FOLDER, addframes=[], 
                  nframes=8, COUNTER = 0, offset=0):
    ''' selects frames from videos and creates metadata files for each one'''
    for path in videopaths:
        folder, name, ext = splitfn(path)
        fname = name+ext;print("### "+ name)
        
        vmeta = VideoMeta(os.path.join(folder,name+'.meta'))
        vid = Video(path)
        
        for fnumber in np.linspace(vmeta.FIRST_GOOD_FRAME, vmeta.LAST_GOOD_FRAME, nframes):
            # Capture frame-by-frame

            frame = vid.get_frame(int(fnumber)+offset)
            fname = "frame_%04d"%COUNTER
            cv.imwrite(fd+ fname+".png",frame)

            ### Add meta files
            metapath = fd+fname+'.meta'
            fmeta = FrameMeta(metapath,int(fnumber),vmeta)
            fmeta.VIDEO = vmeta.path
            fmeta.write()
            
            COUNTER += 1
        
        for fnumber in addframes:
            fnumber += vmeta.FIRST_GOOD_FRAME
            # Capture frame-by-frame
            frame = vid.get_frame(int(fnumber))
            fname = "frame_%04d"%COUNTER
            cv.imwrite(fd+ fname+".png",frame)

            ### Add meta files
            metapath = fd+fname+'.meta'
            fmeta = FrameMeta(metapath,int(fnumber),vmeta)
            fmeta.write()
            
            COUNTER += 1

        # When everything done, release the capture
        vid.close()

existing_frames = glob(FRAME_FOLDER+"*.png")
if SELECT_FRAMES:
    select_frames(videopaths,nframes=16,COUNTER=len(existing_frames), addframes=add_frame_list)
if ADD_FRAMES:
    select_frames(videopaths, nframes=0,COUNTER=len(existing_frames), addframes=add_frame_list)

########################################################################
################### CREATE MASKS FOR ALL SAMPLES #######################
########################################################################

def grab_model(frame):
    """use grabcut algorithm to extract model region

    Args:
        frame (opencv image): numpy ndarray

    Returns:
        modelmask: image mask with maskval =1 for model pixels, 0 for all others
    """
    outname = 'frame_out.png'
    GrabCut().run(fn=frame.copy(),outname=outname,maskval=1)
    cv.destroyAllWindows()
    modelmask = cv.imread(outname,1)
    return modelmask

def grab_shock(frame,modelmask):
    ''' use grabcut algorithm to extract shock region '''

    # zero out a dialated area around model region
    modelmask[modelmask != 1]=0
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15), (3, 3))
    dilatation_dst = cv.dilate(modelmask, element)
    sub = frame.copy()
    sub[dilatation_dst>0] =0

    # extract shock from modified frame
    outname = 'shock_out.png'
    GrabCut().run(fn=sub,outname=outname,maskval=2)
    cv.destroyAllWindows()

    # return collated masks
    shock_mask = cv.imread(outname,1)
    if shock_mask is not None:
        finalmask = modelmask + shock_mask
    else:
        finalmask = modelmask.copy()

    return finalmask

if MAKE_MASKS:
    ### Load existing images
    fpaths = sorted(glob(FRAME_FOLDER+ "*.png"))

    if EDIT_FRAMES:
        flist = edit_frame_list
    else:
        flist = range(60,len(fpaths))
    for i in flist:
        # Load sample frame
        framepath = os.path.join(FRAME_FOLDER, "frame_%04d.png"%i)

        if framepath in fpaths:
            frame = cv.imread(framepath,1)
            folder, name, ext = splitfn(framepath)
            fm = FrameMeta(os.path.join(folder,name+'.meta'))

            # Load mask frame
            mpath = MASK_FOLDER + name + ext
            if os.path.exists(mpath):
                graymask = cv.imread(mpath,0)
                modelmask = convert_mask_gray_to_BGR(graymask)
                alpha = .85
                beta = (1.0 - alpha)
                dst = cv.addWeighted(frame, alpha, modelmask, beta, 0.0)
                plt.imshow(graymask)
                plt.show()

                uin = input("Redo mask? (y/n): ")

                if uin == 'y':
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                    # Extract model first
                    modelmask =  grab_model(frame)
                    finalmask = grab_shock(frame,modelmask)
                
                    cv.imwrite(mpath,finalmask)
            else:
                # Extract model first
                modelmask =  grab_model(frame)
                finalmask = grab_shock(frame,modelmask)
            
                cv.imwrite(mpath,finalmask)


########################################################################
################ MOSAIC samples & masks to 128x128 #####################
########################################################################

def get_mosaic_set(framepath, maskpath, frame_outpath, mask_outpath, regex = "*.png"):
    paths = glob(framepath+regex)
    for element in paths:
        folder, name, ext = splitfn(element)
        frame = cv.imread(element)
        meta = FrameMeta(os.path.join(folder,name+'.meta'))
        mask = cv.imread(os.path.join(maskpath,name+ext))

        crop = meta.crop_range()
        img_crop = cropBGR(frame, crop)
        mask_crop= cropGRAY(mask, crop)

        height, width = img_crop.shape[0:2]
        w = 128
        while (w < width):
            h = 128
            while (h < height):
                cimg = img_crop[h-128:h, w-128:w,:]
                mimg = mask_crop[h-128:h, w-128:w]
                cv.imwrite(frame_outpath + str(name) +"_"+ str(w) +"_"+ str(h) + ".png", cimg)
                cv.imwrite(mask_outpath + str(name) +"_"+ str(w) +"_"+ str(h) + ".png", mimg)
                h = h + 128
            w = w + 128

if GET_MOSAIC:
    get_mosaic_set(FRAME_FOLDER, MASK_FOLDER, MOSAIC_FRAME_FOLDER, MOSAIC_MASK_FOLDER)


    


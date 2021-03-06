### testbed for postprocessing development 
from glob import glob
import sys,pickle
sys.path.append('../')
from utils.Calibrate import splitfn
from scipy.interpolate import splev, splprep, interp1d
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from utils.Models import FrameMeta

from keras.models import Input,load_model
from keras.layers import Dropout,concatenate,UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
from keras_segmentation.models.model_utils import get_segmentation_model
from keras_segmentation.predict import predict_multiple
from keras_segmentation.train import find_latest_checkpoint
from extremities import extremity

def postprocessing(path, folder, rnorms=[-.75,-.5,0,.5,0.75],
                   FIRST_FRAME=340,LAST_FRAME=650,WRITEVIDEO=1):
    # Options
    shock_ext = []
    shield_ext = []
    shield_ypos = []
    shock_points =[]
    shield_points=[]
    time = []

    ### Parse filepath
    pth, name, ext = splitfn(path)
    fname = name+ext
    print("### "+ name)

    ### Load video
    cap = cv.VideoCapture(path)
    ret, frame = cap.read();
    h,w,chan = np.shape(frame)
    step = chan * w
    nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)
    cap.set(cv.CAP_PROP_POS_FRAMES,FIRST_FRAME)
    counter=FIRST_FRAME

    flowDir = flowDirection(path, FIRST_FRAME)
    print("flow direction :")
    print(flowDir)
    ### Write output video
    if WRITEVIDEO:
        vid_cod = cv.VideoWriter_fourcc('M','J','P','G')
        output = cv.VideoWriter(folder+"edit_"+fname[0:-4]+'.m4v', vid_cod, 30.0,(w,h))

    ### Load CNN model
    model = cnn_set(frame)

    ### Loop through video frames
    counter = FIRST_FRAME
    while(counter<LAST_FRAME-1):
        for i in range(0,1):
            ret, frame = cap.read()
            counter += 1
        print(counter)
        if ret==False:
            print("No more frames")
            break

        ### Operations on the frame
        frame_ai = cnn_apply(frame, model)

        if WRITEVIDEO:
            output.write(frame_ai)

        ### Processing frame
        shieldParams,shockParams = extremity(frame_ai, flowDir,rnorms=rnorms)
        
        x,y = shieldParams[0],shieldParams[1]
        xShield, yShield = shieldParams[2], shieldParams[3]
        ShieldY, ShieldR = shieldParams[4],shieldParams[5]
        xs,ys=shockParams[0],shockParams[1]
        xShock, yShock=shockParams[2],shockParams[3]
        ShockY, ShockR = shockParams[4],shockParams[5]

        ### Save edges into arrays
        time.append(counter)
        shock_ext.append([xs,ys])
        shield_ext.append([x,y])

        shield_ypos.append(yShield)
        shock_points.append([xShock, yShock])
        shield_points.append([xShield, yShield])
        
    ### close video objects
    cap.release()
    if WRITEVIDEO:
        output.release()
        
    return shield_ext,shock_ext,shield_ypos,shield_points,shock_points,time

def loadVideo():
    #path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    dialog = QtWidgets.QFileDialog()
    mask = dialog.getOpenFileName(None, "Select Video")
    paths = mask
    #script = "cp -r " + str(folder_path) + " " + str(path)

def flowDirection(path, FIRST_FRAME):

    flow = []
    cap = cv.VideoCapture(path)
    nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)
    cap.set(cv.CAP_PROP_POS_FRAMES,FIRST_FRAME)
    counter = FIRST_FRAME
    for i in range(0,20):
        for i in range(0,10):
            ret, frame = cap.read()
            counter += 1
        if ret==False:
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (11,11), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)

        widthImg = frame.shape[1]
        widthLoc = maxLoc[1]

        fluxLoc = widthLoc/widthImg

        if fluxLoc > 0.5:
            flow.append("left")
        elif fluxLoc < 0.5:
            flow.append("right")

    flowDirection = max(set(flow), key = flow.count)

    return flowDirection

def cnn_set(img, n_classes = 3,
            ckpath = "shock_detection/checkpoints_mosaic/mynet_arcjetCV"):
    height = img.shape[0]
    width= img.shape[1]
    hpx = height- height%4
    wpx = width- width%4
    input_height,input_width = hpx, wpx
    
    ##############################################################################
    img_input = Input(shape=(input_height,input_width,n_classes ))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    out = Conv2D( n_classes, (1, 1) , padding='same')(conv5)
    ##############################################################################

    model = get_segmentation_model(img_input ,  out ) # build the segmentation model

    latest_weights = find_latest_checkpoint(ckpath)
    model.load_weights(latest_weights)

    return model

def cnn_apply(img, model):
    height = img.shape[0]
    width= img.shape[1]
    hpx = height- height%4
    wpx = width- width%4
    img = img[0:hpx,0:wpx,:]

    out = model.predict_segmentation(inp = img)
    return out

if __name__ == "__main__":

    folder = "video/"
    name = "IHF338Run003_WestView_3"
    name = "AHF335Run001_EastView_5"
    name = "IHF338Run002_WestView_1"
    ext = ".mp4"
    mask = folder+ name + ext
    LOAD = 1
    rn = [-.75,-.5,0,.5,0.75]
    if LOAD ==0:
        paths = glob(mask)
        vfolder, name, ext = splitfn(paths[0])
    foutname = os.path.join(folder,name+'.pickle')
    meta = FrameMeta(os.path.join(folder,name+'.meta'))

    if LOAD:
        fin = open(foutname, 'rb')
        out = pickle.load(fin)
        fin.close()
    else:
        FIRSTFRAME = meta.FIRST_GOOD_FRAME
        LASTFRAME = meta.LAST_GOOD_FRAME
        out = postprocessing(paths[0],folder,rnorms=rn,FIRST_FRAME=FIRSTFRAME,LAST_FRAME=LASTFRAME)
        fout = open(foutname, 'wb')
        pickle.dump(out,fout)
        fout.close()
        
    shield_ext,shock_ext,shield_ypos,shield_points,shock_points,time = out

    ### Plot XY edges
    fig0 = plt.figure(0)
    for i in range(0,len(shield_ext)):
        plt.plot(shield_ext[i][0],shield_ext[i][1],'g-')

    ### Plot XT
    fig1 = plt.figure(1)
    plt.title('Sample XT')
    sp = np.array(shield_points)
    sp = np.rollaxis(sp,2,0)
    for i in range(0,len(sp)):
        plt.plot(time,sp[i][:,0],'o-',label = "Radius = "+str(rn[i]))
    plt.legend(loc=0)
    
    ### Plot shock XY edges
    plt.figure(0)
    plt.title('XY Contours')
    for i in range(0,len(shock_ext)):
        plt.plot(shock_ext[i][0],shock_ext[i][1],'b-')

    ### Plot shock XT
    fig3 = plt.figure(3)
    plt.title('Shock XT')
    sp = np.array(shock_points)
    sp = np.rollaxis(sp,2,0)
    for i in range(0,len(sp)):
        plt.plot(time,sp[i][:,0],'o-',label = "Radius = "+str(rn[i]))
    plt.legend(loc=0)
    plt.show()
    

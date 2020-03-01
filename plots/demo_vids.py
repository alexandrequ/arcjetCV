import numpy as np
import cv2 as cv
from classes.Frame import getModelProps
from classes.Calibrate import splitfn
import matplotlib.pyplot as plt
from glob import glob


##fname = "AHF335Run001_EastView_1.mp4"
##fname = "IHF360-005_EastView_3_HighSpeed.mp4"
##fname = "IHF360-003_EastView_3_HighSpeed.mp4"

##folder = "video/AHF335/"
##mask = folder + "AHF335Run001_EastView_1.mp4"

##folder = "../video/IHF360/"
##mask = folder + "IHF360-005_EastView_3_HighSpeed.mp4"

folder = "../video/IHF338/"
mask = folder + "*004_WestView_3.mp4"  # default

##folder = "../video/HyMETS/"
##mask = folder + "PS12*.mp4"  # default

paths = glob(mask)

WRITE_VIDEO = True
WRITE_PICKLE = False
SHOW_CV = True
FIRST_FRAME = 310#+303
LAST_FRAME = 2706

MODELPERCENT = 0.005
STINGPERCENT = 0.5
CC = 'default'
fD = 'right'
iMin = None
iMax = None#255
hueMin = 60#95
hueMax = 170#140

myr = np.loadtxt('IHF338Run004_WestView_3_edges.csv',delimiter=',')
mcy = np.loadtxt('IHF338Run004_WestView_3_edges_cy.csv',delimiter=',')
myt = np.loadtxt('IHF338Run004_WestView_3_edges_time.csv',delimiter=',')
t0= 310

for path in paths:    
    pth, name, ext = splitfn(path)
    fname = name+ext;print("### "+ name)

    cap = cv.VideoCapture(path)
    ret, frame = cap.read(); h,w,chan = np.shape(frame)

    if WRITE_VIDEO:
        vid_cod = cv.VideoWriter_fourcc('m','p','4','v')
        output = cv.VideoWriter(folder+"edit_"+fname[0:-4]+'.avi', vid_cod, 30.0,(720+360,h))
        
    nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv.CAP_PROP_FPS)
    cap.set(cv.CAP_PROP_POS_FRAMES,FIRST_FRAME);
    counter=FIRST_FRAME
    myc,yt,centery=[],[],[]
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret==False:
            print("No more frames")
            break
        
        # Operations on the frame
        if SHOW_CV:
            draw = False
            plot=False
            verbose=False
        else:
            draw = True
            plot=True
            verbose=True
            
        ret = getModelProps(frame,counter,draw=draw,plot=plot,verbose=verbose,
                            modelpercent=MODELPERCENT,stingpercent=STINGPERCENT,
                            contourChoice=CC,flowDirection=fD,
                            intensityMin=iMin,intensityMax=iMax,
                            minHue=hueMin,maxHue=hueMax)

        if ret != None:
            (c,stingc), ROI, (th,cx,cy), flowRight,flags = ret
            (xb,yb,wb,hb) = cv.boundingRect(c)
            area = cv.contourArea(c)
            cv.rectangle(frame,(xb,yb,wb,hb),(255,255,255),3)
            cv.drawContours(frame, c, -1, (0,0,255), 2)

            ### Save contours and useful parameters
            myc.append([counter,cy,hb,area,c,flags])

        my_dpi = 100
        print(counter-310)
        if (myt == counter-310).any():
            i, = np.where(myt == counter-310)[0]
            plt.figure(figsize=(360/my_dpi, 720/my_dpi), dpi=my_dpi)
            plt.subplot(211)
            plt.plot((myt[0:i])/30,myr[0:i],'r^')
            plt.xlim([0,11])
            plt.ylabel('Recession (in)')
            
            plt.subplot(212)
            plt.xlim([0,11])
            mytime = range(0,int(myt[i])+1)
            yt.append((counter-310)/30); centery.append(720-cy)
            plt.plot(yt,centery,'b-')
            plt.ylabel('Vertical position (px)')
            plt.xlabel('Time (s)')
            plt.tight_layout()
            plt.savefig('my_fig_%i.png'%counter, dpi=my_dpi)
            plt.close()
            myplot  = cv.imread('my_fig_%i.png'%counter,1)
            hp,wp,chanp = np.shape(myplot)
            vis = np.concatenate((frame[:,0:720,:], myplot), axis=1)
        
            if WRITE_VIDEO:
                output.write(vis)

            if SHOW_CV:
                cv.imshow(name,vis)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            #cv.imshow(name,frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        counter +=1
      
    # When everything done, release the capture
    cap.release()
    if WRITE_VIDEO:
        output.release()
    cv.destroyAllWindows()

    if WRITE_PICKLE:
        import pickle
        fout = open(folder+fname[0:-4] +'_edges.pkl','wb')
        pickle.dump(myc,fout)
        fout.close()

    #input("Next video?")

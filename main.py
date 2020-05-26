"""
In this example, we demonstrate how to create simple camera viewer using Opencv3 and PyQt5
Author: Magnuis Haw, Alexandre Quintart
Last edited: 10 April 2020
"""

# import system module
import sys, os
import PyQt5
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

sys.path.append('../')
# import some PyQt5 modules
from gui.arcjetCV_gui import Ui_MainWindow
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage, QPixmap

from classes.Frame import getModelProps
from classes.Calibrate import splitfn

from glob import glob

from keras.models import Input,load_model
from keras.layers import Dropout,concatenate,UpSampling2D
from keras.layers import Conv2D, MaxPooling2D
from keras_segmentation.predict import predict_multiple
from keras_segmentation.train import find_latest_checkpoint

class MainWindow(QtWidgets.QMainWindow):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.stop = False
        logo = QPixmap("logo/arcjetCV_logo.png")
        logo = logo.scaledToHeight(451)
        self.ui.label_img.setPixmap(logo)
        self.show()


        self.folder = "video/"
        self.mask = self.folder+ "AHF335Run001_EastView_1.mp4"
        self.paths = glob(self.mask)

        self.ui.pushButton_runEdgesFullVideo.clicked.connect(self.run)
        self.ui.pushButton_stop.clicked.connect(self.stopRun)
        self.ui.actionLoad_video.triggered.connect(self.loadVideo)

    def run(self):

        # Options
        self.WRITE_VIDEO = self.ui.checkBox_writeVideo.isChecked()
        self.WRITE_PICKLE = self.ui.checkBox_writePickle.isChecked()
        self.SHOW_CV = True
        self.FIRST_FRAME = self.ui.spinBox_firstFrame.value() #900#+303
        self.MODELPERCENT = self.ui.spinBox_minArea.value() #0.012
        self.STINGPERCENT = self.ui.spinBox_minStingArea.value() #0.5
        self.CC = str(self.ui.comboBox_filterType.currentText()) #'default'
        self.FD = str(self.ui.comboBox_flowDirection.currentText())#'right'
        self.iMin = None #self.ui.minIntensity.value() #None#150
        self.iMax = None #self.ui.maxIntensity.value() #None#255
        self.hueMin = None #self.ui.minHue.value() #None#95
        self.hueMax = None #self.ui.maxHue.value() #None#140

        self.ui.pushButton_runEdgesFullVideo.hide()
        self.ui.pushButton_stop.show()

        LOAD = 0
        contour = None


        for idx, path in enumerate(self.paths):
            pth, name, ext = splitfn(path)
            fname = name+ext;print("### "+ name)

            cap = cv.VideoCapture(path)
            ret, self.frame = cap.read();
            h,w,chan = np.shape(self.frame)
            step = chan * w
            # AI set
            if (LOAD == 0):
                self.model = self.cnn_set( self.frame)
                LOAD = 1

            if self.WRITE_VIDEO:
                vid_cod = cv.VideoWriter_fourcc('m','p','4')
                output = cv.VideoWriter(self.folder+"edit_"+fname[0:-4]+'.mp4', vid_cod, 100.0,(w,h))

            nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv.CAP_PROP_FPS)
            cap.set(cv.CAP_PROP_POS_FRAMES,self.FIRST_FRAME);
            counter=self.FIRST_FRAME
            myc=[]
            while(True):

                if self.stop == True:
                    self.stop = False
                    return
                # Capture frame-by-frame
                ret, self.frame = cap.read()
                if ret==False:
                    print("No more frames")
                    break

                # Operations on the frame
                if self.SHOW_CV:
                    draw = False
                    plot=False
                    verbose=False
                else:
                    draw = True
                    plot = True
                    verbose = True

                if (idx%600 == 0):
                    frame_ai = self.cnn_apply(self.frame, self.model)
                    edges = cv.Canny(frame_ai,200,100)
                    contours, hierarchy = cv.findContours(edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                if contour is not None:
                    cv.drawContours(self.frame, contours, -1, (0,96,196), 3)

                # to uncomment
                #self.gradient()
                #ret = getModelProps(self.frame,counter,draw=draw,plot=plot,verbose=verbose,
                        #         modelpercent=self.MODELPERCENT,stingpercent=self.STINGPERCENT,
                        #         contourChoice=self.CC,flowDirection=self.FD,
                        #         intensityMin=self.iMin,intensityMax=self.iMax,
                        #         minHue=self.hueMin,maxHue=self.hueMax)

                #cv.drawContours(self.frame, self.contours, -1, (0,96,196), 3)

                #if ret != None:
                #    (c,stingc), ROI, (th,cx,cy), flowRight,flags = ret
                #    (xb,yb,wb,hb) = cv.boundingRect(c)
                #    area = cv.contourArea(c)
                #    cv.rectangle(self.frame,(xb,yb,wb,hb),(255,255,255),3)
                #    cv.drawContours(self.frame, c, -1, (0,255,255), 3)

                    ### Save contours and useful parameters
                    #myc.append([counter,cy,hb,area,c,flags])
                if self.WRITE_VIDEO:
                    output.write(self.frame)

                if self.SHOW_CV:

                    # create QImage from image
                    image = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
                    #image = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
                    #image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
                    qImg = QImage(image.data, w, h, step, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qImg)
                    self.pixmap_resize = pixmap.scaled(731, 451, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                    # show image in img_label
                    self.ui.label_img.setPixmap(self.pixmap_resize)
                    QApplication.processEvents() # update display

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

    def stopRun(self):
        self.stop = True
        self.ui.pushButton_stop.hide()
        self.ui.pushButton_runEdgesFullVideo.show()

    def loadVideo(self):
        #path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        dialog = QtWidgets.QFileDialog()
        self.mask = dialog.getOpenFileName(None, "Select Video")
        self.paths = self.mask
        #script = "cp -r " + str(self.folder_path) + " " + str(path)

    def bright(self):
        gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (11, 11), 0)
        # threshold the image to reveal light regions in the
        # blurred image
        self.thresh = cv.threshold(blurred, 250, 255, cv.THRESH_BINARY)[1]
        # perform a series of erosions and dilations to remove
        # any small blobs of noise from the thresholded image
        self.thresh = cv.erode(self.thresh, None, iterations=2)
        self.thresh = cv.dilate(self.thresh, None, iterations=4)
        return self.thresh

    def gradient(self):

        gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY)  # convert the image in gray
        # create a CLAHE object (Arguments are optional).
        #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #frame = clahe.apply(gray)
        blurred = cv.GaussianBlur(gray, (11, 11), 0) # smoothing (blurring) it to reduce high frequency noise
        self.thresh = cv.threshold(blurred, 250, 255, cv.THRESH_BINARY)[1] # threshold the image to reveal light regions in the
        # perform a series of erosions and dilations to remove
        # any small blobs of noise from the thresholded image
        self.thresh = cv.erode(self.thresh, None, iterations=2)
        self.thresh = cv.dilate(self.thresh, None, iterations=4)
        edges = cv.Canny(self.thresh,200,100)
        self.contours, hierarchy = cv.findContours(edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        #cv.drawContours(self.frame, self.contours, -1, (0,96,196), 3)
        #cv.imshow("hello", frame)

    def clahe(self):

        lab = cv.cvtColor(self.frame, cv.COLOR_BGR2LAB)
        lab_planes = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv.merge(lab_planes)
        self.frame = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
        cv.imshow("hello", self.frame)

    def flowDirection(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (11,11), 0)
        (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(gray)

        # display the results of the naive attempt
        cv.imshow("Naive", image)


        widthImg = image.shape[1]
        widthLoc = maxLoc[1]

        fluxLoc = widthLoc/widthImg

        if fluxLoc > 0.5:
        	flowDirection = "left"
        elif fluxLoc < 0.5:
        	flowDirection = "right"

        print(flowDirection)


    def cnn_set(self,img):

        cv.imwrite("frame.png", img)

        height = img.shape[0]
        width= img.shape[1]
        pix = max(width, height)- (max(width, height) % 4)
        input_height,input_width = pix, pix

        n_classes = 3
        epochs= 2
        ckpath = "shock_detection/checkpoints_mosaic/mynet_arcjetCV"

        ##############################################################################
        img_input = Input(shape=(input_height,input_width , 3 ))

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

        from keras_segmentation.models.model_utils import get_segmentation_model
        self.model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model


        latest_weights = find_latest_checkpoint(ckpath)
        self.model.load_weights(latest_weights)

        return self.model



    def cnn_apply(self, img, model):

        out = model.predict_segmentation(
            inp = "frame.png",
            out_fname = '"frame_out.png"', #out_dir+name+ext,
            colors=[(0,0,255),(0,255,0),(255,0,0)]
            )
        self.frame_ai = cv.imread("frame_out.png")
        return self.frame_ai


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())

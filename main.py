
"""
In this example, we demonstrate how to create simple camera viewer using Opencv3 and PyQt5
Author: Alexandre Quintart
Last edited: 10 April 2020
"""

# import system module
import sys
sys.path.append('../')
# import some PyQt5 modules
from gui.arcjetCV_gui import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
import numpy as np
import cv2 as cv
from classes.Frame import getModelProps
from classes.Calibrate import splitfn
import matplotlib.pyplot as plt
from glob import glob
folder = "video/"
mask = folder+ "AHF335Run001_EastView_1.mp4"

##folder = "video/IHF360/"
##mask = folder + "IHF360-005_EastView_3_HighSpeed.mp4"

##folder = "video/IHF338/"
##mask = folder + "*006_EastView_1.mp4"  # default

##folder = "video/HyMETS/"
##mask = folder + "PS12*.mp4"  # default

paths = glob(mask)

# Options
WRITE_VIDEO = False
WRITE_PICKLE = False
SHOW_CV = True
FIRST_FRAME = 900#+303
MODELPERCENT = 0.012
STINGPERCENT = 0.5
CC = 'default'
fD = 'right'
iMin = None#150
iMax = None#255
hueMin = None#95
hueMax = None#140



class MainWindow(QtWidgets.QMainWindow):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.show()

        # Options
        self.WRITE_VIDEO = self.ui.checkBox_writeVideo.isChecked()
        self.WRITE_PICKLE = self.ui.checkBox_writePickle.isChecked()
        self.SHOW_CV = True
        self.FIRST_FRAME = 900#self.ui.spinBox_firstFrame.value() #900#+303
        self.MODELPERCENT = 0.012#self.ui.spinBox_minArea.value() #0.012
        self.STINGPERCENT = 0.5#self.ui.spinBox_minStingArea.value() #0.5
        self.CC = str(self.ui.comboBox_filterType.currentText()) #'default'
        self.FD = str(self.ui.comboBox_flowDirection.currentText())#'right'
        self.iMin = None#self.ui.maxIntensity.value() #None#150
        self.iMax = None#self.ui.maxIntensity.value() #None#255
        self.hueMin = None#self.ui.minHue.value() #None#95
        self.hueMax = None#self.ui.maxHue.value() #None#140

        self.ui.pushButton_runEdgesFullVideo.clicked.connect(self.run)

    def run(self):

        for path in paths:
            pth, name, ext = splitfn(path)
            fname = name+ext;print("### "+ name)



            cap = cv.VideoCapture(path)
            ret, frame = cap.read();
            h,w,chan = np.shape(frame)
            step = chan * w

            if WRITE_VIDEO:
                vid_cod = cv.VideoWriter_fourcc('m','p','4','v')
                output = cv.VideoWriter(folder+"edit_"+fname[0:-4]+'.m4v', vid_cod, 100.0,(w,h))

            nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv.CAP_PROP_FPS)
            cap.set(cv.CAP_PROP_POS_FRAMES,FIRST_FRAME);
            counter=FIRST_FRAME
            myc=[]
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
                    cv.drawContours(frame, c, -1, (0,255,255), 3)

                    ### Save contours and useful parameters
                    myc.append([counter,cy,hb,area,c,flags])
                if WRITE_VIDEO:
                    output.write(frame)

                if SHOW_CV:
                    magnus =1

                    # create QImage from image
                    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    qImg = QImage(image.data, w, h, step, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qImg)
                    pixmap_resize = pixmap.scaled(731, 451, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                    # show image in img_label
                    self.ui.label_img.setPixmap(pixmap_resize)

                    cv.imshow(name,frame)
                    cv.destroyWindow(name)
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



if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())

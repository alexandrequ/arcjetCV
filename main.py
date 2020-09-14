"""
Primary GUI for arcjetCV project
Author: Magnus Haw, Alexandre Quintart
Last edited: 11 Sept 2020
"""
# import base libraries
import numpy as np
import cv2 as cv

# import some PyQt5 modules
from gui.arcjetCV_gui import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QThread, QTimer,pyqtSignal,pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage, QPixmap
# import analysis functions
from utils.Calibrate import splitfn
from models import ArcjetProcessor, Video, VideoMeta
from cnn import get_unet_model, cnn_apply

class MainWindow(QtWidgets.QMainWindow):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.stop = False
        
        # logo
        logo = QPixmap("gui/logo/arcjetCV_logo.png")
        logo = logo.scaledToHeight(451)
        self.frame = logo
        self.ui.label_img.setPixmap(logo)
        self.show()

        # folder/file properties
        self.folder = None
        self.path = None
        self.filename = None
        self.ext = None
        self.video = None
        self.videometa = None

        # Processor objects
        self.processor = None
        self.cnn = None

        # Connect interface
        self.ui.pushButton_runEdgesFullVideo.clicked.connect(self.run)
        self.ui.pushButton_stop.clicked.connect(self.stop_run)
        self.ui.actionLoad_video.triggered.connect(self.load_video)

    def show_img(self):
        ''' Shows img residing in self.frame '''
        # create QImage from image
        image = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
        qimg = QImage(image.data, self.video.w, self.video.h, self.video.w * self.video.chan, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap_resize = pixmap.scaled(731, 451, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # show image in img_label
        self.ui.label_img.setPixmap(pixmap_resize)
        # update display
        QApplication.processEvents()

    def load_video(self):
        ''' Loads a single video file using dialog '''

        # create fileDialog to select file
        dialog = QtWidgets.QFileDialog()
        pathmask = dialog.getOpenFileName(None, "Select Video")
        self.path = pathmask[0]
        self.folder, self.filename, self.ext = splitfn(self.path)

        # Create video object
        self.video = Video(self.path)
        self.videometa = VideoMeta(self.folder+'/'+self.filename+'.meta')
        self.videometa.write()
        
        # Setup first frame on display
        if self.videometa.FIRST_GOOD_FRAME is None:
            self.frame = self.video.last_frame.copy()
            c_range = None
        else:
            self.frame = self.video.get_frame(self.videometa.FIRST_GOOD_FRAME)
            c_range = self.videometa.crop_range()

        # init processor object
        self.processor = ArcjetProcessor(self.frame, crop_range=c_range, flow_direction=self.videometa.FLOW_DIRECTION)
        self.show_img()

    def run(self):
        # Error check video filepath
         # one or more paths
         # each path is valid

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

        # Setup output video

        # Process frame
        frame = self.video.get_next_frame()
        contour_dict,argdict = self.processor.process(frame, {'SEGMENT_METHOD':'HSV'})

        # Draw contours 
        for key in contour_dict.keys():
            if key is 'MODEL':
                cv.drawContours(frame, contour_dict[key], -1, (0,255,0), 2)
            elif key is 'SHOCK':
                cv.drawContours(frame, contour_dict[key], -1, (0,0,255), 2)
        self.frame = frame.copy()
        self.show_img()
        print(contour_dict)
        # Display processed frames
        # Write output data
        # close output video

    def stop_run(self):
        self.stop = True
        self.ui.pushButton_stop.hide()
        self.ui.pushButton_runEdgesFullVideo.show()

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


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())

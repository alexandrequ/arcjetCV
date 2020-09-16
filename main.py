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
from PyQt5.QtGui import QImage, QPixmap, QColor

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
        self.frame = cv.imread("gui/logo/arcjetCV_logo.png")
        image = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
        self.hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
        self.h,self.w,self.chan = np.shape(image)
        qimg = QImage(image.data, self.w, self.h, self.w*3, QImage.Format_RGB888)
        logo = QPixmap.fromImage(qimg)
        logo = logo.scaledToHeight(451)
        self.SCALE_FACTOR = 1004/452
        self.ui.label_img.setPixmap(logo)
        
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
        self.ui.pushButton_process.clicked.connect(self.process_all)
        self.ui.actionLoad_video.triggered.connect(self.load_video)
        self.ui.label_img.newCursorValue.connect(self.getPixel)

        self.show()

    @pyqtSlot(list)
    def getPixel(self, inputvals):
        x,y = inputvals
        x = min(self.w-1,int(x*self.SCALE_FACTOR))
        y = min(self.h-1,int(y*self.SCALE_FACTOR))
        #print(x, y, self.w, self.h)

        try:
            h,s,v = self.hsv[y,x,:]
            b,g,r = self.frame[y,x,:]
            self.ui.basebar.setText("XY (%i, %i), HSV (%i, %i, %i), RGB (%i, %i, %i)"%(x,y,h,s,v,r,g,b))
        except:
            self.ui.basebar.setText("XY (%i, %i)"%(x,y))

    def show_img(self):
        ''' Shows img residing in self.frame '''
        # create QImage from image
        #cv.rectangle(self.frame,(xb,yb,wb,hb),(255, 255, 255),3)
        image = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
        self.hsv = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)
        qimg = QImage(image.data, self.video.w, self.video.h, self.video.w * self.video.chan, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(731, 451, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # show image in img_label
        self.ui.label_img.setPixmap(pixmap)
        # update display
        QApplication.processEvents()

    def update_frame_index(self):
        frame = self.video.get_frame(self.ui.spinBox_FrameIndex.value())
        inputdict = {'SEGMENT_METHOD':str(self.ui.comboBox_filterType.currentText())}
        inputdict["HSV_MODEL_RANGE"] = [(self.ui.minHue.value(), self.ui.minSaturation.value(), self.ui.minIntensity.value()), 
                                        (self.ui.maxHue.value(), self.ui.maxSaturation.value(), self.ui.maxIntensity.value())]
        inputdict["HSV_SHOCK_RANGE"] = [(self.ui.minHue_2.value(), self.ui.minSaturation_2.value(), self.ui.minIntensity_2.value()), 
                                        (self.ui.maxHue_2.value(), self.ui.maxSaturation_2.value(), self.ui.maxIntensity_2.value())]
        inputdict["THRESHOLD"] = self.ui.minIntensity.value()

        contour_dict,argdict = self.processor.process(frame, inputdict)

        # Draw contours 
        for key in contour_dict.keys():
            if key is 'MODEL':
                cv.drawContours(frame, contour_dict[key], -1, (0,255,0), 2)
            elif key is 'SHOCK':
                cv.drawContours(frame, contour_dict[key], -1, (0,0,255), 2)
        self.frame = frame.copy()
        self.show_img()

    def load_video(self):
        ''' Loads a single video file using dialog '''

        # create fileDialog to select file
        dialog = QtWidgets.QFileDialog()
        pathmask = dialog.getOpenFileName(None, "Select Video")

        self.path = pathmask[0]
        if self.path != '':
            self.folder, self.filename, self.ext = splitfn(self.path)

            # Create video object
            self.video = Video(self.path)
            self.videometa = VideoMeta(self.folder+'/'+self.filename+'.meta')
            self.videometa.write()

            if self.video.w / self.video.h > 731/451:
                self.SCALE_FACTOR = self.video.w / 730
                self.w = self.video.w
                self.h = self.video.h
            else:
                self.SCALE_FACTOR = self.video.h / 450
                self.w = self.video.w
                self.h = self.video.h
            
            # Setup first frame on display
            if self.videometa.FIRST_GOOD_FRAME is None:
                self.frame = self.video.last_frame.copy()
                c_range = None
            else:
                self.frame = self.video.get_frame(self.videometa.FIRST_GOOD_FRAME)
                c_range = self.videometa.crop_range()

            # Init processor object
            self.processor = ArcjetProcessor(self.frame, crop_range=c_range, flow_direction=self.videometa.FLOW_DIRECTION)
            
            # Initialize UI
            self.ui.spinBox_FrameIndex.setRange(0,self.video.nframes-1)
            self.ui.spinBox_FrameIndex.setValue(self.videometa.FIRST_GOOD_FRAME)
            self.ui.spinBox_FirstGoodFrame.setValue(self.videometa.FIRST_GOOD_FRAME)
            self.ui.spinBox_LastGoodFrame.setValue(self.videometa.LAST_GOOD_FRAME)

            # Connect UI
            self.ui.spinBox_FrameIndex.valueChanged.connect(self.update_frame_index)
            self.ui.maxHue.valueChanged.connect(self.update_frame_index)
            self.ui.minHue.valueChanged.connect(self.update_frame_index)
            self.ui.minIntensity.valueChanged.connect(self.update_frame_index)
            self.ui.maxIntensity.valueChanged.connect(self.update_frame_index)
            self.ui.minSaturation.valueChanged.connect(self.update_frame_index)
            self.ui.maxSaturation.valueChanged.connect(self.update_frame_index)

            self.ui.maxHue_2.valueChanged.connect(self.update_frame_index)
            self.ui.minHue_2.valueChanged.connect(self.update_frame_index)
            self.ui.minIntensity_2.valueChanged.connect(self.update_frame_index)
            self.ui.maxIntensity_2.valueChanged.connect(self.update_frame_index)
            self.ui.minSaturation_2.valueChanged.connect(self.update_frame_index)
            self.ui.maxSaturation_2.valueChanged.connect(self.update_frame_index)

            self.ui.comboBox_filterType.currentTextChanged.connect(self.update_frame_index)
            self.update_frame_index()

    def process_all(self):
        # Error check video filepath
         # one or more paths
         # each path is valid


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
        # Display processed frames
        # Write output data
        # close output video

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())

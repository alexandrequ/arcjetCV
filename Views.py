import numpy as np
import cv2 as cv

from PyQt5.QtCore import Qt, QThread, QTimer,pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton
from PyQt5.QtWidgets import QVBoxLayout, QApplication, QSpinBox, QSlider
from pyqtgraph import ImageView

from classes.Models import Video
from classes.Frame import getModelProps

class StartWindow(QMainWindow):
    def __init__(self, video = None):
        super().__init__()
        self.video = video

        self.central_widget = QWidget()
        self.button_frame = QPushButton('View Frame', self.central_widget)
        self.frame_index = QSpinBox(self.central_widget)
        self.frame_index.setRange(0,video.nframes-1)
        self.button_movie = QPushButton('Start Movie', self.central_widget)
        self.image_view = ImageView()
##        self.slider = QSlider(Qt.Horizontal)
##        self.slider.setRange(0,10)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.button_frame)        
        self.layout.addWidget(self.frame_index)
        self.layout.addWidget(self.button_movie)
        self.layout.addWidget(self.image_view)
##        self.layout.addWidget(self.slider)
        self.setCentralWidget(self.central_widget)

        self.frame_index.valueChanged.connect(self.update_image)
        self.button_frame.clicked.connect(self.update_image)
        self.button_movie.clicked.connect(self.start_movie)

##        self.update_timer = QTimer()
##        self.update_timer.timeout.connect(self.update_movie)        

    def update_image(self):
        frame = self.video.get_frame(self.frame_index.value())

        ret = getModelProps(frame,self.frame_index.value(),
                            contourChoice='default',flowDirection='left')

        if ret != None:
            (c,stingc), ROI, (th,cx,cy), flowRight,flags = ret
            (xb,yb,wb,hb) = cv.boundingRect(c)
            area = cv.contourArea(c)
            cv.rectangle(frame,(xb,yb,wb,hb),(255,255,255),3)
            cv.drawContours(frame, c, -1, (0,255,255), 3)
            
        nframe = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.image_view.setImage(cv.transpose(nframe,nframe))

    def update_movie(self):
        frame = cv.cvtColor(self.video.last_frame,cv.COLOR_BGR2RGB)
        nframe = cv.transpose(frame,frame)
        self.image_view.setImage(nframe)

    def start_movie(self):
        ### Define thread and signals
        self.video_thread = VideoThread(self.video,self.frame_index.value(),self.video.nframes-1)
        self.video_thread.finished.connect(self.enableInputs)
        self.video_thread.updateImage.connect(self.update_movie)

        ### Start thread
        self.video_thread.start()
##        self.update_timer.start(60)

        ### Disable inputs during movie
        self.disableInputs()

    def disableInputs(self):
        self.button_frame.setEnabled(False)
        self.button_movie.setEnabled(False)
        self.frame_index.setEnabled(False)

    def enableInputs(self):
        self.button_frame.setEnabled(True)
        self.frame_index.setEnabled(True)
        self.button_movie.setEnabled(True)

class VideoThread(QThread):
    updateImage = pyqtSignal()
    
    def __init__(self, video, start_ind,stop_ind):
        super().__init__()
        self.video = video
        self.start_ind = start_ind
        self.stop_ind = stop_ind

    def run(self):
        frame = self.video.get_frame(self.start_ind)
        for i in range(self.start_ind,self.stop_ind):
            
            frame = self.video.get_frame(i)
            ret = getModelProps(frame,i,contourChoice='default',flowDirection='left')

            if ret != None:
                (c,stingc), ROI, (th,cx,cy), flowRight,flags = ret
                (xb,yb,wb,hb) = cv.boundingRect(c)
                area = cv.contourArea(c)
                cv.rectangle(frame,(xb,yb,wb,hb),(255,255,255),3)
                cv.drawContours(frame, c, -1, (0,255,255), 3)
                self.updateImage.emit()
                
if __name__ == "__main__":
    import sys
    sys._excepthook = sys.excepthook 
    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback) 
        sys.exit(1) 
    sys.excepthook = exception_hook
    
    path = "/home/magnus/Desktop/arcjetCV/video/"
    fname = "AHF335Run001_EastView_1.mp4"
    fname = "IHF360-005_EastView_3_HighSpeed.mp4"
    video = Video(path+fname)
    app = QApplication([])
    window = StartWindow(video)
    window.show()
    app.exit(app.exec_())
    video.close_video()

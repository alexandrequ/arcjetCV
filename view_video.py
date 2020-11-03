import numpy as np
import cv2 as cv

from PyQt5.QtCore import Qt, QThread, QTimer,pyqtSignal,pyqtSlot
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton,QLabel
from PyQt5.QtWidgets import QVBoxLayout, QApplication, QSpinBox, QSlider

from models import Video, ArcjetProcessor

class VideoThread(QThread):
    updateImage = pyqtSignal(QImage)
    stopSignal = pyqtSignal(int)
    
    def __init__(self, video, start_ind,stop_ind, processor=None, argdict={"SEGMENT_METHOD":"AutoHSV"}):
        super().__init__()
        self.video = video
        self.start_ind = start_ind
        self.stop_ind = stop_ind
        self.stopFlag = False
        self.index = start_ind
        self.argdict = argdict

        if processor is None:
            frame = self.video.get_frame(start_ind)
            processor = ArcjetProcessor(frame,argdict)
        self.processor = processor

    def run(self):
        for i in range(self.start_ind,self.stop_ind):
            frame = self.video.get_frame(i)
            self.index = i
            edges,argdict = self.processor.process(frame, self.argdict)
            h,w,chan = np.shape(frame)
            step = chan * w
            if edges is not None:
                for key in edges.keys():
                    cv.drawContours(frame, edges[key], -1, (0, 255, 0), 3)
                    #(xb,yb,wb,hb) = cv.boundingRect(edges[key])
                    #area = cv.contourArea(edges[key])
                    #cv.rectangle(frame,(xb,yb,wb,hb),(255, 255, 255),3)
                
            nframe = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            qImg = QImage(nframe.data, w, h, step, QImage.Format_RGB888)
            self.updateImage.emit(qImg)
            if self.stopFlag:
                break
    def stop(self):
        self.stopFlag = True
        self.stopSignal.emit(self.index)

class MyLabel(QLabel):
    def __init__(self):
        super(MyLabel,self).__init__()

    def mousePressEvent(self,event):
        print(event.x(), event.y())

    def mouseReleaseEvent(self,event):
        print(event.x(), event.y())

class StartWindow(QMainWindow):
    def __init__(self, video = None):
        super().__init__()
        self.video = video
        frame = self.video.get_frame(0)
        self.processor = ArcjetProcessor(frame,flow_direction='left')
        self.argdict = {"SEGMENT_METHOD":"GRAY","THRESHOLD":90}

        self.central_widget = QWidget()
        self.button_stop = QPushButton('Stop Movie', self.central_widget)
        self.frame_index = QSpinBox(self.central_widget)
        self.frame_index.setRange(0,video.nframes-1)
        self.button_movie = QPushButton('Start Movie', self.central_widget)
        self.image_view = MyLabel()
        
##        self.slider = QSlider(Qt.Horizontal)
##        self.slider.setRange(0,10)

        self.layout = QVBoxLayout(self.central_widget)
               
        self.layout.addWidget(self.frame_index)
        self.layout.addWidget(self.button_movie)
        self.layout.addWidget(self.button_stop) 
        self.layout.addWidget(self.image_view)
##        self.layout.addWidget(self.slider)
        self.setCentralWidget(self.central_widget)

        self.frame_index.valueChanged.connect(self.update_image)
        self.button_movie.clicked.connect(self.start_movie)
        self.update_image()
        

    @pyqtSlot(QImage)
    def change_pixmap(self, image):
        pixmap = QPixmap.fromImage(image)
        #self.pixmap_resize = pixmap.scaled(731, 451, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        #self.image_view.setPixmap(self.pixmap_resize)

        self.image_view.setPixmap(pixmap)
        #print("changed pixmap")

    def update_image(self):
        frame = self.video.get_frame(self.frame_index.value())
        edges,argdict = self.processor.process(frame, self.argdict)
        h,w,chan = np.shape(frame)
        step = chan * w
        if edges is not None:
            for key in edges.keys():
                cv.drawContours(frame, edges[key], -1, (0, 255, 0), 3)
            
        nframe = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        qImg = QImage(nframe.data, w, h, step, QImage.Format_RGB888)
        self.change_pixmap(qImg)

    def start_movie(self):
        ### Define thread and signals
        self.video_thread = VideoThread(self.video,self.frame_index.value(),self.video.nframes-1)
        self.video_thread.updateImage.connect(self.change_pixmap)
        self.video_thread.stopSignal.connect(self.enableInputs)
        #self.video_thread.stopSignal.connect(self.frame_index.setValue)
        self.button_stop.clicked.connect(self.video_thread.stop)
        ### Start thread
        self.video_thread.start()
##        self.update_timer.start(60)

        ### Disable inputs during movie
        self.disableInputs()

    def disableInputs(self):
        self.button_stop.setEnabled(True)
        self.button_movie.setEnabled(False)
        self.frame_index.setEnabled(False)

    def enableInputs(self,i):
        self.button_stop.setEnabled(False)
        self.frame_index.setEnabled(True)
        self.button_movie.setEnabled(True)

                
if __name__ == "__main__":
    import sys
    sys._excepthook = sys.excepthook 
    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback) 
        sys.exit(1) 
    sys.excepthook = exception_hook
    
    path = "/home/magnus/Desktop/NASA/arcjetCV/data/video/"
    #path = "/u/wk/mhaw/arcjetCV/video/"
    fname = "AHF335Run001_EastView_5"
    #fname = "IHF360-003_EastView_3_HighSpeed"
    # fname = "AHF335Run001_EastView_1"
    fname = "IHF338Run006_EastView_1"
    fname = "HyMETS-PS12_90"
    ext = ".mp4"
    video = Video(path+fname+ext)
    print(video)
    app = QApplication([])
    window = StartWindow(video)
    window.show()
    app.exit(app.exec_())
    video.close()

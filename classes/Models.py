import cv2 as cv
import numpy as np
from classes.Functions import splitfn      

class Video(object):
    def __init__(self, path):
        ### path variables
        self.fpath = path
        folder, name, ext = splitfn(path)
        self.name = name
        self.folder = folder
        self.ext = ext

        ### opencv video file object
        self.cap = cv.VideoCapture(self.fpath)
        self.nframes = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        ret, frame = self.cap.read()
        self.shape = np.shape(frame)
        self.last_frame = frame

        ### video output
        self.writer = None

    def __str__(self):
        return 'Video: {}, shape={}, nframes={}'.format(self.fpath,self.shape,self.nframes,)

    def get_next_frame(self):
        ret, self.last_frame = self.cap.read()
        return self.last_frame

    def get_frame(self,index):
        self.cap.set(cv.CAP_PROP_POS_FRAMES,index);
        ret, self.last_frame = self.cap.read()
        return self.last_frame

    def count_frames(self):
        self.nframes = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        return self.nframes

    def close_video(self):
        self.cap.release()

    def get_writer(self):
        vid_cod = cv.VideoWriter_fourcc('m','p','4','v')
        self.writer = cv.VideoWriter(self.folder+"edit_"+self.name+'.m4v',
                                         vid_cod, fps,(self.shape[1],self.shape[0]))
        
    def close_writer(self):
        self.writer.release()


if __name__ == '__main__':
    path = "/home/magnus/Desktop/arcjetCV/video/"
    fname = "AHF335Run001_EastView_1.mp4"
    fname = "IHF360-005_EastView_3_HighSpeed.mp4"
    video = Video(path+fname)
    print(video)
##    frame = video.get_next_frame()
##    print(frame)
    video.close_video()

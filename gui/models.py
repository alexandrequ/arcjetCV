import cv2 as cv
import numpy as np

class Camera:
    def __init__(self, cam_num):
        self.cam_num = cam_num
        self.cap = None
        self.last_frame = np.zeros((1,1))
        
    def initialize(self):
        self.cap = cv.VideoCapture(self.cam_num)

    def __str__(self):
        return 'OpenCV Camera {}'.format(self.cam_num)

    def get_next_frame(self):
        ret, self.last_frame = self.cap.read()
        return self.last_frame

    def get_frame(self,index):
        self.cap.set(cv.CAP_PROP_POS_FRAMES,index);
        ret, self.last_frame = self.cap.read()
        return self.last_frame
    
    def set_brightness(self, value):
        self.cap.set(cv.CAP_PROP_BRIGHTNESS, value)

    def get_brightness(self):
        return self.cap.get(cv.CAP_PROP_BRIGHTNESS)

    def close_camera(self):
        self.cap.release()

    def acquire_movie(self, num_frames):
        movie = []
        for _ in range(num_frames):
            movie.append(self.get_next_frame())
        return movie

class Video(object):
    def __init__(self, path):
        self.fpath = path
        self.cap = None
        self.last_frame = np.zeros((1,1))
        
    def initialize(self):
        self.cap = cv.VideoCapture(self.fpath)

    def __str__(self):
        return 'OpenCV Camera {}'.format(self.cam_num)

    def get_next_frame(self):
        ret, self.last_frame = self.cap.read()
        return self.last_frame

    def get_frame(self,index):
        self.cap.set(cv.CAP_PROP_POS_FRAMES,index);
        ret, self.last_frame = self.cap.read()
        return self.last_frame
    
    def set_brightness(self, value):
        self.cap.set(cv.CAP_PROP_BRIGHTNESS, value)

    def get_brightness(self):
        return self.cap.get(cv.CAP_PROP_BRIGHTNESS)

    def close_camera(self):
        self.cap.release()

    def acquire_movie(self, num_frames):
        movie = []
        for _ in range(num_frames):
            movie.append(self.get_next_frame())
        return movie


if __name__ == '__main__':
    cam = Camera(0)
    cam.initialize()
    print(cam)
    frame = cam.get_frame()
    print(frame)
    cam.close_camera()

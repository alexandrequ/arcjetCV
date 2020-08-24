import os
import cv2 as cv
import numpy as np
from classes.Functions import splitfn,contoursHSV,contoursGRAY,getROI      

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

class FrameMeta(object):

    def __init__(self,path):
        ### File parameters
        folder, name, ext = splitfn(path)
        self.folder = folder
        self.name = name
        self.ext = ext
        self.path = path

        ### Meta Parameters for each video
        self.VIDEO = None
        self.WIDTH =None
        self.HEIGHT =None
        self.CHANNELS = None
        self.NFRAMES = None
        self.CALFRAME_INDEX = None
        self.FIRST_GOOD_FRAME = None
        self.LAST_GOOD_FRAME = None
        self.USE_GRAY = None
        self.USE_HSVSQ = None
        self.MODELPERCENT = None
        self.MODEL_WIDTH = None
        self.MODEL_HEIGHT = None
        self.MODEL_CENTER = None
        self.FLOW_DIRECTION = None
        self.iMin = None#150
        self.iMax = None#255
        self.hueMin = None#95
        self.hueMax = None#140
        self.NOTES = None

        if os.path.exists(path):
            self.load()

    def __str__(self):
        outstr = "#Property, Value\n"
        for prop, value in vars(self).items():
            if value is not None:
                outstr += prop +", "+ str(value) + '\n'
            else:
                outstr += prop +", ?\n"
        return outstr
        
    def write(self):
        fout = open(self.path,'w')
        fout.write(str(self))
        fout.close()
        
    def load(self):
        fin = open(self.path,'r')
        lines = fin.readlines()
        inttype = ['WIDTH','HEIGHT','CHANNELS','NFRAMES','CALFRAME_INDEX',
                   'FIRST_GOOD_FRAME','LAST_GOOD_FRAME','MODEL_WIDTH',
                   'MODEL_HEIGHT','iMin','iMax','hueMin','hueMax']
        floattype = ['MODELPERCENT']
        pointtype = ['MODEL_CENTER']
        booltype  = ['USE_GRAY','USE_HSVSQ']
        for i in range(1,len(lines)):                
            attrs = lines[i].split(',')
            if attrs[0] in inttype:
                setattr(self,attrs[0],int(attrs[1].strip()) )
            elif attrs[0] in floattype:
                setattr(self,attrs[0],float(attrs[1]) )
            elif attrs[0] in booltype:
                setattr(self,attrs[0],attrs[1].strip()=="True")
            elif attrs[0] in pointtype:
                attrs[2] = attrs[2].strip()
                setattr(self,attrs[0],','.join(attrs[1:]) )
            else:
                setattr(self,attrs[0],str(attrs[1].strip()) )
                
    def processFrame(self,frame,hueMin=118,hueMax=140,iMin=150,iMax=255,GRAY=False,HSVSQ=True,fD='right' ):        
        self.HEIGHT,self.WIDTH,self.CHANNELS = frame.shape

        if GRAY:
             c,stingc = contoursGRAY(frame,iMin,log=None,draw=False,plot=True)
             ROI = getROI(frame,c,c,draw=True,plot=True)

        if HSVSQ:
             c,stingc = contoursHSV(frame,minHue=hueMin,maxHue=hueMax,flags=None,
                                    modelpercent=.001,intensityMin=iMin,draw=False,plot=True)
             ROI = getROI(frame,c,stingc,draw=True,plot=True)
             
        self.MODELPERCENT = ROI[3]*ROI[2]/(self.WIDTH*self.HEIGHT)
        self.MODEL_WIDTH = ROI[2]
        self.MODEL_HEIGHT = ROI[3]
        self.MODEL_CENTER = (int(ROI[1]+ROI[3]/2), int(ROI[0]+ROI[2]/2) )
        self.hueMin,self.hueMax = hueMin,hueMax
        self.iMin,self.iMax = iMin,iMax
        self.USE_GRAY = GRAY
        self.USE_HSVSQ= HSVSQ
        self.FLOW_DIRECTION = fD
        print(self)
        print('\n')

        if input("write (y/n)?")=='y':
            
            self.write()

    def processVideoFrame(self,vpath,index,hueMin=118,hueMax=140,iMin=150,iMax=255,GRAY=False,HSVSQ=True,fD='right' ):        
        video = Video(vpath)
        self.VIDEO = vpath
        self.NFRAMES = video.count_frames()
        self.CALFRAME_INDEX = index
        frame = video.get_frame(self.CALFRAME_INDEX)
        self.processFrame(frame,hueMin=hueMin,hueMax=hueMax,iMin=iMin,iMax=iMax,GRAY=GRAY,HSVSQ=HSVSQ,fD=fD )


if __name__ == '__main__':
    path = "/home/magnus/Desktop/arcjetCV/video/"
    fname = "AHF335Run001_EastView_1.mp4"
    fname = "IHF360-005_EastView_3_HighSpeed.mp4"
    video = Video(path+fname)
    print(video)
##    frame = video.get_next_frame()
##    print(frame)
    video.close_video()

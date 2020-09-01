import os
import cv2 as cv
import numpy as np
from utils.Functions import splitfn,contoursHSV,contoursGRAY,getROI      

class Processor(object):
    ''' Template class for frame processing classes

    Attributes:
        WIDTH (int): openCV image width
        HEIGHT (int): openCV image height
        SHAPE (tuple): openCV image shape
        CHANNELS (int): openCV number of channels (1 or 3)
        CROP (list): [[ymin,ymax],[xmin,xmax]]
        FLOW_DIRECTION (dict): direction of flow
    '''

    def __init__(self, frame, crop_range=None, flow_direction=None):
        '''initialize object
        
        :param frame: opencv image (RGB or grayscale)
        :param crop_range (list): [[ymin,ymax],[xmin,xmax]]
        :param flow_direction (dict): direction of flow
        '''
        self.SHAPE = frame.shape
        self.HEIGHT = self.SHAPE[0]
        self.WIDTH = self.SHAPE[1]
        if len(frame.shape) == 3:
            self.CHANNELS = self.SHAPE[2]
        else:
            self.CHANNELS = 1
        self.FLOW_DIRECTION = flow_direction
        if crop_range is None:
            self.CROP = [[0,self.HEIGHT], [0,self.WIDTH]]
        else:
            self.CROP = crop_range

        return

    def get_flow_direction(self, frame):
        '''infer flow direction
        
        :param frame: opencv image
        :returns flowDirection: string, "left or "right"
        '''

        # Check img type
        if self.CHANNELS == 3:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        elif self.CHANNELS == 1:
            gray = frame
        else:
            raise IndexError("ERROR: number of channels of input frame (%i) is not 1 or 3!"%self.CHANNELS)

        # smooth image to remove speckles/text
        gray = cv.GaussianBlur(gray, (15,15), 0)

        # find location of max intensity
        (min_val, max_val, min_loc, max_loc) = cv.minMaxLoc(gray)
        width_img, width_loc = frame.shape[1], max_loc[1]
        flux_loc = width_loc/width_img

        # Bright location generally indicates flow direction
        if flux_loc > 0.5:
            flow_direction = "left"
        elif flux_loc < 0.5:
            flow_direction = "right"

        return flow_direction

    def get_image_flags(self, frame, argdict):
        """
        Uses histogram of 8bit grayscale image (0,255) to classify image type

        :param frame: opencv image
        :returns: dictionary of flags
        """
        try:
            verbose = argdict['verbose']
            stingpercent = argdict['stingpercent']
            modelpercent = argdict['modelpercent']
        except KeyError:
            verbose = False; stingpercent=.05; modelpercent=.005

        ### HSV brightness value histogram
        hsv_ = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hsv_=cv.GaussianBlur(hsv_, (5, 5), 0)
        gray = hsv_[:,:,2]

        ### grayscale histogram
        histr = cv.calcHist( [gray], None, None, [256], (0, 256))
        imgsize = gray.size

        ### classification criteria
        modelvis = (histr[12:250]/imgsize > 0.00).sum() != 1
        modelvis *= histr[50:250].sum()/imgsize > modelpercent
        stingvis= histr[50:100].sum()/imgsize > stingpercent
        overexp = histr[243:].sum()/imgsize > modelpercent
        underexp= histr[150:].sum()/imgsize < modelpercent

        ### Determine saturation limit
        slimit,peaki=253,240
        if overexp:
            peaki = histr[240:].argmax() +240
            if histr[peaki:].sum()/imgsize > modelpercent:
                slimit = peaki-1
        saturated = histr[slimit:].sum()/imgsize > modelpercent

        ### Extract intensity threshold
        try:
            total = histr[30:].sum()+1
            exp_val = (histr[30:].ravel()*np.arange(30,256)).sum()/total
            avg = int(max(exp_val,55))
            peaki = histr[avg:].argmax() +avg
            if abs(peaki-avg) < 10:
                thresh=peaki-15
            else:
                thresh = max(histr[avg:peaki].argmin() +avg, peaki-15)
        except:
            thresh = 150
        
        if verbose:
            print('peaki, slimit, thresh', peaki,slimit,thresh)
            print("Model visible",modelvis)
            print("Sting visible",stingvis)
            print("overexposed", overexp)
            print("underexposed", underexp)
            print("saturated",saturated)
        return argdict

    def preprocess(self, frame, argdict):
        '''acquire flags, apply crop, get flow direction'''
        # Get flow direction
        if self.FLOW_DIRECTION is None:
            self.FLOW_DIRECTION = self.get_flow_direction(frame)

        # Crop frame to ROI
        try:
            if self.CHANNELS == 1:
                img_crop = frame[self.CROP[0][0]:self.CROP[0][1], self.CROP[1][0]:self.CROP[1][1]]
            else:
                img_crop = frame[self.CROP[0][0]:self.CROP[0][1], self.CROP[1][0]:self.CROP[1][1], :]
        except IndexError:
            raise IndexError("ERROR: processor crop window %s incompatible with given frame shape %s"%(str(self.CROP),str(frame.shape)))

        argdict = self.get_image_flags(img_crop, argdict)
        return img_crop, argdict

    def segment(self, img_crop, argdict):
        '''segment image, acquire contours'''
        # process img_crop here
        rawdatadict = {} #{"model":modelcontour,"shock":shockcontour}
        return rawdatadict, argdict #rawdatadict

    def reduce(self,rawdatadict,argdict):
        '''get leading edge, interpolate to std shape'''
        # process rawdatadict here
        datadict ={} #{"model":modeledge,"shock":shockedge}
        return datadict, argdict
    
    def process(self, frame, argdict):
        '''fully process image'''
        img_crop, argdict = self.preprocess(frame,argdict)
        rawdatadict, argdict = self.segment(img_crop,argdict)
        datadict, argdict = self.reduce(rawdatadict,argdict)
        return datadict, argdict

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
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
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
                                         vid_cod, self.fps,(self.shape[1],self.shape[0]))
        
    def close_writer(self):
        self.writer.release()

class VideoMeta(object):

    inttype = ['WIDTH','HEIGHT','CHANNELS','NFRAMES',
                'FIRST_GOOD_FRAME','LAST_GOOD_FRAME',
                'YMIN','YMAX','XMIN','XMAX','FRAME_NUMBER']
    floattype = ['MODELPERCENT', 'INTENSITY_THRESHOLD']
    booltype  = ['SHOCK_VISIBLE','MODEL_VISIBLE', 'OVEREXPOSED', 'EDGE_SATURATED']
    pointtype = []

    def __init__(self,path):
        ### File parameters
        folder, name, ext = splitfn(path)
        self.folder = folder
        self.name = name
        self.ext = ext
        self.path = path

        ### Meta Parameters for each video
        self.WIDTH =None
        self.HEIGHT =None
        self.CHANNELS = None
        self.NFRAMES = None
        self.FIRST_GOOD_FRAME = None
        self.LAST_GOOD_FRAME = None
        self.MODELPERCENT = None
        self.FLOW_DIRECTION = None
        self.YMIN = None
        self.YMAX = None
        self.XMIN = None
        self.XMAX = None
        self.NOTES = None

        if os.path.exists(path):
            self.load()

    def __str__(self):
        outstr = "#Property, Value\n"
        for prop, value in vars(self).items():
            if value is not None:
                outstr += prop +", "+str(value) + '\n'
            else:
                outstr += prop +", ?\n"
        return outstr
        
    def write(self):
        fout = open(self.path,'w')
        fout.write(str(self))
        fout.close()
        
    def load(self,path=None):
        if path is None:
            fin = open(self.path,'r')
        else:
            fin = open(path,'r')
        print(self.path)
        lines = fin.readlines()
        
        for i in range(1,len(lines)):                
            attrs = lines[i].split(',')
            if attrs[0] in VideoMeta.inttype:
                setattr(self,attrs[0],int(attrs[1].strip()) )
            elif attrs[0] in VideoMeta.floattype:
                setattr(self,attrs[0],float(attrs[1]) )
            elif attrs[0] in VideoMeta.booltype:
                setattr(self,attrs[0],attrs[1].strip()=="True")
            elif attrs[0] in VideoMeta.pointtype:
                attrs[2] = attrs[2].strip()
                setattr(self,attrs[0],','.join(attrs[1:]) )
            else:
                setattr(self,attrs[0],str(attrs[1].strip()) )

class FrameMeta(VideoMeta):

    def __init__(self,path,fnumber=None,videometa=None):
        super(FrameMeta,self).__init__(path)
        
        if not os.path.exists(path):
            self.load(path=videometa.path)
            self.FRAME_INDEX = fnumber
            
            ### File parameters
            folder, name, ext = splitfn(path)
            self.folder = folder
            self.name = name
            self.ext = ext
            self.path = path

if __name__ == '__main__':
    path = "/home/magnus/Desktop/NASA/arcjetCV/data/video/"
    fname = "AHF335Run001_EastView_5"
    #fname = "IHF360-003_EastView_3_HighSpeed"
    fname = "IHF338Run006_EastView_1"

    vm = VideoMeta(path+fname+".meta")
    video = Video(path+fname+".mp4")
    print(video)
    frame = video.get_frame(vm.FIRST_GOOD_FRAME)
    cv.imwrite("test.png",frame)
    fm = FrameMeta("test.meta",vm.FIRST_GOOD_FRAME,vm)
    #fm.write()
    print(fm)
    video.close_video()
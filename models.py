import os
import abc
import cv2 as cv
import numpy as np
from utils.Functions import splitfn,contoursHSV,contoursGRAY
from utils.Functions import getEdgeFromContour,contoursAutoHSV

class ImageProcessor(object):
    ''' Abstract base class for image processing

    Attributes:
        WIDTH (int): openCV image width
        HEIGHT (int): openCV image height
        SHAPE (tuple): openCV image shape
        CHANNELS (int): openCV number of channels (1 or 3)
        CROP (list): [[ymin,ymax],[xmin,xmax]]
        values (dict): for countours, values
        flags (dict): for errors, flags
    '''
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, frame, crop_range=None):
        '''initialize object
        
        :param frame: opencv image (RGB or grayscale)
        :param crop_range (list): [[ymin,ymax],[xmin,xmax]]
        '''
        self.SHAPE = frame.shape
        self.FRAME = frame
        self.HEIGHT = self.SHAPE[0]
        self.WIDTH = self.SHAPE[1]
        self.values = {}
        self.flags = {}
        if len(frame.shape) == 3:
            self.CHANNELS = self.SHAPE[2]
        else:
            self.CHANNELS = 1
        if crop_range is None:
            self.CROP = [[0,self.HEIGHT], [0,self.WIDTH]]
        else:
            self.CROP = crop_range
        # Crop frame to ROI
        try:
            if self.CHANNELS == 1:
                self.FRAME_CROP = frame[self.CROP[0][0]:self.CROP[0][1], self.CROP[1][0]:self.CROP[1][1]]
            else:
                self.FRAME_CROP = frame[self.CROP[0][0]:self.CROP[0][1], self.CROP[1][0]:self.CROP[1][1], :]
        except IndexError:
            self.FRAME_CROP = None
            raise IndexError("ERROR: processor crop window %s incompatible with given frame shape %s"%(str(self.CROP),str(frame.shape)))

        return

    @abc.abstractmethod
    def preprocess(self, frame, argdict):
        ''' preprocess img, crop, get flags '''
        return
    
    @abc.abstractmethod
    def segment(self, frame, argdict):
        ''' segment image, acquire contours '''
        return

    @abc.abstractmethod
    def reduce(self, input, argdict):
        ''' get metrics, reduce to minimum set '''
        return

    @abc.abstractmethod
    def process(self, frame, argdict):
        '''fully process image'''
        return

class ArcjetProcessor(ImageProcessor):
    ''' Abstract base class for image processing

    Attributes:
        WIDTH (int): openCV image width
        HEIGHT (int): openCV image height
        SHAPE (tuple): openCV image shape
        CHANNELS (int): openCV number of channels (1 or 3)
        CROP (list): [[ymin,ymax],[xmin,xmax]]
        values (dict): for countours, values
        flags (dict): for errors, flags
        FLOW_DIRECTION (str): 'left' or 'right'
        FRAME: openCV image
        FRAME_CROP: cropped image
    '''
    def __init__(self, frame, crop_range=None, flow_direction=None):
        super(ArcjetProcessor,self).__init__(frame, crop_range=crop_range)
        if flow_direction is None:
            self.FLOW_DIRECTION = self.get_flow_direction(frame)
        else:
            self.FLOW_DIRECTION = flow_direction

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
            verbose = argdict['VERBOSE']
            modelpercent = argdict['MODEL_FRACTION']
        except KeyError:
            verbose = False; modelpercent=0.005

        ### HSV brightness value histogram
        gray_ = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray  = cv.GaussianBlur(gray_, (5, 5), 0)

        ### grayscale histogram
        histr = cv.calcHist( [gray], None, None, [256], (0, 256))
        imgsize = gray.size

        ### classification criteria
        modelvis = (histr[12:250]/imgsize > 0.00).sum() != 1
        modelvis *= histr[50:250].sum()/imgsize > modelpercent
        argdict['MODEL_VISIBLE'] = modelvis
        argdict['OVEREXPOSED'] =  histr[243:].sum()/imgsize > modelpercent
        argdict['UNDEREXPOSED'] =  histr[150:].sum()/imgsize < modelpercent

        return argdict

    def preprocess(self, frame, argdict):
        '''acquire flags, apply crop, get flow direction'''
        # Crop frame to ROI
        try:
            if self.CHANNELS == 1:
                self.FRAME_CROP = frame[self.CROP[0][0]:self.CROP[0][1], self.CROP[1][0]:self.CROP[1][1]]
            else:
                self.FRAME_CROP = frame[self.CROP[0][0]:self.CROP[0][1], self.CROP[1][0]:self.CROP[1][1], :]
        except IndexError:
            self.FRAME_CROP = None
            raise IndexError("ERROR: processor crop window %s incompatible with given frame shape %s"%(str(self.CROP),str(frame.shape)))

        # Get flow direction
        if self.FLOW_DIRECTION is None:
            self.FLOW_DIRECTION = self.get_flow_direction(frame)
        
        argdict = self.get_image_flags(self.FRAME_CROP, argdict)
        return self.FRAME_CROP, argdict

    def segment(self, img_crop, argdict):
        ''' segment image using one of several methods'''
        if argdict["SEGMENT_METHOD"] == 'AutoHSV':
            #use contoursAutoHSV
            contour_dict, flags = contoursAutoHSV(img_crop, flags=argdict)

        elif argdict["SEGMENT_METHOD"] == 'HSV':
            #use contoursHSV
            contour_dict, flags = contoursHSV(img_crop,log=None,
                                        minHSVModel=(95,0,150),maxHSVModel=(121,125,255),
                                        minHSVShock=(125,78,115),maxHSVShock=(145,190,230))

        elif argdict["SEGMENT_METHOD"] == 'GRAY':
            #use contoursGRAY
            try:
                thresh = argdict["THRESHOLD"]
            except:
                thresh = 140
            contour_dict, flags = contoursGRAY(img_crop,thresh=thresh,log=None)

        elif argdict["SEGMENT_METHOD"] == 'CNN':
            #use machine learning CNN
            pass
        
        argdict.update(flags)
        return contour_dict, argdict

    def reduce(self, contour_dict, argdict):
        ''' get edges and metrics '''
        edges = {}
        for key in contour_dict.keys():
            c = contour_dict[key]
            if c is not None:
                edges[key] = getEdgeFromContour(c,self.FLOW_DIRECTION)
            else:
                edges[key] = None
        return edges, argdict

    def process(self, frame, argdict):
        ''' fully process image '''
        frame_crop, argdict = self.preprocess(frame, argdict)
        contour_dict, argdict = self.segment(frame_crop, argdict)
        edges, argdict = self.reduce(contour_dict, argdict)
        return edges, argdict

class Video(object):
    ''' Convenience wrapper for opencv video capture 

    Methods:
        get_frame: gets arbitrary frame
        get_next_frame: gets next frame
        get_writer: get a video writer object
        close: closes video and writer object
        close_writer: closes writer object
    '''
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
        self.cap.set(cv.CAP_PROP_POS_FRAMES,index)
        ret, self.last_frame = self.cap.read()
        return self.last_frame

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self.cap.release()

    def get_writer(self):
        vid_cod = cv.VideoWriter_fourcc('m','p','4','v')
        self.writer = cv.VideoWriter(self.folder+"edit_"+self.name+'.m4v',
                                         vid_cod, self.fps,(self.shape[1],self.shape[0]))
        
    def close_writer(self):
        self.writer.release()

class VideoMeta(object):
    ''' Class designed to save/load video metadata in readable txt format
            creates *.meta files with useful information
    '''

    inttype = ['WIDTH','HEIGHT','CHANNELS','NFRAMES',
                'FIRST_GOOD_FRAME','LAST_GOOD_FRAME',
                'YMIN','YMAX','XMIN','XMAX','FRAME_NUMBER']
    floattype = ['MODELPERCENT', 'INTENSITY_THRESHOLD']
    booltype  = ['SHOCK_VISIBLE','MODEL_VISIBLE', 'OVEREXPOSED', 'UNDEREXPOSED','SATURATED']
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

    def crop_range(self):
        return [[self.YMIN,self.YMAX],[self.XMIN, self.XMAX]]

class FrameMeta(VideoMeta):
    ''' Stores frame metadata in text files.

    '''
    def __init__(self,path,fnumber=None,videometa=None):
        super(FrameMeta,self).__init__(path)
        
        if not os.path.exists(path) and videometa is not None:
            ### load video metadata
            self.load(path=videometa.path)
            self.FRAME_INDEX = fnumber
            
            ### restore frame file parameters
            folder, name, ext = splitfn(path)
            self.folder = folder
            self.name = name
            self.ext = ext
            self.path = path

class Logger(object):
    def __init__(self,filename,PRINT=True,FILEIO=False,prefix=''):
        self.filename=filename
        self.prefix = prefix
        self.print = PRINT
        self.fileio = FILEIO
        
    def write(self,line):
        if self.print:
            print(self.prefix+line.__str__())
        if self.fileio:
            fh = open(self.filename,'a')
            fh.write(self.prefix+line.__str__()+'\n')
            fh.close()

if __name__ == '__main__':
    path = "/home/magnus/Desktop/NASA/arcjetCV/data/video/"
    fname = "AHF335Run001_EastView_5"
    #fname = "IHF360-003_EastView_3_HighSpeed"
    fname = "IHF338Run006_EastView_1"
    fname = "HyMETS-PS03_90"

    vm = VideoMeta(path+fname+".meta")
    video = Video(path+fname+".mp4")
    print(video)
    frame = video.get_frame(vm.FIRST_GOOD_FRAME+1000)

    # Process frame
    p = ArcjetProcessor(frame,crop_range=vm.crop_range(),flow_direction = vm.FLOW_DIRECTION)
    contour_dict,argdict = p.process(frame, {'SEGMENT_METHOD':'AutoHSV'})
    print(argdict,contour_dict)

    # Plot edges
    c = contour_dict['MODEL']
    import matplotlib.pyplot as plt
    plt.plot(c[:,0,0],c[:,0,1],'g-')
    plt.show()

    # cv.imwrite("test.png",frame)
    # fm = FrameMeta("test.meta",vm.FIRST_GOOD_FRAME,vm)
    # fm.write()
    # print(fm)
    video.close()

import os
import abc
import cv2 as cv
import numpy as np
import pickle
from utils.Functions import splitfn,contoursHSV,contoursGRAY,contoursCNN
from utils.Functions import getEdgeFromContour,contoursAutoHSV, getPoints
from cnn import get_unet_model, cnn_apply

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

        self.cnn = get_unet_model(self.FRAME_CROP)

    def set_crop(self,crop_range):
        ''' sets crop window range [[ymin,ymax], [xmin,xmax]] '''
        self.CROP = crop_range
        try:
            if self.CHANNELS == 1:
                self.FRAME_CROP = frame[self.CROP[0][0]:self.CROP[0][1], self.CROP[1][0]:self.CROP[1][1]]
            else:
                self.FRAME_CROP = frame[self.CROP[0][0]:self.CROP[0][1], self.CROP[1][0]:self.CROP[1][1], :]
        except IndexError:
            self.FRAME_CROP = None
            raise IndexError("ERROR: processor crop window %s incompatible with given frame shape %s"%(str(self.CROP),str(frame.shape)))

        # Reinitialize CNN
        self.cnn = get_unet_model(self.FRAME_CROP)
        
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
        argdict['PIXEL_MIN'] = gray.min()
        argdict['PIXEL_MAX'] = gray.max()

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
        '''acquire flags, get flow direction'''
        # Crop image
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
        
        # Get exposure classification
        argdict = self.get_image_flags(self.FRAME_CROP, argdict)

        return self.FRAME_CROP, argdict

    def segment(self, img_crop, argdict):
        ''' segment image using one of several methods'''

        if argdict["SEGMENT_METHOD"] == 'AutoHSV':
            #use contoursAutoHSV
            contour_dict, flags = contoursAutoHSV(img_crop, flags=argdict)
            argdict.update(flags)

        elif argdict["SEGMENT_METHOD"] == 'HSV':
            #use contoursHSV
            try:
                HSVModelRange = argdict["HSV_MODEL_RANGE"]
                HSVShockRange = argdict["HSV_SHOCK_RANGE"]
            except KeyError:
                HSVModelRange = [(0,0,150), (121,125,255)]
                HSVShockRange = [(125,40,85), (170,80,230)]
                
            contour_dict, flags = contoursHSV(img_crop,log=None,
                                        minHSVModel=HSVModelRange[0],maxHSVModel=HSVModelRange[1],
                                        minHSVShock=HSVShockRange[0],maxHSVShock=HSVShockRange[1])
            argdict.update(flags)

        elif argdict["SEGMENT_METHOD"] == 'GRAY':
            #use contoursGRAY
            try:
                thresh = argdict["THRESHOLD"]
            except:
                thresh = 240
            contour_dict, flags = contoursGRAY(img_crop,thresh=thresh,log=None)
            argdict.update(flags)

        elif argdict["SEGMENT_METHOD"] == 'CNN':
            #use machine learning CNN
            contour_dict, flags = contoursCNN(img_crop, self.cnn)
            argdict.update(flags)
        
        return contour_dict, argdict

    def reduce(self, contour_dict, argdict):
        ''' get edges and metrics '''
        edges = {}
        for key in contour_dict.keys():
            c = contour_dict[key]

            if c is not None:
                ### get contour area
                M = cv.moments(c)
                argdict[key+"_AREA"] = M["m00"]

                ### get centroid
                if M["m00"] > 0:
                    argdict[key+"_CENTROID_X"] = int(M["m10"] / M["m00"])
                    argdict[key+"_CENTROID_Y"] = int(M["m01"] / M["m00"])
                else:
                    argdict[key+"_CENTROID_X"] = np.nan
                    argdict[key+"_CENTROID_Y"] = np.nan

                ### get front edge
                edges[key] = getEdgeFromContour(c,self.FLOW_DIRECTION, offset =(self.CROP[0][0],self.CROP[1][0]) )

                if key == "MODEL":
                    outputs = getPoints(edges[key], flow_direction= self.FLOW_DIRECTION, r=[-.75,-.25,0,.25,.75], prefix='MODEL')
                    argdict.update(outputs)
                elif key == "SHOCK":
                    outputs = getPoints(edges[key], flow_direction= self.FLOW_DIRECTION, r=[0],prefix="SHOCK")
                    argdict.update(outputs)
            else:
                edges[key] = None
        return edges, argdict

    def process(self, frame, argdict):
        ''' fully process image '''
        try: 
            frame_crop, argdict = self.preprocess(frame, argdict)
            contour_dict, argdict = self.segment(frame_crop, argdict)
            edges, argdict = self.reduce(contour_dict, argdict)
        except:
            edges = {"MODEL":None,"SHOCK":None}
        return edges, argdict.copy()

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
        self.h,self.w,self.chan = self.shape
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
        print(self.folder+"edit_"+self.name+'.m4v')
        self.writer = cv.VideoWriter(os.path.join(self.folder,"edit_"+self.name+'.m4v'),
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
            self.load(path)

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
        
    def load(self,path):
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
        
        ### Ensure path vars are correct
        folder, name, ext = splitfn(path)
        self.folder = folder
        self.name = name
        self.ext = ext
        self.path = path

    def crop_range(self):
        return [[self.YMIN,self.YMAX],[self.XMIN, self.XMAX]]

class FrameMeta(VideoMeta):
    ''' Stores frame metadata in text files.

    '''
    def __init__(self,path,fnumber=None,videometa=None):
        super(FrameMeta,self).__init__(path)
        
        if not os.path.exists(path) and videometa is not None:
            ### load video metadata
            self.load(videometa.path)
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

class OutputList(list):
    """Extension of list with write to file function
        expected to hold dictionary objects corresponding to 
        analysis of individual video frames

        filepath argument must contain low and high index constraints
        which are delimited by underscores:
        e.g. myoutput_0_10.opl

    Args:
        list (list): base list class
        filepath (string): path for saving file
    """
    def __init__(self,filepath):
        super(OutputList,self).__init__()
        self.filepath=filepath
        folder, name, ext = splitfn(filepath)
        self.folder = folder

        namesplit = name.split('_')
        self.prefix = namesplit[0:-2]
        self.low_index = int(namesplit[-2])
        self.high_index = int(namesplit[-1])
    
    def write(self):
        fout = open(self.filepath,'wb')
        pickle.dump(self, fout)
        fout.close()

    def append(self,obj):
        if obj["INDEX"] <= self.high_index and obj["INDEX"] >= self.low_index:
            super(OutputList,self).append(obj)

if __name__ == '__main__':
    TEST_FUNCTIONS = True
    BASE_TEST = False

    if TEST_FUNCTIONS:
        import matplotlib.pyplot as plt
        from utils.Functions import cleanEdge, getEdgeDifference
        files = ["/home/magnus/Desktop/NASA/arcjetCV/data/video/HyMETS-PS03_90_1250_1260.out",
                 "/home/magnus/Desktop/NASA/arcjetCV/data/video/HyMETS-PS03_90_2250_6250.out",
                 "/home/magnus/Desktop/NASA/arcjetCV/data/video/HyMETS-PS03_90_7000_7123.out"]
        raw_outputs =[]
        for fname in files:
            with open(fname,'rb') as file:
                opl = pickle.load(file)
                raw_outputs.extend(opl)
        
        e1 = raw_outputs[0]["MODEL"]
        em = raw_outputs[11]["MODEL"]
        e2 = raw_outputs[-1]["MODEL"]
        dl = 0.0558
        y,diff,v1,v2 = getEdgeDifference(e2*dl,e1*dl,ninterp=1000)
        ym,diffm,v1m,v2m = getEdgeDifference(e2*dl,em*dl,ninterp=1000)
        fig = plt.figure()
        plt.subplot(212)
        plt.plot(y,diff,label="Differential recession")
        
        plt.xlabel("Y (mm)")
        plt.ylabel("Recession (mm)")
        plt.subplot(211)

        plt.plot(y,v2,"--",label="Initial"); 

        plt.plot(y,v2m,label="Mid-point")
        plt.plot(y,v1,label="Final")
        plt.legend(loc=0)
        plt.xlabel("Y (mm)")
        plt.ylabel("X (mm)")
        plt.show()

    if BASE_TEST:
        path = "/home/magnus/Desktop/NASA/arcjetCV/data/video/"
        fname = "AHF335Run001_EastView_1"
        #fname = "IHF360-003_EastView_3_HighSpeed"
        #fname = "IHF338Run006_EastView_1"
        #fname = "HyMETS-PS03_90"

        vm = VideoMeta(path+fname+".meta")
        video = Video(path+fname+".mp4")
        print(video)
        frame = video.get_frame(vm.FIRST_GOOD_FRAME)

        # Create OutputList object to store results
        opl = OutputList("/home/magnus/Desktop/NASA/arcjetCV/test_3070_4070.out")

        # Process frame
        p = ArcjetProcessor(frame,crop_range=vm.crop_range(),flow_direction = vm.FLOW_DIRECTION)
        contour_dict,argdict = p.process(frame, {'SEGMENT_METHOD':'CNN',"INDEX":vm.FIRST_GOOD_FRAME})

        argdict.update(contour_dict)
        opl.append(argdict)
        opl.write()
        print(argdict.keys(),contour_dict.keys())

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

    
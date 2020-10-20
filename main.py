"""
Primary GUI for arcjetCV project
Author: Magnus Haw, Alexandre Quintart
Last edited: 11 Sept 2020
"""
# import base libraries
import os
import numpy as np
import pandas as pd
import cv2 as cv
import pickle

# import some PyQt5 modules
from gui.arcjetCV_gui import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QThread, QTimer,pyqtSignal,pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage, QPixmap, QColor

# import analysis functions
from utils.Calibrate import splitfn
from utils.Functions import getPoints, getOutlierMask
from models import ArcjetProcessor, Video, VideoMeta, OutputList
from cnn import get_unet_model, cnn_apply

import matplotlib.pyplot as plt 

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

        # Data structures
        self.raw_outputs = []
        self.time_series = None
        self.PLOTKEYS =[]
        self.fit_dict = None

        # Connect interface
        self.ui.pushButton_process.clicked.connect(self.process_all)
        self.ui.actionLoad_video.triggered.connect(self.load_video)
        self.ui.pushButton_loadVideo.clicked.connect(self.load_video)
        self.ui.pushButton_export_csv.clicked.connect(self.export_to_csv)
        self.ui.pushButton_fitData.clicked.connect(self.fit_data)

        self.ui.label_img.newCursorValue.connect(self.getPixel)
        self.ui.pushButton_LoadFiles.clicked.connect(self.load_outputs)
        self.ui.pushButton_PlotData.clicked.connect(self.plot_outputs)

        self.show()

    @pyqtSlot(list)
    def getPixel(self, inputvals):
        xi,yi, w,h = inputvals
        #print(inputvals, self.SCALE_FACTOR, self.w, self.h)
        yi -= int((h - self.h/self.SCALE_FACTOR)/2)
        yi = max(0,yi)
        x = min(self.w-1,int(xi*self.SCALE_FACTOR))
        y = min(self.h-1,int(yi*self.SCALE_FACTOR))

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
        inputdict["FLOW_DIRECTION"] = self.ui.comboBox_flowDirection.currentText()
        self.processor.FLOW_DIRECTION = self.ui.comboBox_flowDirection.currentText()

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
            self.videometa = VideoMeta(os.path.join(self.folder,self.filename+'.meta'))
            print(self.videometa.path)
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
            self.ui.lineEdit_filename.setText(self.video.name)

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
            self.ui.comboBox_flowDirection.currentTextChanged.connect(self.update_frame_index)
            self.update_frame_index()

    def process_all(self):
        # Create OutputList object to store results
        ilow,ihigh = self.ui.spinBox_FirstGoodFrame.value(), self.ui.spinBox_LastGoodFrame.value()
        prefix = self.ui.lineEdit_filename.text()
        filename = "%s_%i_%i.out"%(prefix,ilow,ihigh)
        opl = OutputList(os.path.join(self.video.folder, filename))
        
        inputdict = {'SEGMENT_METHOD':str(self.ui.comboBox_filterType.currentText())}
        inputdict["HSV_MODEL_RANGE"] = [(self.ui.minHue.value(), self.ui.minSaturation.value(), self.ui.minIntensity.value()), 
                                        (self.ui.maxHue.value(), self.ui.maxSaturation.value(), self.ui.maxIntensity.value())]
        inputdict["HSV_SHOCK_RANGE"] = [(self.ui.minHue_2.value(), self.ui.minSaturation_2.value(), self.ui.minIntensity_2.value()), 
                                        (self.ui.maxHue_2.value(), self.ui.maxSaturation_2.value(), self.ui.maxIntensity_2.value())]
        inputdict["THRESHOLD"] = self.ui.minIntensity.value()
        
        # Setup output video
        if self.ui.checkBox_writeVideo.isChecked():
            self.video.get_writer()

        # Process frame
        for frame_index in range(ilow,ihigh+1):
            frame = self.video.get_frame(frame_index)
            inputdict["INDEX"] = frame_index
            contour_dict,argdict = self.processor.process(frame, inputdict)

            # Draw contours 
            for key in contour_dict.keys():
                if key is 'MODEL':
                    cv.drawContours(frame, contour_dict[key], -1, (0,255,0), 2)
                elif key is 'SHOCK':
                    cv.drawContours(frame, contour_dict[key], -1, (0,0,255), 2)
            self.frame = frame.copy()
            self.show_img()

            argdict.update(contour_dict)
            opl.append(argdict.copy())

            # Add processed frame to video output
            if self.ui.checkBox_writeVideo.isChecked():
                self.video.writer.write(self.frame)

        # Write output data
        opl.write()
        self.raw_outputs = opl

        # close output video
        if self.ui.checkBox_writeVideo.isChecked():
            self.video.close_writer()

    def load_outputs(self):
        # create fileDialog to select file
        options = QtWidgets.QFileDialog.Options()
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"Load ouput files", "","Output Files (*.out);;All Files (*)", options=options)
        
        self.ui.basebar.setText("Loading %i files"%len(files))

        # Load all files & concatenate 
        self.raw_outputs =[]
        for fname in files:
            with open(fname,'rb') as file:
                opl = pickle.load(file)
                self.raw_outputs.extend(opl)

        if len(files) > 0:
            fpath, name, ext = splitfn(files[0])
            # Show summary of loaded data
            summary = "Loaded %i files\n"%len(files)
            summary += "Folder: %s\n"%fpath
            for fname in files:
                fpath, name, ext = splitfn(fname)
                summary += "File: %s\n"%name
            summary += "Total frames: %i\n"%len(self.raw_outputs)

            self.ui.label_data_summary.setText(summary)
            self.ui.basebar.setText("Finished loading files")
        
    def plot_outputs(self):
        self.ui.basebar.setText("Plotting data...")
        
        # Reset plotting windows
        self.ui.Window1.canvas.figure.clf()
        self.ui.Window2.canvas.figure.clf()

        ax1 = self.ui.Window1.canvas.figure.subplots()
        ax2 = self.ui.Window2.canvas.figure.subplots()

        # Plotting params
        n = len(self.raw_outputs)
        index,m75,m25,mc,p25,p75,radius = [],[],[],[],[],[],[]
        time, sarea,marea,sc,sm,ypos = [],[],[],[],[],[]

        diameter = self.ui.doubleSpinBox_diameter.value()
        units = self.ui.comboBox_units.currentText()
        fps = self.ui.doubleSpinBox_fps.value()
        maskn = self.ui.spinBox_mask_frames.value()

        if len(self.raw_outputs) > 10:
        
            for i in range(0,len(self.raw_outputs),maskn):
                # Save frame index, time
                index.append(self.raw_outputs[i]["INDEX"])
                time.append(self.raw_outputs[i]["INDEX"]/fps)

                # Model positions (-75%, -25%, center, 25%, 75% radius)
                if self.raw_outputs[i]['MODEL'] is not None:
                    xpos = self.raw_outputs[i]['MODEL_INTERP_XPOS']
                    center = self.raw_outputs[i]['MODEL_YCENTER']

                    m75.append(xpos[0])
                    m25.append(xpos[1])
                    mc.append(xpos[2])
                    p25.append(xpos[3])
                    p75.append(xpos[4])
                    ypos.append(center)
                    radius.append(self.raw_outputs[i]['MODEL_RADIUS'])
                else:
                    m75.append(np.nan)
                    m25.append(np.nan)
                    mc.append(np.nan)
                    p25.append(np.nan)
                    p75.append(np.nan)
                    ypos.append(np.nan)
                    radius.append(np.nan)
                
                # Shock center x-position
                if self.raw_outputs[i]['SHOCK'] is not None:
                    sc.append(self.raw_outputs[i]['SHOCK_INTERP_XPOS'][0])
                else:
                    sc.append(np.nan)

                # Shock and model area
                if self.raw_outputs[i]['SHOCK'] is not None:
                    sarea.append(self.raw_outputs[i]['SHOCK_AREA'])
                else:
                    sarea.append(np.nan)
                
                if self.raw_outputs[i]['MODEL'] is not None:
                    marea.append(self.raw_outputs[i]['MODEL_AREA'])
                else:
                    marea.append(np.nan)

                # Shock-model separation, center
                if (self.raw_outputs[i]['MODEL'] is not None) and (self.raw_outputs[i]['SHOCK'] is not None):
                    sm.append( abs(sc[-1]-mc[-1]) )
                else:
                    sm.append(np.nan)                
                
                ### Plot XY contours
                if self.raw_outputs[i]['MODEL'] is not None:
                    ax1.plot(np.array(self.raw_outputs[i]['MODEL'][:,0,0]),
                                            np.array(self.raw_outputs[i]['MODEL'][:,0,1]),'g-',label="model_%i"%index[-1])
                if self.raw_outputs[i]['SHOCK'] is not None:
                    ax1.plot(np.array(self.raw_outputs[i]['SHOCK'][:,0,0]),
                                            np.array(self.raw_outputs[i]['SHOCK'][:,0,1]),'r--',label="shock_%i"%index[-1])
                ax1.set_xlabel("X (px)")
                ax1.set_ylabel("Y (px)")
                ax1.figure.tight_layout()
                ax1.figure.canvas.draw()

            ### Mask outliers
            metrics = [marea,ypos,radius,mc,sc]
            mask = np.zeros(len(time))
            for metric in metrics:
                mask += getOutlierMask(metric)

            ### Infer px length
            radius_masked = np.ma.masked_where(mask>0,radius)
            pixel_length = (diameter/(2*radius_masked.max()))

            ### Plot XT series
            ym75 = np.ma.masked_where(mask > 0, m75)*pixel_length
            ym25 = np.ma.masked_where(mask > 0, m25)*pixel_length
            ymc  = np.ma.masked_where(mask > 0, mc)*pixel_length
            yp25 = np.ma.masked_where(mask > 0, p25)*pixel_length
            yp75 = np.ma.masked_where(mask > 0, p75)*pixel_length
            ysarea = np.ma.masked_where(mask > 0, sarea)
            ymarea = np.ma.masked_where(mask > 0, marea)
            ysc = np.ma.masked_where(mask > 0, sc)*pixel_length
            ysm = np.ma.masked_where(mask > 0, sm)*pixel_length
            yypos = np.ma.masked_where(mask > 0, ypos)*pixel_length
            
            self.PLOTKEYS =[] 

            if self.ui.checkBox_m75_radius.isChecked():
                ax2.plot(time, ym75, 'mo',label="Model -75%R")
                self.PLOTKEYS.append("MODEL_-0.75R "+units)

            if self.ui.checkBox_m25_radius.isChecked():
                ax2.plot(time, ym25, 'bo',label="Model -25%R")
                self.PLOTKEYS.append("MODEL_-0.25R "+units)

            if self.ui.checkBox_model_center.isChecked():
                ax2.plot(time, ymc, 'go',label="Model center")
                self.PLOTKEYS.append("MODEL_CENTER "+units)

            if self.ui.checkBox_25_radius.isChecked():
                ax2.plot(time, yp25, 'co',label="Model +25%R")
                self.PLOTKEYS.append("MODEL_0.25R "+units)

            if self.ui.checkBox_75_radius.isChecked():
                ax2.plot(time, yp75, 'ro',label="Model +75%R")
                self.PLOTKEYS.append("MODEL_0.75R "+units)

            if self.ui.checkBox_shock_area.isChecked():
                ax2.plot(time, ysarea, 'y^',label="Shock area (px)")
                self.PLOTKEYS.append("SHOCK_AREA [px]")

            if self.ui.checkBox_model_area.isChecked():
                ax2.plot(time, ymarea, 'yx',label="Model area (px)")
                self.PLOTKEYS.append("MODEL_AREA [px]")

            if self.ui.checkBox_shock_center.isChecked():
                ax2.plot(time, ysc, 'ks',label="Shock center")
                self.PLOTKEYS.append("SHOCK_CENTER "+units)

            if self.ui.checkBox_shockmodel.isChecked():
                ax2.plot(time, ysm, 'r--',label="Shock-model distance")
                self.PLOTKEYS.append("SHOCK_TO_MODEL "+units)

            if self.ui.checkBox_ypos.isChecked():
                ax2.plot(time, yypos, 'ks',label="Vertical position")
                self.PLOTKEYS.append("MODEL_YPOS "+units)

            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("%s"%(self.ui.comboBox_units.currentText()))
            ax2.figure.tight_layout()
            legend = ax2.legend()
            legend.set_draggable(True)
            #ax2.figure.legend()
            ax2.figure.canvas.draw()

            # Save to dictionary data structure
            output_dict = {"TIME [s]":time}
            length_units = [ym75,ym25,ymc,yp25,yp75,ysc,ysm,yypos]
            length_labels= ["MODEL_-0.75R","MODEL_-0.25R","MODEL_CENTER","MODEL_0.25R","MODEL_0.75R",
                            "SHOCK_CENTER","SHOCK_TO_MODEL","MODEL_YPOS"]
            for k in range(0,len(length_units)):
                output_dict[length_labels[k]+" "+units] = length_units[k]

            px_units = [ymarea,ysarea,radius_masked]
            px_labels= ["MODEL_AREA [px]", "SHOCK_AREA [px]", "MODEL_RADIUS [px]"]
            for k in range(0,len(px_units)):
                output_dict[px_labels[k]] = px_units[k]

            output_dict['CONFIG'] = ['UNITS: %s'%units,'MODEL_DIAMETER: %.2f'%diameter,"FPS: %.2f"%fps, "MASK_NFRAMES: %i"%maskn]
            self.time_series = output_dict.copy()
            #self.ui.textBrowser.setText(str(self.time_series.keys()))

            # Update ui metrics
            self.ui.doubleSpinBox_fit_start_time.setMinimum(time[0])
            self.ui.doubleSpinBox_fit_start_time.setMaximum(time[-1])
            self.ui.doubleSpinBox_fit_last_time.setMaximum(time[-1])
            self.ui.doubleSpinBox_fit_last_time.setMinimum(time[0])
            
            self.ui.doubleSpinBox_fit_start_time.setValue(time[0])
            self.ui.doubleSpinBox_fit_last_time.setValue(time[-1])

            # Update data summary with pixel length
            summary = self.ui.label_data_summary.text()
            lines = summary.strip().split('\n')
            if lines[-1][0:5] == "Pixel":
                lines[-1] = "Pixel length %s: %.4f"%(self.ui.comboBox_units.currentText(), pixel_length)
            else:
                lines.append("Pixel length %s: %.4f"%(self.ui.comboBox_units.currentText(), pixel_length))
            newsummary =''
            for line in lines:
                newsummary += line+'\n'
            self.ui.label_data_summary.setText(newsummary.strip())

            # Infobar update
            self.ui.basebar.setText("Finished plotting data")
        else:
            # Infobar update
            self.ui.basebar.setText("Not enough data to plot")

    def export_to_csv(self):
        if self.time_series is not None:
            dialog = QtWidgets.QFileDialog()
            pathmask = dialog.getSaveFileName(None, "Export CSV","", "CSV files (*.csv)")

            ### Convert time series into dataframe
            df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in self.time_series.items() ]))

            ### Convert fits into dataframe
            if self.fit_dict is not None:
                df_fit = pd.DataFrame(self.fit_dict)
                df = df.join(df_fit)
            df.to_csv(pathmask[0])

    def fit_data(self):
        if self.time_series is not None:
            ### Retrieve user inputs for time/units
            units = self.ui.comboBox_units.currentText()
            time = np.array(self.time_series["TIME [s]"])
            t0 = self.ui.doubleSpinBox_fit_start_time.value()
            t1 = self.ui.doubleSpinBox_fit_last_time.value()
            dt = t1-t0

            ### identify relevant indicies
            inds = (time>t0)*(time<t1)

            ### Initialize data structure
            fit_dict = {}

            ### Get list of keys to loop through
            keys = list(self.time_series.keys())
            keys.remove("TIME [s]")
            keys.remove("CONFIG")
            
            ### Linear fits
            if self.ui.comboBox_fit_type.currentText() == "linear":
                longstring = ""
                for key in keys:
                    t = time[inds]
                    y = self.time_series[key][inds]
                    fitp,cov = np.ma.polyfit(t,y,1,cov=True)
                    err = np.sqrt(np.diag(cov))
                    m,b = fitp[0],fitp[1]
                    fit_dict[key+"_LINEAR_FIT"] = (fitp,err)
                    if key in self.PLOTKEYS:
                        longstring += key+ "\tMin = %f\t Max = %f\t Delta = %f\n"%(y.min(),y.max(),y.max()-y.min())
                        longstring += "\tLINEAR FIT: y = mx+b \t"
                        longstring += "m = %f+-%f %s"%(m,err[0],units+"/s") + "\t"
                        longstring += "b = %f+-%f %s"%(b,err[1],units) + "\n\n"
                        
                self.ui.textBrowser.setText(longstring)

            ### Quadratic fits
            if self.ui.comboBox_fit_type.currentText() == "quadratic":
                longstring =""
                for key in keys:
                    t = time[inds]
                    y = self.time_series[key][inds]
                    fitp,cov = np.polyfit(t,y,2,cov=True)
                    err = np.sqrt(np.diag(cov))
                    a,b,c = fitp[0],fitp[1],fitp[2]
                    fit_dict[key+"_QUADRATIC_FIT"] = (fitp,err)
                    if key in self.PLOTKEYS:
                        longstring += key+ "\tMin = %f\t Max = %f\t Delta = %f\n"%(y.min(),y.max(),y.max()-y.min())
                        longstring += "\tQUAD FIT: y = a*t^2 + b*t + c \n"
                        longstring += "a = %f+-%f %s"%(a,err[0],units+"/s^2") + "\t"
                        longstring += "b = %f+-%f %s"%(b,err[1],units+"/s") + "\t"
                        longstring += "c = %f+-%f %s"%(c,err[2],units) +"\n\n"
                self.ui.textBrowser.setText(longstring)

            self.fit_dict = fit_dict

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())

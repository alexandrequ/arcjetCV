# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1097, 635)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 1081, 561))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.layoutWidget = QtWidgets.QWidget(self.tab)
        self.layoutWidget.setGeometry(QtCore.QRect(780, 410, 281, 25))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_resetFrame = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_resetFrame.setObjectName("pushButton_resetFrame")
        self.horizontalLayout.addWidget(self.pushButton_resetFrame)
        self.pushButton_findEdges = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_findEdges.setObjectName("pushButton_findEdges")
        self.horizontalLayout.addWidget(self.pushButton_findEdges)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(780, 450, 157, 50))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.checkBox_writeVideo = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBox_writeVideo.setObjectName("checkBox_writeVideo")
        self.verticalLayout.addWidget(self.checkBox_writeVideo)
        self.checkBox_writePickle = QtWidgets.QCheckBox(self.verticalLayoutWidget)
        self.checkBox_writePickle.setObjectName("checkBox_writePickle")
        self.verticalLayout.addWidget(self.checkBox_writePickle)
        self.pushButton_runEdgesFullVideo = QtWidgets.QPushButton(self.tab)
        self.pushButton_runEdgesFullVideo.setGeometry(QtCore.QRect(980, 480, 80, 23))
        self.pushButton_runEdgesFullVideo.setObjectName("pushButton_runEdgesFullVideo")
        self.pushButton_stop = QtWidgets.QPushButton(self.tab)
        self.pushButton_stop.setGeometry(QtCore.QRect(980, 480, 80, 23))
        self.pushButton_stop.setObjectName("pushButton_stop")
        self.pushButton_stop.hide()
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        self.groupBox.setGeometry(QtCore.QRect(780, 210, 281, 191))
        self.groupBox.setObjectName("groupBox")
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.groupBox)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(10, 30, 261, 151))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.comboBox_filterType = QtWidgets.QComboBox(self.formLayoutWidget_2)
        self.comboBox_filterType.setObjectName("comboBox_filterType")
        self.comboBox_filterType.addItem("")
        self.comboBox_filterType.addItem("")
        self.comboBox_filterType.addItem("")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_filterType)
        self.label_4 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_4.setObjectName("label_4")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.minIntensity = QtWidgets.QDoubleSpinBox(self.formLayoutWidget_2)
        self.minIntensity.setMaximum(242)
        self.minIntensity.setProperty("value", 150)
        self.minIntensity.setObjectName("minIntensity")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.minIntensity)
        self.maxIntensity = QtWidgets.QDoubleSpinBox(self.formLayoutWidget_2)
        self.maxIntensity.setMaximum(255)
        self.maxIntensity.setProperty("value", 255)
        self.maxIntensity.setObjectName("maxIntensity")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.maxIntensity)
        self.minHue = QtWidgets.QDoubleSpinBox(self.formLayoutWidget_2)
        self.minHue.setMaximum(180)
        self.minHue.setProperty("value", 95)
        self.minHue.setObjectName("minHue")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.minHue)
        self.maxHue = QtWidgets.QDoubleSpinBox(self.formLayoutWidget_2)
        self.maxHue.setMaximum(180)
        self.maxHue.setProperty("value", 140)
        self.maxHue.setObjectName("maxHue")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.maxHue)
        self.label_5 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.label_6 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_6.setObjectName("label_6")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.label_8 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_8.setObjectName("label_8")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.label_9 = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_9.setObjectName("label_9")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_2.setGeometry(QtCore.QRect(780, 40, 281, 161))
        self.groupBox_2.setObjectName("groupBox_2")
        self.formLayoutWidget = QtWidgets.QWidget(self.groupBox_2)
        self.formLayoutWidget.setGeometry(QtCore.QRect(10, 30, 261, 121))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label_7 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.spinBox_firstFrame = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.spinBox_firstFrame.setMaximum(100000)
        self.spinBox_firstFrame.setProperty("value", 900)
        self.spinBox_firstFrame.setObjectName("spinBox_firstFrame")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.spinBox_firstFrame)
        self.label = QtWidgets.QLabel(self.formLayoutWidget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.spinBox_minArea = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.spinBox_minArea.setObjectName("spinBox_minArea")
        self.spinBox_minArea.setDecimals(4)
        self.spinBox_minArea.setProperty("value", 0.012)
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.spinBox_minArea)
        self.label_2 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.spinBox_minStingArea = QtWidgets.QDoubleSpinBox(self.formLayoutWidget)
        self.spinBox_minStingArea.setObjectName("spinBox_minStingArea")
        self.spinBox_minStingArea.setProperty("value", 0.5)
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.spinBox_minStingArea)
        self.label_3 = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.comboBox_flowDirection = QtWidgets.QComboBox(self.formLayoutWidget)
        self.comboBox_flowDirection.setObjectName("comboBox_flowDirection")
        self.comboBox_flowDirection.addItem("")
        self.comboBox_flowDirection.addItem("")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.comboBox_flowDirection)
        self.frame = QtWidgets.QFrame(self.tab)
        self.frame.setGeometry(QtCore.QRect(10, 40, 751, 471))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_img = QtWidgets.QLabel(self.frame)
        self.label_img.setGeometry(QtCore.QRect(10, 10, 731, 451))
        self.label_img.setText("")
        self.label_img.setObjectName("label_img")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.tab_2)
        self.graphicsView_2.setGeometry(QtCore.QRect(10, 10, 351, 311))
        self.graphicsView_2.setTabletTracking(False)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.postProcessParams = QtWidgets.QGroupBox(self.tab_2)
        self.postProcessParams.setGeometry(QtCore.QRect(750, 20, 341, 301))
        self.postProcessParams.setObjectName("postProcessParams")
        self.formLayoutWidget_3 = QtWidgets.QWidget(self.postProcessParams)
        self.formLayoutWidget_3.setGeometry(QtCore.QRect(30, 30, 291, 145))
        self.formLayoutWidget_3.setObjectName("formLayoutWidget_3")
        self.formLayout_3 = QtWidgets.QFormLayout(self.formLayoutWidget_3)
        self.formLayout_3.setContentsMargins(0, 0, 0, 0)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_10 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_10.setObjectName("label_10")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.spinBox_firstFrame_2 = QtWidgets.QDoubleSpinBox(self.formLayoutWidget_3)
        self.spinBox_firstFrame_2.setMaximum(5000)
        self.spinBox_firstFrame_2.setObjectName("spinBox_firstFrame_2")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.spinBox_firstFrame_2)
        self.label_11 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_11.setObjectName("label_11")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.spinBox_3 = QtWidgets.QDoubleSpinBox(self.formLayoutWidget_3)
        self.spinBox_3.setObjectName("spinBox_3")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.spinBox_3)
        self.label_12 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_12.setObjectName("label_12")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.spinBox_4 = QtWidgets.QDoubleSpinBox(self.formLayoutWidget_3)
        self.spinBox_4.setMaximum(10000)
        self.spinBox_4.setObjectName("spinBox_4")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.spinBox_4)
        self.label_13 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_13.setObjectName("label_13")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.comboBox_flowDirectionPP = QtWidgets.QComboBox(self.formLayoutWidget_3)
        self.comboBox_flowDirectionPP.setObjectName("comboBox_flowDirectionPP")
        self.comboBox_flowDirectionPP.addItem("")
        self.comboBox_flowDirectionPP.addItem("")
        self.formLayout_3.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.comboBox_flowDirectionPP)
        self.spinBox = QtWidgets.QDoubleSpinBox(self.formLayoutWidget_3)
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(5)
        self.spinBox.setObjectName("spinBox")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.spinBox)
        self.label_14 = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_14.setObjectName("label_14")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.textBrowser = QtWidgets.QTextBrowser(self.tab_2)
        self.textBrowser.setGeometry(QtCore.QRect(10, 340, 721, 181))
        self.textBrowser.setObjectName("textBrowser")
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.tab_2)
        self.graphicsView_3.setGeometry(QtCore.QRect(380, 10, 351, 311))
        self.graphicsView_3.setTabletTracking(False)
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1097, 20))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        self.menuFilters = QtWidgets.QMenu(self.menuMenu)
        self.menuFilters.setObjectName("menuFilters")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionLoad_video = QtWidgets.QAction(MainWindow)
        self.actionLoad_video.setObjectName("actionLoad_video")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionSave_Filter = QtWidgets.QAction(MainWindow)
        self.actionSave_Filter.setObjectName("actionSave_Filter")
        self.actionLoad_Filter_2 = QtWidgets.QAction(MainWindow)
        self.actionLoad_Filter_2.setObjectName("actionLoad_Filter_2")
        self.actionSave_Filter_2 = QtWidgets.QAction(MainWindow)
        self.actionSave_Filter_2.setObjectName("actionSave_Filter_2")
        self.actionExit_2 = QtWidgets.QAction(MainWindow)
        self.actionExit_2.setObjectName("actionExit_2")
        self.menuFilters.addAction(self.actionLoad_Filter_2)
        self.menuFilters.addAction(self.actionSave_Filter_2)
        self.menuMenu.addAction(self.actionLoad_video)
        self.menuMenu.addAction(self.menuFilters.menuAction())
        self.menuMenu.addAction(self.actionExit_2)
        self.menubar.addAction(self.menuMenu.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "arcjetCV"))
        self.pushButton_resetFrame.setText(_translate("MainWindow", "Reset frame"))
        self.pushButton_findEdges.setText(_translate("MainWindow", "Find Edges"))
        self.checkBox_writeVideo.setText(_translate("MainWindow", "Write video?"))
        self.checkBox_writePickle.setText(_translate("MainWindow", "Write pickle file?"))
        self.pushButton_runEdgesFullVideo.setText(_translate("MainWindow", "Run"))
        self.pushButton_stop.setText(_translate("MainWindow", "Stop"))
        self.groupBox.setTitle(_translate("MainWindow", "Filter options"))
        self.comboBox_filterType.setItemText(0, _translate("MainWindow", "default"))
        self.comboBox_filterType.setItemText(1, _translate("MainWindow", "GRAY"))
        self.comboBox_filterType.setItemText(2, _translate("MainWindow", "HSV-sq"))
        self.label_4.setText(_translate("MainWindow", "Filter type:"))
        self.label_5.setText(_translate("MainWindow", "Min Intensity"))
        self.label_6.setText(_translate("MainWindow", "Max Intensity"))
        self.label_8.setText(_translate("MainWindow", "Min Hue (0-180)"))
        self.label_9.setText(_translate("MainWindow", "Max Hue (0-180)"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Image parameters"))
        self.label_7.setText(_translate("MainWindow", "Start frame #"))
        self.label.setText(_translate("MainWindow", "Min model area (%)"))
        self.label_2.setText(_translate("MainWindow", "Min sting area (%)"))
        self.label_3.setText(_translate("MainWindow", "Flow direction:"))
        self.comboBox_flowDirection.setItemText(0, _translate("MainWindow", "right"))
        self.comboBox_flowDirection.setItemText(1, _translate("MainWindow", "left"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Extract Edges"))
        self.postProcessParams.setTitle(_translate("MainWindow", "Post process parameters"))
        self.label_10.setText(_translate("MainWindow", "Frames per second"))
        self.label_11.setText(_translate("MainWindow", "Min model area (%)"))
        self.label_12.setText(_translate("MainWindow", "Model radius"))
        self.label_13.setText(_translate("MainWindow", "Flow direction:"))
        self.comboBox_flowDirectionPP.setItemText(0, _translate("MainWindow", "right"))
        self.comboBox_flowDirectionPP.setItemText(1, _translate("MainWindow", "left"))
        self.label_14.setText(_translate("MainWindow", "Skip index"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Post Process"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.menuFilters.setTitle(_translate("MainWindow", "Filters"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionLoad_video.setText(_translate("MainWindow", "Load video"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionSave_Filter.setText(_translate("MainWindow", "Save Filter"))
        self.actionLoad_Filter_2.setText(_translate("MainWindow", "Load Filter"))
        self.actionSave_Filter_2.setText(_translate("MainWindow", "Save Filter"))
        self.actionExit_2.setText(_translate("MainWindow", "Exit"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

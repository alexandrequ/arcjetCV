# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'arcjetCV.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap, QColor

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1146, 677)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.tab)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_img = QtWidgets.QLabel(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_img.sizePolicy().hasHeightForWidth())
        self.label_img.setSizePolicy(sizePolicy)
        self.label_img.setMinimumSize(QtCore.QSize(731, 451))
        self.label_img.setMaximumSize(QtCore.QSize(731, 451))
        self.label_img.setMouseTracking(True)
        self.label_img.setText("")
        self.label_img.setObjectName("label_img")
        self.verticalLayout.addWidget(self.label_img)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_8.addLayout(self.verticalLayout)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem1)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.pushButton_loadVideo = QtWidgets.QPushButton(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_loadVideo.sizePolicy().hasHeightForWidth())
        self.pushButton_loadVideo.setSizePolicy(sizePolicy)
        self.pushButton_loadVideo.setObjectName("pushButton_loadVideo")
        self.verticalLayout_6.addWidget(self.pushButton_loadVideo)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setMinimumSize(QtCore.QSize(350, 0))
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.formLayout.setObjectName("formLayout")
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.spinBox_FrameIndex = QtWidgets.QSpinBox(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.spinBox_FrameIndex.sizePolicy().hasHeightForWidth())
        self.spinBox_FrameIndex.setSizePolicy(sizePolicy)
        self.spinBox_FrameIndex.setMaximum(100000)
        self.spinBox_FrameIndex.setProperty("value", 0)
        self.spinBox_FrameIndex.setObjectName("spinBox_FrameIndex")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.spinBox_FrameIndex)
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.comboBox_flowDirection = QtWidgets.QComboBox(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_flowDirection.sizePolicy().hasHeightForWidth())
        self.comboBox_flowDirection.setSizePolicy(sizePolicy)
        self.comboBox_flowDirection.setObjectName("comboBox_flowDirection")
        self.comboBox_flowDirection.addItem("")
        self.comboBox_flowDirection.addItem("")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_flowDirection)
        self.comboBox_filterType = QtWidgets.QComboBox(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_filterType.sizePolicy().hasHeightForWidth())
        self.comboBox_filterType.setSizePolicy(sizePolicy)
        self.comboBox_filterType.setObjectName("comboBox_filterType")
        self.comboBox_filterType.addItem("")
        self.comboBox_filterType.addItem("")
        self.comboBox_filterType.addItem("")
        self.comboBox_filterType.addItem("")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_filterType)
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.verticalLayout_5.addLayout(self.formLayout)
        self.FilterTabs = QtWidgets.QTabWidget(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.FilterTabs.sizePolicy().hasHeightForWidth())
        self.FilterTabs.setSizePolicy(sizePolicy)
        self.FilterTabs.setObjectName("FilterTabs")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout_2.setLabelAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_8 = QtWidgets.QLabel(self.tab_3)
        self.label_8.setObjectName("label_8")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.minHue = QtWidgets.QSpinBox(self.tab_3)
        self.minHue.setMaximum(180)
        self.minHue.setProperty("value", 0)
        self.minHue.setObjectName("minHue")
        self.horizontalLayout_3.addWidget(self.minHue)
        self.maxHue = QtWidgets.QSpinBox(self.tab_3)
        self.maxHue.setMaximum(180)
        self.maxHue.setProperty("value", 121)
        self.maxHue.setObjectName("maxHue")
        self.horizontalLayout_3.addWidget(self.maxHue)
        self.formLayout_2.setLayout(0, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_3)
        self.label_9 = QtWidgets.QLabel(self.tab_3)
        self.label_9.setObjectName("label_9")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.minSaturation = QtWidgets.QSpinBox(self.tab_3)
        self.minSaturation.setMaximum(255)
        self.minSaturation.setProperty("value", 0)
        self.minSaturation.setObjectName("minSaturation")
        self.horizontalLayout_4.addWidget(self.minSaturation)
        self.maxSaturation = QtWidgets.QSpinBox(self.tab_3)
        self.maxSaturation.setMaximum(255)
        self.maxSaturation.setProperty("value", 125)
        self.maxSaturation.setObjectName("maxSaturation")
        self.horizontalLayout_4.addWidget(self.maxSaturation)
        self.formLayout_2.setLayout(1, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_4)
        self.label_5 = QtWidgets.QLabel(self.tab_3)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.minIntensity = QtWidgets.QSpinBox(self.tab_3)
        self.minIntensity.setMaximum(242)
        self.minIntensity.setProperty("value", 150)
        self.minIntensity.setObjectName("minIntensity")
        self.horizontalLayout_2.addWidget(self.minIntensity)
        self.maxIntensity = QtWidgets.QSpinBox(self.tab_3)
        self.maxIntensity.setMaximum(255)
        self.maxIntensity.setProperty("value", 255)
        self.maxIntensity.setObjectName("maxIntensity")
        self.horizontalLayout_2.addWidget(self.maxIntensity)
        self.formLayout_2.setLayout(2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_2)
        self.verticalLayout_2.addLayout(self.formLayout_2)
        self.FilterTabs.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.tab_4)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.formLayout_4.setLabelAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_15 = QtWidgets.QLabel(self.tab_4)
        self.label_15.setObjectName("label_15")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.minHue_2 = QtWidgets.QSpinBox(self.tab_4)
        self.minHue_2.setMaximum(180)
        self.minHue_2.setProperty("value", 125)
        self.minHue_2.setObjectName("minHue_2")
        self.horizontalLayout_6.addWidget(self.minHue_2)
        self.maxHue_2 = QtWidgets.QSpinBox(self.tab_4)
        self.maxHue_2.setMaximum(180)
        self.maxHue_2.setProperty("value", 170)
        self.maxHue_2.setObjectName("maxHue_2")
        self.horizontalLayout_6.addWidget(self.maxHue_2)
        self.formLayout_4.setLayout(1, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_6)
        self.label_16 = QtWidgets.QLabel(self.tab_4)
        self.label_16.setObjectName("label_16")
        self.formLayout_4.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.minSaturation_2 = QtWidgets.QSpinBox(self.tab_4)
        self.minSaturation_2.setMaximum(255)
        self.minSaturation_2.setProperty("value", 40)
        self.minSaturation_2.setObjectName("minSaturation_2")
        self.horizontalLayout_7.addWidget(self.minSaturation_2)
        self.maxSaturation_2 = QtWidgets.QSpinBox(self.tab_4)
        self.maxSaturation_2.setMaximum(255)
        self.maxSaturation_2.setProperty("value", 80)
        self.maxSaturation_2.setObjectName("maxSaturation_2")
        self.horizontalLayout_7.addWidget(self.maxSaturation_2)
        self.formLayout_4.setLayout(2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_7)
        self.label_6 = QtWidgets.QLabel(self.tab_4)
        self.label_6.setObjectName("label_6")
        self.formLayout_4.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.minIntensity_2 = QtWidgets.QSpinBox(self.tab_4)
        self.minIntensity_2.setMaximum(242)
        self.minIntensity_2.setProperty("value", 85)
        self.minIntensity_2.setObjectName("minIntensity_2")
        self.horizontalLayout_5.addWidget(self.minIntensity_2)
        self.maxIntensity_2 = QtWidgets.QSpinBox(self.tab_4)
        self.maxIntensity_2.setMaximum(255)
        self.maxIntensity_2.setProperty("value", 230)
        self.maxIntensity_2.setObjectName("maxIntensity_2")
        self.horizontalLayout_5.addWidget(self.maxIntensity_2)
        self.formLayout_4.setLayout(3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_5)
        self.horizontalLayout_9.addLayout(self.formLayout_4)
        self.FilterTabs.addTab(self.tab_4, "")
        self.verticalLayout_5.addWidget(self.FilterTabs)
        self.verticalLayout_6.addWidget(self.groupBox_2)
        self.groupBox = QtWidgets.QGroupBox(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.formLayout_5 = QtWidgets.QFormLayout()
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_17 = QtWidgets.QLabel(self.groupBox)
        self.label_17.setObjectName("label_17")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.spinBox_FirstGoodFrame = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_FirstGoodFrame.setMaximum(100000)
        self.spinBox_FirstGoodFrame.setProperty("value", 0)
        self.spinBox_FirstGoodFrame.setObjectName("spinBox_FirstGoodFrame")
        self.horizontalLayout_10.addWidget(self.spinBox_FirstGoodFrame)
        self.spinBox_LastGoodFrame = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox_LastGoodFrame.setMaximum(100000)
        self.spinBox_LastGoodFrame.setProperty("value", 0)
        self.spinBox_LastGoodFrame.setObjectName("spinBox_LastGoodFrame")
        self.horizontalLayout_10.addWidget(self.spinBox_LastGoodFrame)
        self.formLayout_5.setLayout(0, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_10)
        self.label_20 = QtWidgets.QLabel(self.groupBox)
        self.label_20.setObjectName("label_20")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.lineEdit_filename = QtWidgets.QLineEdit(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_filename.sizePolicy().hasHeightForWidth())
        self.lineEdit_filename.setSizePolicy(sizePolicy)
        self.lineEdit_filename.setObjectName("lineEdit_filename")
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_filename)
        self.verticalLayout_3.addLayout(self.formLayout_5)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.checkBox_writeVideo = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_writeVideo.setObjectName("checkBox_writeVideo")
        self.horizontalLayout.addWidget(self.checkBox_writeVideo)
        self.pushButton_process = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_process.setObjectName("pushButton_process")
        self.horizontalLayout.addWidget(self.pushButton_process)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem2)
        self.verticalLayout_6.addWidget(self.groupBox)
        self.horizontalLayout_8.addLayout(self.verticalLayout_6)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.Window1 = QtWidgets.QWidget(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Window1.sizePolicy().hasHeightForWidth())
        self.Window1.setSizePolicy(sizePolicy)
        self.Window1.setMinimumSize(QtCore.QSize(400, 350))
        self.Window1.setObjectName("Window1")
        self.horizontalLayout_12.addWidget(self.Window1)
        self.Window2 = QtWidgets.QWidget(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Window2.sizePolicy().hasHeightForWidth())
        self.Window2.setSizePolicy(sizePolicy)
        self.Window2.setMinimumSize(QtCore.QSize(400, 350))
        self.Window2.setObjectName("Window2")
        self.horizontalLayout_12.addWidget(self.Window2)
        self.verticalLayout_8.addLayout(self.horizontalLayout_12)
        self.textBrowser = QtWidgets.QTextBrowser(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)
        self.textBrowser.setMaximumSize(QtCore.QSize(16777215, 250))
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_8.addWidget(self.textBrowser)
        self.horizontalLayout_11.addLayout(self.verticalLayout_8)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.pushButton_LoadFiles = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_LoadFiles.setObjectName("pushButton_LoadFiles")
        self.horizontalLayout_13.addWidget(self.pushButton_LoadFiles)
        self.pushButton_export_csv = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_export_csv.setObjectName("pushButton_export_csv")
        self.horizontalLayout_13.addWidget(self.pushButton_export_csv)
        self.verticalLayout_7.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.pushButton_PlotData = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_PlotData.setObjectName("pushButton_PlotData")
        self.horizontalLayout_15.addWidget(self.pushButton_PlotData)
        self.pushButton_fitData = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_fitData.setObjectName("pushButton_fitData")
        self.horizontalLayout_15.addWidget(self.pushButton_fitData)
        self.verticalLayout_7.addLayout(self.horizontalLayout_15)
        self.tabWidget_2 = QtWidgets.QTabWidget(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget_2.sizePolicy().hasHeightForWidth())
        self.tabWidget_2.setSizePolicy(sizePolicy)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.formLayout_3 = QtWidgets.QFormLayout(self.tab_5)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label = QtWidgets.QLabel(self.tab_5)
        self.label.setObjectName("label")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.label_11 = QtWidgets.QLabel(self.tab_5)
        self.label_11.setObjectName("label_11")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.label_18 = QtWidgets.QLabel(self.tab_5)
        self.label_18.setObjectName("label_18")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.doubleSpinBox_diameter = QtWidgets.QDoubleSpinBox(self.tab_5)
        self.doubleSpinBox_diameter.setMinimum(0.1)
        self.doubleSpinBox_diameter.setMaximum(1000.0)
        self.doubleSpinBox_diameter.setProperty("value", 4.0)
        self.doubleSpinBox_diameter.setObjectName("doubleSpinBox_diameter")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_diameter)
        self.comboBox_units = QtWidgets.QComboBox(self.tab_5)
        self.comboBox_units.setObjectName("comboBox_units")
        self.comboBox_units.addItem("")
        self.comboBox_units.addItem("")
        self.comboBox_units.addItem("")
        self.comboBox_units.addItem("")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_units)
        self.doubleSpinBox_fps = QtWidgets.QDoubleSpinBox(self.tab_5)
        self.doubleSpinBox_fps.setMinimum(1.0)
        self.doubleSpinBox_fps.setMaximum(120.0)
        self.doubleSpinBox_fps.setProperty("value", 30.0)
        self.doubleSpinBox_fps.setObjectName("doubleSpinBox_fps")
        self.formLayout_3.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_fps)
        self.spinBox_mask_frames = QtWidgets.QSpinBox(self.tab_5)
        self.spinBox_mask_frames.setMinimumSize(QtCore.QSize(0, 0))
        self.spinBox_mask_frames.setMaximumSize(QtCore.QSize(150, 30))
        self.spinBox_mask_frames.setMinimum(1)
        self.spinBox_mask_frames.setMaximum(1000)
        self.spinBox_mask_frames.setProperty("value", 50)
        self.spinBox_mask_frames.setObjectName("spinBox_mask_frames")
        self.formLayout_3.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.spinBox_mask_frames)
        self.label_2 = QtWidgets.QLabel(self.tab_5)
        self.label_2.setMaximumSize(QtCore.QSize(100, 30))
        self.label_2.setObjectName("label_2")
        self.formLayout_3.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.tabWidget_2.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.tab_6)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab_6)
        self.groupBox_3.setObjectName("groupBox_3")
        self.formLayout_6 = QtWidgets.QFormLayout(self.groupBox_3)
        self.formLayout_6.setObjectName("formLayout_6")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.formLayout_6.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox)
        self.label_12 = QtWidgets.QLabel(self.groupBox_3)
        self.label_12.setObjectName("label_12")
        self.formLayout_6.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.label_13 = QtWidgets.QLabel(self.groupBox_3)
        self.label_13.setObjectName("label_13")
        self.formLayout_6.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.label_14 = QtWidgets.QLabel(self.groupBox_3)
        self.label_14.setObjectName("label_14")
        self.formLayout_6.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.doubleSpinBox_fit_start_time = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.doubleSpinBox_fit_start_time.setObjectName("doubleSpinBox_fit_start_time")
        self.formLayout_6.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_fit_start_time)
        self.doubleSpinBox_fit_last_time = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.doubleSpinBox_fit_last_time.setObjectName("doubleSpinBox_fit_last_time")
        self.formLayout_6.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_fit_last_time)
        self.verticalLayout_10.addWidget(self.groupBox_3)
        self.tabWidget_2.addTab(self.tab_6, "")
        self.verticalLayout_7.addWidget(self.tabWidget_2)
        self.groupBox_data_summary = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_data_summary.setObjectName("groupBox_data_summary")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBox_data_summary)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.label_data_summary = QtWidgets.QLabel(self.groupBox_data_summary)
        self.label_data_summary.setText("")
        self.label_data_summary.setObjectName("label_data_summary")
        self.verticalLayout_9.addWidget(self.label_data_summary)
        self.verticalLayout_7.addWidget(self.groupBox_data_summary)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_7.addItem(spacerItem3)
        self.groupBox_XT_params = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_XT_params.setObjectName("groupBox_XT_params")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_XT_params)
        self.gridLayout.setObjectName("gridLayout")
        self.checkBox_75_radius = QtWidgets.QCheckBox(self.groupBox_XT_params)
        self.checkBox_75_radius.setChecked(False)
        self.checkBox_75_radius.setObjectName("checkBox_75_radius")
        self.gridLayout.addWidget(self.checkBox_75_radius, 4, 0, 1, 1)
        self.checkBox_m25_radius = QtWidgets.QCheckBox(self.groupBox_XT_params)
        self.checkBox_m25_radius.setChecked(True)
        self.checkBox_m25_radius.setObjectName("checkBox_m25_radius")
        self.gridLayout.addWidget(self.checkBox_m25_radius, 1, 0, 1, 1)
        self.checkBox_ypos = QtWidgets.QCheckBox(self.groupBox_XT_params)
        self.checkBox_ypos.setObjectName("checkBox_ypos")
        self.gridLayout.addWidget(self.checkBox_ypos, 4, 1, 1, 1)
        self.checkBox_25_radius = QtWidgets.QCheckBox(self.groupBox_XT_params)
        self.checkBox_25_radius.setChecked(True)
        self.checkBox_25_radius.setObjectName("checkBox_25_radius")
        self.gridLayout.addWidget(self.checkBox_25_radius, 3, 0, 1, 1)
        self.checkBox_shockmodel = QtWidgets.QCheckBox(self.groupBox_XT_params)
        self.checkBox_shockmodel.setObjectName("checkBox_shockmodel")
        self.gridLayout.addWidget(self.checkBox_shockmodel, 3, 1, 1, 1)
        self.checkBox_shock_center = QtWidgets.QCheckBox(self.groupBox_XT_params)
        self.checkBox_shock_center.setObjectName("checkBox_shock_center")
        self.gridLayout.addWidget(self.checkBox_shock_center, 2, 1, 1, 1)
        self.checkBox_model_center = QtWidgets.QCheckBox(self.groupBox_XT_params)
        self.checkBox_model_center.setChecked(True)
        self.checkBox_model_center.setObjectName("checkBox_model_center")
        self.gridLayout.addWidget(self.checkBox_model_center, 2, 0, 1, 1)
        self.checkBox_m75_radius = QtWidgets.QCheckBox(self.groupBox_XT_params)
        self.checkBox_m75_radius.setChecked(False)
        self.checkBox_m75_radius.setTristate(False)
        self.checkBox_m75_radius.setObjectName("checkBox_m75_radius")
        self.gridLayout.addWidget(self.checkBox_m75_radius, 0, 0, 1, 1)
        self.checkBox_model_area = QtWidgets.QCheckBox(self.groupBox_XT_params)
        self.checkBox_model_area.setObjectName("checkBox_model_area")
        self.gridLayout.addWidget(self.checkBox_model_area, 1, 1, 1, 1)
        self.checkBox_shock_area = QtWidgets.QCheckBox(self.groupBox_XT_params)
        self.checkBox_shock_area.setObjectName("checkBox_shock_area")
        self.gridLayout.addWidget(self.checkBox_shock_area, 0, 1, 1, 1)
        self.verticalLayout_7.addWidget(self.groupBox_XT_params)
        self.horizontalLayout_11.addLayout(self.verticalLayout_7)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayout_4.addWidget(self.tabWidget)
        self.basebar = QtWidgets.QLabel(self.centralwidget)
        self.basebar.setText("")
        self.basebar.setObjectName("basebar")
        self.verticalLayout_4.addWidget(self.basebar)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1146, 22))
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
        self.FilterTabs.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_loadVideo.setText(_translate("MainWindow", "Load Video"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Input parameters"))
        self.label_7.setText(_translate("MainWindow", "Frame Index:"))
        self.label_3.setText(_translate("MainWindow", "Flow direction:"))
        self.comboBox_flowDirection.setItemText(0, _translate("MainWindow", "right"))
        self.comboBox_flowDirection.setItemText(1, _translate("MainWindow", "left"))
        self.comboBox_filterType.setItemText(0, _translate("MainWindow", "AutoHSV"))
        self.comboBox_filterType.setItemText(1, _translate("MainWindow", "CNN"))
        self.comboBox_filterType.setItemText(2, _translate("MainWindow", "HSV"))
        self.comboBox_filterType.setItemText(3, _translate("MainWindow", "GRAY"))
        self.label_4.setText(_translate("MainWindow", "Filter Method:"))
        self.label_8.setText(_translate("MainWindow", "Hue (0-180)"))
        self.label_9.setText(_translate("MainWindow", "Saturation (0-255)"))
        self.label_5.setText(_translate("MainWindow", "Value (0-255)"))
        self.FilterTabs.setTabText(self.FilterTabs.indexOf(self.tab_3), _translate("MainWindow", "Model Filter"))
        self.label_15.setText(_translate("MainWindow", "Hue (0-180)"))
        self.label_16.setText(_translate("MainWindow", "Saturation (0-255)"))
        self.label_6.setText(_translate("MainWindow", "Value (0-255)"))
        self.FilterTabs.setTabText(self.FilterTabs.indexOf(self.tab_4), _translate("MainWindow", "Shock Filter"))
        self.groupBox.setTitle(_translate("MainWindow", "Output parameters"))
        self.label_17.setText(_translate("MainWindow", "Frame range:"))
        self.label_20.setText(_translate("MainWindow", "Output filename:"))
        self.checkBox_writeVideo.setText(_translate("MainWindow", "Write video?"))
        self.pushButton_process.setText(_translate("MainWindow", "Process All"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Extract Edges"))
        self.pushButton_LoadFiles.setText(_translate("MainWindow", "Load Files"))
        self.pushButton_export_csv.setText(_translate("MainWindow", "Export CSV"))
        self.pushButton_PlotData.setText(_translate("MainWindow", "Plot Data"))
        self.pushButton_fitData.setText(_translate("MainWindow", "Fit Data"))
        self.label.setText(_translate("MainWindow", "Model diameter: "))
        self.label_11.setText(_translate("MainWindow", "Length units:"))
        self.label_18.setText(_translate("MainWindow", "Frames per sec:"))
        self.comboBox_units.setItemText(0, _translate("MainWindow", "[in]"))
        self.comboBox_units.setItemText(1, _translate("MainWindow", "[cm]"))
        self.comboBox_units.setItemText(2, _translate("MainWindow", "[mm]"))
        self.comboBox_units.setItemText(3, _translate("MainWindow", "pixels"))
        self.label_2.setText(_translate("MainWindow", "Mask nframes:"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_5), _translate("MainWindow", "Plotting params"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Fitting Parameters"))
        self.comboBox.setItemText(0, _translate("MainWindow", "linear"))
        self.comboBox.setItemText(1, _translate("MainWindow", "quadratic"))
        self.comboBox.setItemText(2, _translate("MainWindow", "exponential"))
        self.label_12.setText(_translate("MainWindow", "Fit type:"))
        self.label_13.setText(_translate("MainWindow", "Start time:"))
        self.label_14.setText(_translate("MainWindow", "End time:"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_6), _translate("MainWindow", "Fitting params"))
        self.groupBox_data_summary.setTitle(_translate("MainWindow", "Data Summary"))
        self.groupBox_XT_params.setTitle(_translate("MainWindow", "Visible Traces on XT plot"))
        self.checkBox_75_radius.setText(_translate("MainWindow", "+75% radius"))
        self.checkBox_m25_radius.setText(_translate("MainWindow", "-25% radius"))
        self.checkBox_ypos.setText(_translate("MainWindow", "Model y-position"))
        self.checkBox_25_radius.setText(_translate("MainWindow", "+25% radius "))
        self.checkBox_shockmodel.setText(_translate("MainWindow", "Shock-model dist"))
        self.checkBox_shock_center.setText(_translate("MainWindow", "Shock center"))
        self.checkBox_model_center.setText(_translate("MainWindow", "Model center"))
        self.checkBox_m75_radius.setText(_translate("MainWindow", "-75% radius"))
        self.checkBox_model_area.setText(_translate("MainWindow", "Model area"))
        self.checkBox_shock_area.setText(_translate("MainWindow", "Shock area"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Analyze"))
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

    ### logo
    import cv2 as cv
    import numpy as np
    frame = cv.imread("gui/logo/arcjetCV_logo.png")
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    h,w,chan = np.shape(image)
    qimg = QImage(image.data, w, h, w*3, QImage.Format_RGB888)
    logo = QPixmap.fromImage(qimg)
    logo = logo.scaledToHeight(451)
    SCALE_FACTOR = 1004/452
    ui.label_img.setPixmap(logo)

    # ### Connect interface
    # ui.pushButton_process.clicked.connect(process_all)
    # ui.actionLoad_video.triggered.connect(load_video)
    # ui.label_img.newCursorValue.connect(getPixel)
    # ui.pushButton_LoadFiles.clicked.connect(load_outputs)
    # ui.pushButton_PlotData.clicked.connect(plot_outputs)

    MainWindow.show()
    sys.exit(app.exec_())


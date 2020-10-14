import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class TrackLabel(QtWidgets.QLabel):
    newCursorValue = QtCore.pyqtSignal(list)

    def mouseMoveEvent(self,event):
        self.newCursorValue.emit([event.x(), event.y(),self.width(), self.height()])

class MplCanvas(FigureCanvas):
    """ Convenience class to embed matplotlib canvas

    Args:
        FigureCanvas (matplotlib object): matplotlib canvas object
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        self.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.updateGeometry(self)


class MatplotlibWidget(QtWidgets.QWidget):
    """Plotting widget using matplotlib, embedding into Qt

    Args:
        MplCanvas (FigureCanvas): matplotlib standard canvas
    """
    def __init__(self,*args):
        super().__init__(*args)
        layout = QtWidgets.QVBoxLayout(self)

        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        
        self.ax = self.canvas.figure.subplots()
        # t = np.linspace(0, 10, 501)
        # self.ax.plot(t, np.tan(t), ".")
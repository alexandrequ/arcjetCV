import numpy as np
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class TrackLabel(QtWidgets.QLabel):
    newCursorValue = QtCore.pyqtSignal(list)

    def mouseMoveEvent(self,event):
        self.newCursorValue.emit([event.x(), event.y()])

class MplCanvas(FigureCanvas):
    """ Convenience class to embed matplotlib canvas

    Args:
        FigureCanvas (matplotlib object): matplotlib canvas object
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MatplotlibWidget(QtWidgets.QWidget):
    """Plotting widget using matplotlib, embedding into Qt

    Args:
        MplCanvas (FigureCanvas): matplotlib standard canvas
    """
    def __init__(self,*args):
        super().__init__(*args)
        layout = QtWidgets.QVBoxLayout(self)

        static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        toolbar = NavigationToolbar(static_canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(static_canvas)
        
        self.ax = static_canvas.figure.subplots()
        # t = np.linspace(0, 10, 501)
        # self.ax.plot(t, np.tan(t), ".")
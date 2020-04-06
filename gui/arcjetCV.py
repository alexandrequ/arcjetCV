#!/usr/bin/env python3

# Filename: arcjetCV.py

"""
arcjetCV is a simple application which measures
realtime recession using openCV, Python, and PyQt5.
"""

import sys
from functools import partial

# Import QApplication and the required widgets from PyQt5.QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QVBoxLayout

__version__ = '0.1'
__author__ = 'Magnus Albert Haw'

# Create a subclass of QMainWindow to setup the calculator's GUI
class arcUi(QMainWindow):
    """arcjetCV's View (GUI)."""
    def __init__(self):
        """View initializer."""
        super().__init__()
        # Set some main window's properties
        self.setWindowTitle('arcjetCV')
        self.setFixedSize(235, 235)
        # Set the central widget and the general layout
        self.generalLayout = QVBoxLayout()
        self._centralWidget = QWidget(self)
        self.setCentralWidget(self._centralWidget)
        self._centralWidget.setLayout(self.generalLayout)
        # Create the display
        self._createDisplay()

    def _createDisplay(self):
        """Create the display."""
        # Create the display widget
        self.display = QLineEdit()
        # Set some display's properties
        self.display.setFixedHeight(35)
        self.display.setAlignment(Qt.AlignRight)
        self.display.setReadOnly(True)
        # Add the display to the general layout
        self.generalLayout.addWidget(self.display)
        
    def setDisplayText(self, text):
        """Set display's text."""
        self.display.setText(text)
        self.display.setFocus()

    def displayText(self):
        """Get display's text."""
        return self.display.text()

    def clearDisplay(self):
        """Clear the display."""
        self.setDisplayText('')

# Create a Controller class to connect the GUI and the model
class PyCalcCtrl:
    """PyCalc's Controller."""
    def __init__(self, model, view):
        """Controller initializer."""
        self._evaluate = model
        self._view = view
        # Connect signals and slots
        self._connectSignals()

    def _calculateResult(self):
        """Evaluate expressions."""
        result = self._evaluate(expression=self._view.displayText())
        self._view.setDisplayText(result)

    def _buildExpression(self, sub_exp):
        """Build expression."""
        if self._view.displayText() == ERROR_MSG:
            self._view.clearDisplay()

        expression = self._view.displayText() + sub_exp
        self._view.setDisplayText(expression)

    def _connectSignals(self):
        """Connect signals and slots."""
        pass
##        for btnText, btn in self._view.buttons.items():
##            if btnText not in {'=', 'C'}:
##                btn.clicked.connect(partial(self._buildExpression, btnText))
##
##        self._view.buttons['='].clicked.connect(self._calculateResult)
##        self._view.display.returnPressed.connect(self._calculateResult)
##        self._view.buttons['C'].clicked.connect(self._view.clearDisplay)

def evaluateExpression(expression):
    """Evaluate an expression."""
    try:
        result = str(eval(expression, {}, {}))
    except Exception:
        result = ERROR_MSG

    return result

# Client code
def main():
    """Main function."""
    # Create an instance of QApplication
    pycalc = QApplication(sys.argv)
    # Show the calculator's GUI
    view = arcUi()
    view.show()
    # Create instances of the model and the controller
    model = evaluateExpression
    PyCalcCtrl(model=model, view=view)
    # Execute the calculator's main loop
    sys.exit(pycalc.exec_())

if __name__ == '__main__':
    main()

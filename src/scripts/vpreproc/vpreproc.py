#
#pregui.py
#
import sys
import string
import os
from PyQt4 import QtCore, QtGui, QtWebKit
import ConfigParser
from isis import data

import MainWindow


# Execute Function
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = MainWindow.MainWindow(app)
    myapp.show()
    sys.exit(app.exec_())
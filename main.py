import myUI

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow


class MyWindow(QMainWindow, myUI.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())



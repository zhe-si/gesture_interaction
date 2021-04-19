import sys

import qdarkstyle
from PyQt5 import QtCore, QtWidgets
from PyQt5.Qt import QTimer
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QWidget


class state_show(QWidget):
    def setupUi(self, MainWindow, gesture):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(200, 100)
        MainWindow.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        MainWindow.setWindowFlag(Qt.FramelessWindowHint)  # 隐藏标题栏
        MainWindow.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        MainWindow.setWindowOpacity(0.8)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 25, 200, 50))
        # gesture = '鼠标模式'
        self.label.setText(gesture)
        self.label.setStyleSheet("font: 75 15pt \"Cambria\";")
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = state_show()
    ui.setupUi(MainWindow, "鼠标模式")
    MainWindow.show()
    QTimer.singleShot(50000, app.quit)  # 0.5s后关闭窗口
    sys.exit(app.exec_())

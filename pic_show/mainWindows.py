"""

按钮控件（QPushButton）

QAbstractButton

QPushButton
AToolButton
QRadioButton
QCheckBox


"""

import sys
from time import sleep

import pyautogui
from PyQt5.QtWidgets import *
import subprocess
import control
from control.imageview_control import normal, mouse


class QPushButtonDemo(QDialog):
    def __init__(self):
        super(QPushButtonDemo, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('QPushButton Demo')

        layout = QVBoxLayout()

        self.button1 = QPushButton('第1个按钮')
        self.button1.setText('ppt')
        self.button1.setCheckable(True)
        self.button1.toggle()
        # self.button1.clicked.connect(self.buttonState)
        self.button1.clicked.connect(lambda: self.whichButton(self.button1))

        layout.addWidget(self.button1)
        self.setLayout(layout)
        self.resize(400, 300)

        # 在文本前面显示图像

    def buttonState(self):
        if self.button1.isChecked():
            print('按钮1已经被选中')
        else:
            print('按钮1未被选中')

    def whichButton(self, btn):
        subprocess.Popen([r'D:\myapp\Fastpic\fastpic.exe', 'imageExample\\1.jpg'])
        sleep(3)
        control.imageview_control.run()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QPushButtonDemo()
    main.show()
    sys.exit(app.exec_())

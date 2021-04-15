"""

让程序定时关闭

QTimer.singleShot

"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gesture = 'click'
    label = QLabel(gesture)
    # palette = QPalette()  # 创建调色板类实例
    # palette.setColor(QPalette.Window, Qt.darkBlue)  # 设置为蓝色
    # label.setPalette(palette)
    # label.setAlignment(Qt.AlignCenter)  # 设置居中
    label.setFrameShape(QFrame.Box)
    label.setStyleSheet('border-width: 20px;border-style: solid;label - color: rgb(255, 170, 0);background - color: '
                        'rgb(100, 149, 237);')
    label.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
    # 获取屏幕坐标系
    # screen = QDesktopWidget().screenGeometry()
    # 获取窗口坐标系
    # size = label.geometry()
    # newLeft = (screen.width() - size.width()) / 2
    # newTop = (screen.height() - size.height()) / 2
    # label.move(newLeft, newTop)
    # label.setAlignment(Qt.AlignCenter)
    label.show()
    QTimer.singleShot(2000, app.quit)
    sys.exit(app.exec_())

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget,QLabel,QPushButton
from PyQt5.Qt import QTimer
import sys
"""
显示一个透明窗口,0.5秒后自动关闭
"""
class win(QWidget): #创建一个类，为了集成控件
    def __init__(self):
        super(win, self).__init__()
        self.setWindowTitle('定时器的使用')
        self.resize(200,200)
        self.setWindowOpacity(0.6)
        self.setWindowFlag(Qt.FramelessWindowHint)#隐藏标题栏
        #self.setup_ui()
        #self.num=0

    def setup_ui(self):
        self.timer = QTimer(self)  # 初始化一个定时器
        self.timer.timeout.connect(self.operate)  # 每次计时到时间时发出信号
        self.timer.start(1000)  # 设置计时间隔并启动；单位毫秒

    def operate(self):
        self.num=self.num+1
        print(self.num)

if __name__=='__main__':
    app=QApplication(sys.argv)  #创建应用
    window=win()
    window.show()
    QTimer.singleShot(500,app.quit)#0.5s后关闭窗口
    sys.exit(app.exec_())
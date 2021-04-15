"""
放大crtl + +     big
缩小crtl + -     small
顺时针旋转 crtl + shift + +   clockwise
逆时针旋转 crtl + shift + -   anticlockwise
编辑 alt + r            edit
退出 esc                esc
上一页 上/左箭头         up/left
下一页 下/右箭头         down/right
抓取进入手型状态 ctrl + h       hand
选择状态 CTRL + r              select
"""
from time import sleep
import sys
from PyQt5.QtWidgets import QApplication, QWidget
import pyautogui
from pykeyboard import *
from communication import receive

#鼠标移到屏幕中央
def mouseCenter():
    screenWidth, screenHeight = pyautogui.size()
    pyautogui.moveTo(x=screenWidth/2, y=screenHeight/2)
    pyautogui.click()

def handle(message):
    if(message=="big"):
        #pyautogui.hotkey('ctrl','+')      wps失效,效果为旋转
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(1)
        pyautogui.keyUp('ctrl')
        print("big")
    if(message=='small'):
        #pyautogui.hotkey('ctrl','-')    wps失效,效果为旋转
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-1)
        pyautogui.keyUp('ctrl')
        print("small")
    if (message == "clockwise"):
        pyautogui.hotkey('ctrl', 'shift', '+')
        print("clockwise")
    if (message == 'anticlockwise'):
        pyautogui.hotkey('ctrl', 'shift', '-')
        print("anticlockwise")
    if (message == "click"):
        pyautogui.click()
        print("click")
    if(message=='up'):
        pyautogui.keyDown('up')
        pyautogui.keyUp('up')
        print('up')
    if (message == 'down'):
        pyautogui.keyDown('down')
        pyautogui.keyUp('down')
        print('down')
    if (message == 'left'):
        pyautogui.keyDown('left')
        pyautogui.keyUp('left')
        print('left')
    if (message == 'right'):
        pyautogui.keyDown('right')
        pyautogui.keyUp('right')
        print('right')
    if(message=='edit'):
        pyautogui.hotkey('alt','r')
        print('edit')
    if(message=='esc'):
        # pyautogui.keyDown('esc')
        # pyautogui.keyUp('esc')
        pyautogui.press('esc')
        print('esc')
    if(message=='hand'):
        pyautogui.hotkey('ctrl','h')
        print('hand')
    if(message=='select'):
        pyautogui.hotkey('ctrl', 'r')
        print('select')

#提取信息
def handleMessage(message):
    begin=0
    end=0
    for i in message:
        begin=begin+1
        if(i=='f'):
            break


def stateWindow(state):
    app = QApplication(sys.argv)

    w = QWidget()
    w.resize(250, 150)
    #w.move(300, 300)
    w.setWindowTitle(state)
    w.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    #receiver = receive.Receiver()
    #receiver.start()

    ans = ['big', 'small', 'clockwise', 'anticlockwise', 'edit', 'esc', 'up', 'down',
           'left', 'right', 'hand', 'select']
    sleep(5)
    #mouseCenter()
    #pyautogui.hotkey('shift','f5')
    print(pyautogui.position())
    while(True):
        #handle("edit")
        pyautogui.hotkey('shift', 'f5')
        sleep(3)
        pyautogui.press('escape')
        handle('esc')
        sleep(3)
    # for i in ans:
    #     handle(i)
    #     sleep(1)







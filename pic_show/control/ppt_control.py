"""
放大crtl + +     big
缩小crtl + -     small
编辑 alt + r            edit
退出 esc                esc
上一页 上/左箭头         up/left
下一页 下/右箭头         down/right
抓取进入手型状态 ctrl + h       hand
选择状态 CTRL + r              select
从当前播放 shift + f5
由于ppt没有旋转
所以 clockwise 对应 从当前播放 shift +f5
anticlockwise 对应 退出播放 esc
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

def normal(message,flag):
    if (message['gesture'] == "big"):
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(1)
        pyautogui.keyUp('ctrl')
        print("big")
        return 0
    if (message['gesture'] == 'small'):
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-1)
        pyautogui.keyUp('ctrl')
        print("small")
        return 0
    if (message['gesture'] == "clockwise"):
        pyautogui.hotkey('shift', 'f5')
        print("clockwise")
        return 0
    if (message['gesture'] == 'anticlockwise'):
        pyautogui.press('esc')
        print("anticlockwise")
        return 0
    if (message['gesture'] == "click"):
        pyautogui.click()
        print("click")
        return 0
    if (message['gesture'] == 'catch'):
        print('catch')
        return 1
    if ((message['gesture'] == 'swap')&(flag==0)):
        if(message['vector'][0]>0):
            pyautogui.press('left')
            print('left')
            return 0
        if(message['vector'][0]<0):
            pyautogui.press('right')
            print('right')
            return 0
    if ((message['gesture'] == 'swap') & (flag == 1)):
        pyautogui.dragRel(message['vector'][0],message['vector'][1],duration=0.1)
        print('catch and swap')
        return 1

def mouse(message,flag):
    if (message['gesture'] == "big"):
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(1)
        pyautogui.keyUp('ctrl')
        print("big")
        return 0
    if (message['gesture'] == 'small'):
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-1)
        pyautogui.keyUp('ctrl')
        print("small")
        return 0
    if (message['gesture'] == "clockwise"):
        pyautogui.hotkey('shift', 'f5')
        print("clockwise")
        return 0
    if (message['gesture'] == 'anticlockwise'):
        pyautogui.press('esc')
        print("anticlockwise")
        return 0
    if (message['gesture'] == "click"):
        pyautogui.click()
        print("click")
        return 0
    if (message['gesture'] == 'catch'):
        print('catch')
        return 1
    if ((message['gesture'] == 'swap') & (flag == 0)):
        pyautogui.moveRel(message['vector'][0], message['vector'][1], duration=0.1)
        return 0
    if ((message['gesture'] == 'swap') & (flag == 1)):
        pyautogui.dragRel(message['vector'][0], message['vector'][1], duration=0.1)
        print('catch and swap')
        return 1

if __name__ == '__main__':
    ans = ['big', 'small', 'clockwise', 'anticlockwise', 'edit', 'esc', 'up', 'down',
           'left', 'right', 'hand', 'select']
    # receiver = receive.Receiver()
    # receiver.start()
    flag1 = 0
    flag2 = 0
    while True:
        # message1 = receiver.get_packages()[0]
        # message2 = receiver.get_packages()[0]
        # message3 = receiver.get_packages()[0]

        message1={'gesture':'clockwise','send_time':1}
        message2 = {'gesture':'clockwise', 'send_time':2}
        message3 = {'gesture':'clockwise', 'send_time':3}
        sleep(3)
        state='normal'
        if(((message1['gesture']==message2['gesture'])&(message1['gesture']==message3['gesture']))
                &((message1['send_time']<message2['send_time'])
                &(message2['send_time']<message3['send_time']))):
            if(message1['gesture']=='silent'):
                flag1=0
                flag2=0
                state='silent'
                print('silent')
                continue
            elif(message1['gesture']=='normal'):
                state='normal'
                print('normal')
            elif(message1['gesture']=='mouse'):
                state='mouse'
                print('mouse')
            if(state=='normal'):
                flag1 = normal(message1, flag1)
            elif(state=='mouse'):
                flag2 = mouse(message1, flag2)
        else:
            continue

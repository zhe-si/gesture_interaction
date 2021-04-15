"""
放大crtl + +    big
缩小crtl + -    small
编辑 alt + r            edit
退出 esc                esc
上一页 上/左箭头         left
下一页 下/右箭头         right
抓取进入手型状态 ctrl + h       hand
选择状态 CTRL + r              select
ctrl + r        clockwise
ctrl + l        anticlockwise
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
        pyautogui.scroll(1)
        print("big")
        return 0
    if (message['gesture'] == 'small'):
        pyautogui.scroll(-1)
        print("small")
        return 0
    if (message['gesture'] == "clockwise"):
        pyautogui.hotkey('ctrl', 'r')
        print("clockwise")
        return 0
    if (message['gesture'] == 'anticlockwise'):
        pyautogui.hotkey('ctrl', 'l')
        print("anticlockwise")
        return 0
    if (message['gesture'] == "clickdown"):
        pyautogui.mouseDown()
        print("clickdown")
        return 0
    if (message['gesture'] == "clickup"):
        pyautogui.mouseUp()
        print("clickup")
        return 0
    if (message['gesture'] == 'catch'):
        print('catch')
        return 1
    if ((message['gesture'] == 'swap')&(flag==0)):
        if(message['vector'][0]>0):
            pyautogui.press('left')
            # pyautogui.keyDown('left')
            # pyautogui.keyUp('left')
            print('left')
            return 0
        if(message['vector'][0]<0):
            pyautogui.press('right')
            print('right')
            return 0
    if ((message['gesture'] == 'swap') & (flag == 1)):
        pyautogui.dragRel(message['vector'][0], message['vector'][1], duration=0.1)
        print('catch and swap')
        return 1

def mouse(message,flag):
    if (message['gesture'] == "big"):
        pyautogui.scroll(1)
        print("big")
        return 0
    if (message['gesture'] == 'small'):
        pyautogui.scroll(-1)
        print("small")
        return 0
    if (message['gesture'] == "clockwise"):
        pyautogui.hotkey('ctrl', 'r')
        print("clockwise")
        return 0
    if (message['gesture'] == 'anticlockwise'):
        pyautogui.hotkey('ctrl', 'l')
        print("anticlockwise")
        return 0
    if (message['gesture'] == "clickdown"):
        pyautogui.mouseDown()
        print("clickdown")
        return 0
    if (message['gesture'] == "clickup"):
        pyautogui.mouseUp()
        print("clickup")
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


def run():
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

        message1 = {'gesture':'big', 'send_time':1}
        message2 = {'gesture':'big', 'send_time':2}
        message3 = {'gesture':'big', 'send_time':3}

        # message1 = {'gesture': 'swap', 'send_time': 1, 'vector': (1, 1)}
        # message2 = {'gesture': 'swap', 'send_time': 2, 'vector': (1, 1)}
        # message3 = {'gesture': 'swap', 'send_time': 3, 'vector': (1, 1)}

        sleep(3)
        state = 'normal'
        if (((message1['gesture'] == message2['gesture']) & (message1['gesture'] == message3['gesture']))
                & ((message1['send_time'] < message2['send_time'])
                   & (message2['send_time'] < message3['send_time']))):
            if (state == 'silent'):
                flag1 = 0
                flag2 = 0
                print('silent')
                continue
            elif (state == 'normal'):
                flag1 = normal(message1, flag1)
                print('normal')
            elif (state == 'mouse'):
                flag2 = mouse(message1, flag2)
                print('mouse')

            # if(state=='normal'):
            #     flag1 = normal(message1, flag1)
            # elif(state=='mouse'):
            #     flag2 = mouse(message1, flag2)

        else:
            continue


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
        """


        message1 = {'gesture':'big', 'send_time':1}
        message2 = {'gesture':'big', 'send_time':2}
        message3 = {'gesture':'big', 'send_time':3}
        """
        message1 = {'gesture': 'swap', 'send_time': 1,'vector':(1,1)}
        message2 = {'gesture': 'swap', 'send_time': 2,'vector':(1,1)}
        message3 = {'gesture': 'swap', 'send_time': 3,'vector':(1,1)}




        sleep(3)
        state='normal'
        if(((message1['gesture']==message2['gesture'])&(message1['gesture']==message3['gesture']))
                &((message1['send_time']<message2['send_time'])
                &(message2['send_time']<message3['send_time']))):
            if(state =='silent'):
                flag1=0
                flag2=0
                print('silent')
                continue
            elif(state =='normal'):
                flag1=normal(message1, flag1)
                print('normal')
            elif(state =='mouse'):
                flag2=mouse(message1, flag2)
                print('mouse')

            # if(state=='normal'):
            #     flag1 = normal(message1, flag1)
            # elif(state=='mouse'):
            #     flag2 = mouse(message1, flag2)

        else:
            continue

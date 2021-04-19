from time import sleep
import sys
import pyautogui
from pykeyboard import *
from communication import receive


# 鼠标移到屏幕中央
def mouseCenter():
    screenWidth, screenHeight = pyautogui.size()
    pyautogui.moveTo(x=screenWidth / 2, y=screenHeight / 2)
    pyautogui.click()


def normal(message, mouse_state):
    if message['gesture'] == "big":
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(1)
        pyautogui.keyUp('ctrl')
        print("big")
        mouse_state = {'isCatch': False, 'isClickDown': False}
        return mouse_state
    if message['gesture'] == 'small':
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-1)
        pyautogui.keyUp('ctrl')
        print("small")
        mouse_state = {'isCatch': False, 'isClickDown': False}
        return mouse_state
    if message['gesture'] == "clockwise":
        pyautogui.hotkey('shift', 'f5')
        print("clockwise")
        mouse_state = {'isCatch': False, 'isClickDown': False}
        return mouse_state
    if message['gesture'] == 'anticlockwise':
        pyautogui.press('esc')
        print("anticlockwise")
        mouse_state = {'isCatch': False, 'isClickDown': False}
        return mouse_state
    if (message['gesture'] == "clickdown") & (mouse_state['isClickDown'] == False):
        print("clickdown")
        mouse_state = {'isCatch': False, 'isClickDown': True}
        return mouse_state
    if (message['gesture'] == "clickup") & (mouse_state['isClickDown'] == True):
        pyautogui.click()
        print("clickup")
        mouse_state = {'isCatch': False, 'isClickDown': False}
        return mouse_state
    if (message['gesture'] == 'catch') & (mouse_state['isCatch'] == False):
        print('catch')
        mouse_state = {'isCatch': True, 'isClickDown': False}
        return mouse_state
    if (message['gesture'] == 'swap') & (mouse_state['isCatch'] == False):
        if message['vector'][0] > 0:
            pyautogui.press('left')
            print('left')
            mouse_state = {'isCatch': False, 'isClickDown': False}
            return mouse_state
        if message['vector'][0] < 0:
            pyautogui.press('right')
            print('right')
            mouse_state = {'isCatch': False, 'isClickDown': False}
            return mouse_state
    if (message['gesture'] == 'swap') & (mouse_state['isCatch'] == True):
        pyautogui.dragRel(message['vector'][0], message['vector'][1], duration=0.1)
        print('catch and swap')
        mouse_state = {'isCatch': True, 'isClickDown': False}
        return mouse_state


def mouse(message, mouse_state):
    if message['gesture'] == "big":
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(1)
        pyautogui.keyUp('ctrl')
        print("big")
        mouse_state = {'isCatch': False, 'isClickDown': False}
        return mouse_state
    if message['gesture'] == 'small':
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-1)
        pyautogui.keyUp('ctrl')
        print("small")
        mouse_state = {'isCatch': False, 'isClickDown': False}
        return mouse_state
    if message['gesture'] == "clockwise":
        pyautogui.hotkey('shift', 'f5')
        print("clockwise")
        mouse_state = {'isCatch': False, 'isClickDown': False}
        return mouse_state
    if message['gesture'] == 'anticlockwise':
        pyautogui.press('esc')
        print("anticlockwise")
        mouse_state = {'isCatch': False, 'isClickDown': False}
        return mouse_state
    if (message['gesture'] == "clickdown") & (mouse_state['isClickDown'] == False):
        print("clickdown")
        mouse_state = {'isCatch': False, 'isClickDown': True}
        return mouse_state
    if (message['gesture'] == "clickup") & (mouse_state['isClickDown'] == True):
        pyautogui.click()
        print("clickup")
        mouse_state = {'isCatch': False, 'isClickDown': False}
        return mouse_state
    if message['gesture'] == 'catch':
        print('catch')
        mouse_state = {'isCatch': True, 'isClickDown': False}
        return mouse_state
    if (message['gesture'] == 'swap') & (mouse_state['isCatch'] == False):
        pyautogui.moveRel(message['vector'][0], message['vector'][1], duration=0.1)
        mouse_state = {'isCatch': False, 'isClickDown': False}
        return mouse_state
    if (message['gesture'] == 'swap') & (mouse_state['isCatch'] == True):
        pyautogui.dragRel(message['vector'][0], message['vector'][1], duration=0.1)
        print('catch and swap')
        mouse_state = {'isCatch': True, 'isClickDown': False}
        return mouse_state


def run(message, mouse_state, mode):
    if mode == 'silent':
        mode = 'silent'
        print('silent')
    if mode == 'normal':
        mouse_state = normal(message, mouse_state)
    elif mode == 'mouse':
        mouse_state = mouse(message, mouse_state)
    return mouse_state


if __name__ == '__main__':
    run()

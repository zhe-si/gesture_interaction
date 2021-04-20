from time import sleep
import sys
import pyautogui
from pykeyboard import *
from communication import receive


def normal(message):
    if message['gesture'] == "big":
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(1)
        pyautogui.keyUp('ctrl')
        print("big")
    if message['gesture'] == 'small':
        pyautogui.keyDown('ctrl')
        pyautogui.scroll(-1)
        pyautogui.keyUp('ctrl')
        print("small")
    if message['gesture'] == "clockwise":
        pyautogui.hotkey('shift', 'f5')
        print("clockwise")
    if message['gesture'] == 'anticlockwise':
        pyautogui.press('esc')
        print("anticlockwise")
    if message['gesture'] == 'swap':
        if message['vector'][0] > 0:
            pyautogui.press('left')
            print('left')
        if message['vector'][0] < 0:
            pyautogui.press('right')
            print('right')


def run(message, mode):
    if mode == 'silent':
        mode = 'silent'
        print('silent')
    if mode == 'normal':
        normal(message)


if __name__ == '__main__':
    pass

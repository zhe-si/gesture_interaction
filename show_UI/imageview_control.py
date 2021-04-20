from time import sleep
import sys
from PyQt5.QtWidgets import QApplication, QWidget
import pyautogui
from pykeyboard import *
from communication import receive


def normal(message):
    if message['gesture'] == "big":
        pyautogui.scroll(20)
        print("big")
    if message['gesture'] == 'small':
        pyautogui.scroll(-20)
        print("small")
    if message['gesture'] == "clockwise":
        pyautogui.press('up')
        print("clockwise")
    if message['gesture'] == 'anticlockwise':
        pyautogui.hotkey('down')
        print("anticlockwise")
    if (message['gesture'] == 'swap'):
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
    run()

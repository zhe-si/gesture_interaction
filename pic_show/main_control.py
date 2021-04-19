import sys
import time
import pywintypes
import win32gui as w
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from communication import receive


def show_state(state):
    from state_show import state_show
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = state_show()
    ui.setupUi(MainWindow, state)
    MainWindow.show()
    print(state)
    QTimer.singleShot(1000, app.quit)  # 0.5s后关闭窗口
    app.exec_()
    # sys.exit(app.exec_())


def messsage_filter(packege):
    label = packege['gesture'][0]
    message = {}
    if label == "Catch":
        message["gesture"] = "catch"
    elif label == "Click Down":
        message["gesture"] = "clickdown"
    elif label == "Click Up":
        message["gesture"] = "clickup"
    elif label == "Doing other things":
        message["gesture"] = "doingotherthings"
    elif label == "Pushing Two Fingers Away":
        message["gesture"] = "silent"
    elif label == "Shaking Hand":
        message["gesture"] = "mouse"
    elif label == "Swipe":
        message["gesture"] = "swap"
    elif label == "Thumb Up":
        message["gesture"] = "normal"
    elif label == "Turn With Two Hands Clockwise":
        message["gesture"] = "clockwise"
    elif label == "Turn With Two Hands Counterclockwise":
        message["gesture"] = "anticlockwise"
    elif label == "Zooming In With Two Hands":
        message["gesture"] = "small"
    elif label == "Zooming Out With Two Hands":
        message["gesture"] = "big"
    else:
        raise Exception("label type {} not allowed".format(label))
    if message['gesture'] == 'swap':
        message['vector'] = packege['vector']
    return message


def test():
    ans = ['big', 'small', 'clockwise', 'anticlockwise', 'swap', 'catch', 'silent', 'mouse', 'normal']
    message = {'gesture': 'big'}
    # time.sleep(2)
    print(message)
    show_state('big')
    return message


def ppt_run(message, mouse_state, mode):
    from ppt_control import run
    mouse_state = run(message, mouse_state, mode)
    return mouse_state


def imageview_run(message, mouse_state, mode):
    from imageview_control import run
    mouse_state = run(message, mouse_state, mode)
    return mouse_state


def main():
    global mode
    mouse_state = {'isCatch': False, 'isClickDown': False}
    mode = 'normal'
    pop_last_name = 'silent'
    receiver = receive.Receiver()
    receiver.start()
    while True:
        packages = receiver.get_packages()
        while packages is None:
            time.sleep(0.1)
            packages = receiver.get_packages()
        package = packages[0]
        message = messsage_filter(package)
        print(message)
        # message = {'gesture': 'normal'}
        # message = {'gesture': 'big'}

        if (pop_last_name == 'silent') & ((message['gesture'] == 'normal') | (message['gesture'] == 'mouse')):
            continue
        if message['gesture'] == 'silent':
            mode = 'silent'
            print('silent')
            if pop_last_name != 'silent':
                pop_last_name = 'silent'
                show_state('silent')
            continue
        if message['gesture'] == 'normal':
            mode = 'normal'
            print('normal')
            if pop_last_name != 'normal':
                pop_last_name = 'normal'
                show_state('normal')
            continue
        if message['gesture'] == 'mouse':
            mode = 'mouse'
            print('mouse')
            if pop_last_name != 'mouse':
                pop_last_name = 'mouse'
                show_state('mouse')
            continue
        running_app = w.GetWindowText(w.GetForegroundWindow())  # 检测最上册窗口的名字
        print(running_app)
        if running_app.find("JPEGView") != -1:
            imageview_run(message, mouse_state, mode)
            print("JPEGView")
        if running_app.find("WPS Office") != -1:
            ppt_run(message, mouse_state, mode)
            print("WPS Office")


if __name__ == '__main__':
    main()

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


def messsage_change(packege):
    label = packege['gesture']
    message = {}
    if label == "Catch":
        message["gesture"] = "catch"
    elif label == "Click Down":
        message["gesture"] = "clickdown"
    elif label == "Click Up":
        message["gesture"] = "clickup"
    elif label == "Doing other things":
        message["gesture"] = "doingotherthings"
    elif label == "Swipe":
        message["gesture"] = "swap"
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


def message_filer(packages):
    package = {}
    if (packages['gesture'][0] == 'Click Down') or (packages['gesture'][1] == 'Click Down') \
            or (packages['gesture'][2] == 'Click Down'):
        package['gesture'] = 'Click Down'
    elif packages['gesture'][0] == 'Catch':
        package['gesture'] = 'Catch'
    else:
        package['gesture'] = packages['gesture'][0]
    package['vector'] = packages['vector']
    if package['vector'] is None:
        package['vector'] = (0, 0)
    return messsage_change(package)


def ppt_run(message, mode):
    from ppt_control import run
    mouse_state = run(message, mode)
    return mouse_state


def imageview_run(message, mode):
    from imageview_control import run
    run(message, mode)


# catch为normal，click为silent
def main():
    global mode
    mode = 'normal'
    pop_last_name = 'normal'
    receiver = receive.Receiver()
    receiver.start()
    while True:
        packages = receiver.get_packages()
        while packages is None:
            time.sleep(0.1)
            packages = receiver.get_packages()
        message = message_filer(packages[0])
        print(message)

        if (pop_last_name == 'silent') & (message['gesture'] != 'catch'):
            continue
        print("mode:" + mode)
        if message['gesture'] == 'clickdown':
            mode = 'silent'
            print('silent')
            if pop_last_name != 'silent':
                pop_last_name = 'silent'
                show_state('silent')
            continue
        if message['gesture'] == 'catch':
            mode = 'normal'
            print('normal')
            if pop_last_name != 'normal':
                pop_last_name = 'normal'
                show_state('normal')
            continue
        running_app = w.GetWindowText(w.GetForegroundWindow())  # 检测最上册窗口的名字
        print(running_app)
        if (running_app.find("WPS Office") != -1) \
                & ((running_app.find(".pptx") != -1) | (running_app.find(".ppt") != -1)):
            ppt_run(message, mode)
            print("WPS Office")
        if running_app.find('JPEGView') != -1:
            imageview_run(message, mode)
            print('JPEGView')


if __name__ == '__main__':
    main()

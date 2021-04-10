# 接受数据并通知给观察者
import json
import socket
import threading
from time import sleep


class Receiver:
    """
    支持同步和异步方式接收
    同步：使用wait或wait_all(通过fun回调函数进行事件驱动)
    异步：用start启动，之后用get_packages获取

    注意：读取的字典数据包含时间戳数据："send_time"
    """

    HOST = "localhost"
    PORT = 10889

    def __init__(self, wait_receive_last=False, receive_list_length=10, no_repeat=True):
        """
        *wait_receive_last* 同步接收是否接收后到的旧数据
        *receive_list_length* 异步接收列表最大长度
        *no_repeat* 异步接收，是否读取重复数据
        """
        self.now_time = 0
        self.wait_receive_last = wait_receive_last
        self.receive_list_length = receive_list_length
        self.no_repeat = no_repeat

        self.receive_list = []
        self.receive_list_lock = threading.Lock()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((Receiver.HOST, Receiver.PORT))
        self.socket.listen(1)
        self.sender_socket, self.sender_addr = self.socket.accept()
        print("sender connect successful")

    def wait(self, fun):
        """同步接收方法（一次读）"""
        msg_dict = self._wait_package()
        msg_send_time = msg_dict["send_time"]
        if self.now_time < msg_send_time:
            self.now_time = msg_send_time
            fun(msg_dict)
        elif self.wait_receive_last:
            fun(msg_dict)

    def _wait_package(self):
        bytes_num = self.sender_socket.recv(4)
        bytes_num = int.from_bytes(bytes_num, byteorder="little", signed=True)
        msg_package = self.sender_socket.recv(bytes_num)
        msg_dict = json.loads(msg_package)
        return msg_dict

    def wait_all(self, fun):
        """同步接收方法（一直读）"""
        while True:
            self.wait(fun)

    def start(self):
        """异步启动方法"""
        threading.Thread(target=self._update_receive_list).start()

    def _update_receive_list(self):
        while True:
            msg_dict = self._wait_package()
            self.receive_list_lock.acquire()
            try:
                self.receive_list.append(msg_dict)
                self.receive_list.sort(key=lambda x: x["send_time"])
                if self.receive_list_length < len(self.receive_list):
                    self.receive_list.pop(0)
            finally:
                self.receive_list_lock.release()

    def get_packages(self, num=1):
        """异步数据获取方法"""
        self.receive_list_lock.acquire()
        try:
            receive_list_length = len(self.receive_list)
            if receive_list_length == 0:
                result = None
            elif receive_list_length < num:
                result = self.receive_list.copy()
            else:
                result = self.receive_list[-num:]
            if self.no_repeat:
                self.receive_list.clear()
        finally:
            self.receive_list_lock.release()
        return result


if __name__ == '__main__':
    receiver = Receiver()
    receiver.start()
    while True:
        sleep(1)
        print(receiver.get_packages())

# 数据发送
import json
import socket
import time
from threading import Thread, Lock

from communication.receive import Receiver


class Sender:
    """
    异步数据发送（添加时间戳"send_time"）
    发送方法：send
    """

    def __init__(self, tar_host, tar_port):
        self.tar_host = tar_host
        self.tar_port = tar_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect = False
        self.connect_lock = Lock()
        self.__make_sure_connect()

    def send(self, msg_dict: dict):
        """发送字典格式数据"""
        msg_package = self.__make_msg_package(msg_dict)
        thread = Thread(target=self._send, args=(msg_package,))
        thread.start()

    @staticmethod
    def __make_msg_package(msg_dict: dict):
        msg_dict["send_time"] = time.time()
        msg_json = json.dumps(msg_dict).encode("utf-8")
        return len(msg_json).to_bytes(4, byteorder="little", signed=True) + msg_json

    def _send(self, msg_package: bytes):
        self.socket.send(msg_package)

    def __make_sure_connect(self):
        if not self.connect:
            self.connect_lock.acquire()
            if not self.connect:
                self.socket.connect((self.tar_host, self.tar_port))
                self.connect = True
                print("receiver connect successful")
            self.connect_lock.release()


if __name__ == '__main__':
    sender = Sender("localhost", Receiver.PORT)
    k = {1: 1, 2: "2"}
    while True:
        sender.send(k)
        time.sleep(0.1)

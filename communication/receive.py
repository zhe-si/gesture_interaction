# 接受数据并通知给观察者
import json
import socket


class Receiver:
    HOST = "localhost"
    PORT = 10889

    def __init__(self, receive_last=False):
        self.receive_list = []
        self.now_time = 0
        self.receive_last = receive_last
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((Receiver.HOST, Receiver.PORT))
        self.socket.listen(1)
        self.sender_socket, self.sender_addr = self.socket.accept()
        print("sender connect successful")

    def wait(self, fun):
        # 同步接收方法（一次读）
        bytes_num = self.sender_socket.recv(4)
        bytes_num = int.from_bytes(bytes_num, byteorder="little", signed=True)
        msg_package = self.sender_socket.recv(bytes_num)
        msg_dict = json.loads(msg_package)
        msg_send_time = msg_dict["send_time"]
        if self.now_time < msg_send_time:
            self.now_time = msg_send_time
            fun(msg_dict)
        elif self.receive_last:
            fun(msg_dict)

    def wait_all(self, fun):
        # 同步接收方法（一直读）
        while True:
            self.wait(fun)


if __name__ == '__main__':
    receiver = Receiver()
    receiver.wait_all(print)

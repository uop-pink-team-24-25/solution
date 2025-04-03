#!/usr/bin/env python3

import yaml
import time
from socket import socket, AF_INET, SOCK_STREAM, SHUT_RDWR

from src.interfaces import Send, Martial
from src.martial import JsonMartialler

class SocketSend(Send):
    def __init__(self, martialler: Martial, port, host = ''):
        self.host = host
        self.port = port
        self.martialler = martialler

        # socket creation
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.setblocking(True)
        self.socket.bind((self.host, self.port))

        self.socket.listen(1)
        self.conn, self.addr = self.socket.accept()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def send(self, data):
        sendable = self.martialler.serialise(data)
        print(f"conn on {self.addr}")
        self.conn.sendall(sendable)

    def close(self):
        self.conn.shutdown(SHUT_RDWR)
        self.socket.shutdown(SHUT_RDWR)

        self.conn.close()
        self.socket.close()

if __name__ == '__main__':
    with open('../config.yml' , 'r') as f:
        config = yaml.safe_load(f)['yolov5_deepsort']['main']

    with SocketSend(JsonMartialler(), config['port'], config['host']) as net2:
        net2.send({"test": 3})

    net = SocketSend(JsonMartialler(), config['port'], config['host'])

    net.send({"hello": "world"})
    time.sleep(1)
    net.send({"data": "2"})

    net.close()


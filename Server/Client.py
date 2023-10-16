import socket

from ServerModule.Connection import Connection
from ServerModule.util.Debug import Debug


class Client:

    def __init__(self):
        super().__init__()
        self.connection = None
        self.client_socket = None

        self.ip = 'localhost'
        self.port = 5922

        Debug.log(f"Set connection properties: [IP: {self.ip}, PORT: {self.port}]")

    def start_client(self):
        Debug.log("Trying to connect to server. . .")

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.ip, self.port))

        Debug.log("Successfully connect to server. . .")

    def wait_client(self):
        client_socket, address = self.client_socket.accept()
        self.connection = Connection(client_socket, address)
        self.connection.start()

        Debug.log(f"Client[{address}] is connected to server!")

    def send_packet_to_client(self, packet):
        if self.connection is not None:
            self.connection.send_packet(packet)

    def stop(self):
        self.connection.stop()
        self.client_socket.close()

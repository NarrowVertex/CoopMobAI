import socket

from Server.util.Debug import Debug


class ServerModule:

    def __init__(self):
        super().__init__()
        self.connection = None
        self.server_socket = None

        self.ip = 'localhost'
        self.port = 5922

        Debug.log(f"Set server properties: [IP: {self.ip}, PORT: {self.port}]")

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.ip, self.port))
        self.server_socket.listen()

        Debug.log("Server starts now!")

    def wait_client(self):
        client_socket, address = self.server_socket.accept()
        self.connection = Connection(client_socket, address)
        self.connection.start()

        Debug.log(f"Client[{address}] is connected to server!")

    def send_packet_to_client(self, packet):
        if self.connection is not None:
            self.connection.send_packet(packet)

    def stop(self):
        self.connection.stop()
        self.server_socket.close()

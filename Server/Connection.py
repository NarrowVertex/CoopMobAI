import threading

from ServerModule.ClientCounter import ClientCounter
from ServerModule.PacketDecoder import PacketDecoder
from ServerModule.PacketListener import PacketListener
from ServerModule.util.Bytes import Bytes
from ServerModule.util.Debug import Debug

MAX_CHUNK_SIZE = 2048

class Connection(threading.Thread):

    def __init__(self, socket, address):
        super().__init__()
        self.socket = socket
        self.address = address

        self.stop_flag = False

        self.packet_decoder = PacketDecoder
        self.client_counter = ClientCounter(self)
        self.packet_listener = PacketListener(self, self.client_counter)

        self.debug_mode = False

    def run(self):
        try:
            while not self.stop_flag:
                # data = self.socket.recv(100)
                data = self.receive_data()
                if not data:
                    # disconnect
                    Debug.log(f"Client[{self.address}] is disconnected!!")
                    break

                if self.debug_mode is True:
                    Debug.log(f"received: {data.bytes}")

                packet = self.packet_decoder.decode(data)
                if packet is None:
                    continue

                self.packet_listener.on_receive_packet(packet)
        except ConnectionResetError:
            Debug.log(f"Client[{self.address}] is disconnected!")

    def receive_data(self):
        data = self.socket.recv(4)
        if not data:
            return None

        data_length = int.from_bytes(data, "little")

        chunks = []
        bytes_record = 0
        while bytes_record < data_length:
            chunk = self.socket.recv(min(data_length - bytes_record, MAX_CHUNK_SIZE))

            if chunk == b'':
                raise RuntimeError("socket connection broken")

            chunks.append(chunk)
            bytes_record += len(chunk)

        return Bytes(b''.join(chunks))

    def send_data(self, data):
        # print(data.bytes)
        data = bytes(data.bytes)

        if self.debug_mode is True:
            Debug.log(f"sending: {data}")

        data_length = len(data)
        self.socket.send(data_length.to_bytes(4, byteorder="little"))

        total_sent_byte_length = 0
        while total_sent_byte_length < data_length:
            sent_byte_length = self.socket.send(
                data[total_sent_byte_length:total_sent_byte_length + MAX_CHUNK_SIZE])

            if sent_byte_length == 0:
                raise RuntimeError("socket connection broken")

            total_sent_byte_length += sent_byte_length

    def send_packet(self, packet):
        data = Bytes()
        data.write_byte(packet.packet_id)
        data.write(packet.get_bytes())
        self.send_data(data)

    def override_packet_managers(self, packet_decoder_class, client_counter_class, packet_listener_class):
        self.packet_decoder = packet_decoder_class
        self.client_counter = client_counter_class(self)
        self.packet_listener = packet_listener_class(self, self.client_counter)

    def stop(self):
        self.stop_flag = True

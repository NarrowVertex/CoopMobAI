from Server.util.Bytes import Bytes


class Packet:

    def __init__(self, packet_id):
        self.packet_id = packet_id

    def handle(self, packet_handler):
        pass

    def get_bytes(self) -> Bytes:
        return Bytes()

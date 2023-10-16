from Server.packet.Packet import Packet
from Server.util.Bytes import Bytes


class RemoveTraceMapPacket(Packet):

    def __init__(self):
        super().__init__(5)

    @classmethod
    def write(cls, data):
        return None

    def handle(self, packet_handler):
        pass

    def get_bytes(self):
        data = super().get_bytes()
        data: Bytes
        return data

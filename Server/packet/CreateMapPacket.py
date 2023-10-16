from Server.packet.Packet import Packet
from Server.util.Bytes import Bytes


class CreateMapPacket(Packet):

    def __init__(self, map_data):
        super().__init__(0)
        self.map_data = map_data

    @classmethod
    def write(cls, data):
        return None

    def handle(self, packet_handler):
        pass

    def get_bytes(self):
        data = super().get_bytes()
        data: Bytes
        data.write_map(self.map_data)
        return data

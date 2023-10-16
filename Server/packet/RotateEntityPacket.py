from Server.packet.Packet import Packet
from Server.util.Bytes import Bytes


class RotateEntityPacket(Packet):

    def __init__(self, entity_name, rotation):
        super().__init__(3)
        self.entity_name = entity_name
        self.rotation = rotation

    @classmethod
    def write(cls, data):
        return None

    def handle(self, packet_handler):
        pass

    def get_bytes(self):
        data = super().get_bytes()
        data: Bytes
        data.write_string(self.entity_name)
        data.write_vector(self.rotation)
        return data

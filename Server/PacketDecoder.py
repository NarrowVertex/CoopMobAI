from ServerModule.util.Debug import Debug


class PacketDecoder:

    @classmethod
    def decode(cls, data):
        operator = data.read_byte()

        packet = cls.get_packet(operator, data)
        if packet is not None:
            return packet

        Debug.log(f"Can't find any packet with operator[{operator}]")
        return None

    @classmethod
    def get_packet(cls, operator, data):

        """
        if operator == 0:
            return JoinPacket.write(data)
        elif operator == 1:
            return InitParametersPacket.write(data)
        elif operator == 2:
            return RequestActionPacket.write(data)
        elif operator == 3:
            return LearnPacket.write(data)
        elif operator == 4:
            return UpdateParametersPacket.write(data)
        elif operator == 5:
            return SaveWeightsPacket.write(data)
        """
        return None

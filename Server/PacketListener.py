from ServerModule.ClientCounter import ClientCounter


class PacketListener:

    client_counter: ClientCounter

    def __init__(self, connection, client_counter):
        self.connection = connection
        self.client_counter = client_counter

    def on_receive_packet(self, packet):
        packet.handle(self)

"""
    def handle_join(self, game_id, game_significant_id):
        self.client_counter.create_new_game(game_id, game_significant_id)
        self.connection.send_packet(JoinAcceptPacket(1))
"""



class ClientCounter:

    def __init__(self, connection):
        self.connection = connection
        self.game = None

"""
    def create_new_game(self, game_id, game_significant_id):
        Debug.log(f"Create new game for Client[{self.connection.address}]")
        Debug.log(f"Client significant id: {game_significant_id}")

        from Simulator import Simulator
        self.game, packet_decoder_class, client_counter_class, packet_listener_class \
            = Simulator.instance().game_manager.create_new_game(game_id, game_significant_id)

        self.connection.override_packet_managers(packet_decoder_class, client_counter_class, packet_listener_class)
"""

"""
    def init_parameters(self, epsilon):
        # Debug.log("init parameter")
        # Debug.log(str(epsilon))
        self.game.agent.init_parameters(epsilon)

    def get_action(self, available_action, state):
        # Debug.log("get action")
        # Debug.log(str(available_action))
        # Debug.log(str(state))
        state = self.game.env.state_convert(state)
        action = self.game.agent.policy(available_action, state)

        # send back action to client
        self.connection.send_packet(ActionPacket(action))

    def learn(self, old_state, old_action, current_state, available_actions, reward, is_done):
        # Debug.log("learn")
        
        old_state = self.game.env.state_convert(old_state)
        current_state = self.game.env.state_convert(current_state)
        self.game.agent.learn(old_state, old_action, current_state, available_actions, reward, is_done)

        # send back
        self.connection.send_packet(RespondLearnPacket())

    def update_parameters(self):
        self.game.agent.copy_network()

    def save_weights(self, number):
        self.game.agent.save_weights(number)
"""

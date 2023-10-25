import datetime


class LogManager:

    def __init__(self, data_manager, logs_file):
        self.data_manager = data_manager
        self.logs_file = logs_file
        self.debug_file = None

    def print(self, context, category):
        message = ""

        current_time = datetime.datetime.now()
        time_prefix = current_time.strftime("[%Y-%m-%d|%H:%M:%S]")

        category_prefix = f"[{category}]"

        message += time_prefix + category_prefix + " "
        if "\n" in context:
            message += "\n"
        message += context
        if category == "info":
            print(message)
        message += "\n"
        self.logs_file.write(message)
        self.logs_file.flush()

    def start_episode(self, episode, timestep):
        self.debug_file = self.data_manager.create_debug_file(episode, timestep)
        message = f"start episode : {episode} - {timestep}\n"
        self.debug_file.write(message)
        self.debug_file.flush()

    def start_test_episode(self, episode, timestep):
        self.debug_file = self.data_manager.create_test_debug_file(episode, timestep)
        message = f"start test episode : {episode} - {timestep}\n"
        self.debug_file.write(message)
        self.debug_file.flush()

    def debug(self, context):
        message = ""
        message += context
        message += "\n"
        self.debug_file.write(message)
        self.debug_file.flush()

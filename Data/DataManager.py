import os
import numpy as np

from Utils.IDGenerator import generate_id


class DataManager:

    def __init__(self, save_directory_path="../Saves/"):
        self.significant_id = None
        self.loaded = False

        self.save_directory = save_directory_path  # "../Saves/"
        self.checkpoints_directory = "checkpoints/"
        self.debugs_directory = "debugs/"
        self.description_file_name = "description.txt"
        self.hyperparameters_file_name = "hyperparameters.txt"
        self.options_file_name = "options.txt"
        self.logs_file_name = "logs.csv"
        # self.checkpoint_parameters_directory = "parameters/"
        self.checkpoint_performance_file_name = "performance.txt"
        self.checkpoint_parameter_file_name = "parameter.pth"

        self.description_file = None
        self.hyperparameters_file = None
        self.options_file = None
        self.logs_file = None
        self.debug_file = None

        self.log_manager = None

    def set_log_manager(self, log_manager):
        self.log_manager = log_manager

    def init(self, env_name, is_loading=False, significant_id=None):
        if not is_loading:
            self.significant_id = generate_id()
            self.loaded = False
        else:
            self.significant_id = significant_id
            self.loaded = True
        self.build_path(env_name, self.significant_id)
        # self.log_manager.print("DataManager initialized!", "info")
        # init_type = "Load" if is_loading else "Create"
        # self.log_manager.print(f"Init type : {init_type}, Significant ID : {self.significant_id}, Env : {env_name}", "info")
        return significant_id

    def build_path(self, env_name, significant_id):
        self.save_directory += f"{env_name}/{significant_id}/"
        self.checkpoints_directory = self.save_directory + self.checkpoints_directory
        self.debugs_directory = self.save_directory + self.debugs_directory

        self.description_file_name = self.save_directory + self.description_file_name
        self.hyperparameters_file_name = self.save_directory + self.hyperparameters_file_name
        self.options_file_name = self.save_directory + self.options_file_name
        self.logs_file_name = self.save_directory + self.logs_file_name

    def load(self, model):
        # print(self.checkpoints_directory, next(os.walk(self.checkpoints_directory)))
        last_checkpoint_name = next(os.walk(self.checkpoints_directory))[1][-1]

        tokens = last_checkpoint_name.split("_")
        last_time_step = int(tokens[0])
        last_episode = int(tokens[1])

        self.hyperparameters_file = open_file(self.hyperparameters_file_name)
        self.options_file = open_file(self.options_file_name)
        self.logs_file = open_file_writable(self.logs_file_name)

        last_checkpoint_directory = self.checkpoints_directory + last_checkpoint_name + "/"
        parameter_file_fullname = last_checkpoint_directory + self.checkpoint_parameter_file_name
        model.load(parameter_file_fullname)

        return last_time_step, last_episode

    def load_hyperparameters(self):
        update_timestep, K_epochs, eps_clip, gamma, lr_actor, lr_critic = None, None, None, None, None, None

        try:
            lines = self.hyperparameters_file.readlines()

            for line in lines:
                # 각 줄에서 원하는 하이퍼파라미터 값을 추출합니다.
                if "PPO update frequency" in line:
                    update_timestep = int(line.split(":")[-1].strip().split()[0])
                elif "PPO K epochs" in line:
                    K_epochs = int(line.split(":")[-1].strip())
                elif "PPO epsilon clip" in line:
                    eps_clip = float(line.split(":")[-1].strip())
                elif "discount factor (gamma)" in line:
                    gamma = float(line.split(":")[-1].strip())
                elif "optimizer learning rate actor" in line:
                    lr_actor = float(line.split(":")[-1].strip())
                elif "optimizer learning rate critic" in line:
                    lr_critic = float(line.split(":")[-1].strip())

            return update_timestep, K_epochs, eps_clip, gamma, lr_actor, lr_critic
        except FileNotFoundError:
            print("하이퍼파라미터 파일을 찾을 수 없습니다.")
            return update_timestep, K_epochs, eps_clip, gamma, lr_actor, lr_critic

    def load_options(self):
        max_ep_len, max_training_timesteps, print_freq, log_freq, save_model_freq, state_dim, action_dim = None, None, None, None, None, None, None

        try:
            lines = self.options_file.readlines()

            for line in lines:
                # 각 줄에서 원하는 옵션 값을 추출합니다.
                if "max training timesteps" in line:
                    max_training_timesteps = int(line.split(":")[-1].strip())
                elif "max timesteps per episode" in line:
                    max_ep_len = int(line.split(":")[-1].strip())
                elif "printing average reward over episodes in last" in line:
                    print_freq = int(line.split(":")[-1].strip().split()[0])
                elif "log frequency" in line:
                    log_freq = int(line.split(":")[-1].strip().split()[0])
                elif "model saving frequency" in line:
                    save_model_freq = int(line.split(":")[-1].strip().split()[0])
                elif "state space dimension" in line:
                    state_dim_str = line.split(":")[-1].strip()
                    state_dim = np.fromstring(state_dim_str[1:-1], sep=' ')  # Convert the string to a NumPy array
                elif "action space dimension" in line:
                    action_dim_str = line.split(":")[-1].strip()
                    action_dim = np.fromstring(action_dim_str[1:-1], sep=' ')  # Convert the string to a NumPy array

            return max_ep_len, max_training_timesteps, print_freq, log_freq, save_model_freq, state_dim, action_dim
        except FileNotFoundError:
            print("옵션 파일을 찾을 수 없습니다.")
            return max_ep_len, max_training_timesteps, print_freq, log_freq, save_model_freq, state_dim, action_dim

    def initial_save(self):
        if self.loaded:
            print("Can't initial save after loading. . .")
            return

        # base save directory
        mkdir(self.save_directory)

        # episodes directory
        mkdir(self.checkpoints_directory)
        mkdir(self.debugs_directory)

        self.description_file = create_file(self.description_file_name)
        self.hyperparameters_file = create_file(self.hyperparameters_file_name)
        self.options_file = create_file(self.options_file_name)
        self.logs_file = create_file(self.logs_file_name)

    def checkout(self, episode, time_step, model, elapsed_time=None):
        checkpoint_directory = f"{time_step}_{episode}/"
        checkpoint_directory = self.checkpoints_directory + checkpoint_directory
        mkdir(checkpoint_directory)

        checkpoint_performance_file_fullname = checkpoint_directory + self.checkpoint_performance_file_name
        create_file(checkpoint_performance_file_fullname)

        checkpoint_parameter_file_fullname = checkpoint_directory + self.checkpoint_parameter_file_name

        string = ""
        string += "---------------------------------------------------------------------------------------------\n"
        string += f"saving model at : {checkpoint_directory}                                                    \n"
        model.save(checkpoint_parameter_file_fullname)
        string += "model saved                                                                                  \n"
        if elapsed_time is not None:
            string += f"Elapsed Time : {elapsed_time}                                                           \n"
        string += "---------------------------------------------------------------------------------------------\n"
        self.log_manager.print(string, "info")

    def create_debug_file(self, episode, time_step):
        debug_file_name = f"{episode}.csv"
        debug_file_fullname = self.debugs_directory + debug_file_name
        self.debug_file = create_write_only_file(debug_file_fullname)
        return self.debug_file

    def save_hyperparameters(self, update_timestep, K_epochs, eps_clip, gamma, lr_actor, lr_critic):
        file = self.hyperparameters_file
        string = ""
        string += "---------------------------------------------------------------------------------------------\n"
        string += f"PPO update frequency : {str(update_timestep)} timesteps                                     \n"
        string += f"PPO K epochs : {K_epochs}                                                                   \n"
        string += f"PPO epsilon clip : {eps_clip}                                                               \n"
        string += f"discount factor (gamma) : {gamma}                                                           \n"
        string += f"optimizer learning rate actor : {lr_actor}                                                  \n"
        string += f"optimizer learning rate critic : {lr_critic}                                                \n"
        string += "---------------------------------------------------------------------------------------------\n"
        file.write(string)
        file.flush()
        self.log_manager.print(string, "info")

    def save_options(self, max_ep_len, max_training_timesteps, print_freq, log_freq, save_model_freq,
                     state_dim, action_dim):
        file = self.options_file
        string = ""
        string += "---------------------------------------------------------------------------------------------\n"
        string += f"max training timesteps : {max_training_timesteps}                                           \n"
        string += f"max timesteps per episode : {max_ep_len}                                                    \n"
        string += f"printing average reward over episodes in last : {str(print_freq)} timesteps                 \n"
        string += f"log frequency : {str(log_freq)} timesteps                                                   \n"
        string += f"model saving frequency : {str(save_model_freq)} timesteps                                   \n"
        string += "---------------------------------------------------------------------------------------------\n"
        string += f"state space dimension : {state_dim}                                                         \n"
        string += f"action space dimension : {action_dim}                                                       \n"
        string += "---------------------------------------------------------------------------------------------\n"
        file.write(string)
        file.flush()
        self.log_manager.print(string, "info")

def mkdir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def create_file(file_name):
    return open(file_name, "w+")

def create_write_only_file(file_name):
    return open(file_name, "w")

def open_file(file_name):
    return open(file_name, "r")

def open_file_writable(file_name):
    return open(file_name, "w+")

def concat_path(base_path, next_path):
    return base_path + next_path

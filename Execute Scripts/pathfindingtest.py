from datetime import datetime

from Data.DataManager import DataManager
from Log.LogManager import LogManager
from Units.MixedInputMultipleOutput.PathfindingTrain.Env import PathFindingTrainEnv
from Units.MixedInputMultipleOutput.PathfindingTrain.Agent import PPO
from Utils import PathFinder, CurveMaker
from Utils.TimeCheck import TimeChecker

# SAVE_DIRECTORY_PATH = "/content/drive/MyDrive/RL/Simulation3"
SAVE_DIRECTORY_PATH = ".."
LOADING_SIGNIFICANT_ID = "samples/20231023094748033747 - pathfinding trained"
################################### Training ###################################
def test():
    ####### initialize environment hyperparameters ######
    env_name = "Pathfinding_MIMO"
    significant_id = None

    max_ep_len = 1000                   # max timesteps in one episode
    # max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps
    max_test_episode_count = 10

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
    #####################################################

    ############### Simulation Variables ################

    time_step = 0
    i_episode = 0

    #####################################################

    env = PathFindingTrainEnv(save_directory_path=SAVE_DIRECTORY_PATH + "/Game Data/map/empty_map.txt")
    state_dim = env.observation_space
    action_dim = env.action_space

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    data_manager = DataManager(save_directory_path=SAVE_DIRECTORY_PATH + "/Saves/")

    data_manager.init(env_name, is_loading=True, significant_id=LOADING_SIGNIFICANT_ID)
    last_time_step, last_episode = data_manager.load(ppo_agent)

    log_manager = LogManager(data_manager, data_manager.logs_file)
    data_manager.set_log_manager(log_manager)

    update_timestep, K_epochs, eps_clip, gamma, lr_actor, lr_critic = data_manager.load_hyperparameters()
    max_ep_len, max_training_timesteps, print_freq, log_freq, save_model_freq, state_dim, action_dim = data_manager.load_options()

    log_manager.print("Finish initializing!", "info")
    init_type = "Load"
    log_manager.print(f"Init type : {init_type}, Significant ID : {significant_id}, Env : {env_name}", "info")
    log_manager.print(f"Load information: [Last Episode : {last_episode}, Last Time Step : {last_time_step}]", "info")

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    log_manager.print(f"Started testing at (GMT) : {start_time}", "info")

    time_checker = TimeChecker()

    rewards = []
    total_rewards = []
    # training loop
    while i_episode < max_test_episode_count:

        time_checker.start("reset")

        state = env.reset()

        log_manager.start_test_episode(i_episode, time_step)
        log_manager.debug("reset env")
        log_manager.debug(f"map : {env.map}")
        log_manager.debug(f"trace_map : {env.trace_map}")
        log_manager.debug(f"target_pos : {env.target_pos}")

        log_manager.print(f"Start test episode {i_episode} - {time_step}", "info")

        time_checker.end("reset")
        time_checker.start("test")

        rewards.append([])
        total_rewards.append([])
        for t in range(1, max_ep_len+1):
            log_manager.debug(f"episode : {i_episode}, time_step : {time_step}")

            # select action with policy
            # action, action_logprob, state_val = ppo_agent.select_action(state)
            action, action_logprob, state_val = ppo_agent.select_action(state)
            action = [action[0].item(), action[1].item()]

            log_manager.debug(f"last_agent_pos : {env.agent_pos}, last_agent_angle : {env.agent_angle}")
            log_manager.debug(f"action: {action}")

            state, reward, done, _ = env.step(action)

            if t == max_ep_len:
                reward = -1
                done = True

            log_manager.debug(f"curr_agent_pos : {env.agent_pos}, curr_agent_angle : {env.agent_angle}")
            log_manager.debug(f"reward: {reward}, done: {done}")


            # saving reward and is_terminals
            time_step += 1
            rewards[-1].append(reward)

            # break; if the episode is over
            if done:
                break

        i_episode += 1

        log_manager.print(f"End test episode [{i_episode} - {time_step}] with reward [{sum(rewards[-1])}]", "info")
        time_checker.end("test")

    for i in range(len(rewards)):
        total_rewards[i] = sum(rewards[i])
    average_reward = sum(total_rewards) / len(total_rewards)

    print(f"Average Reward: {average_reward}")
    time_checker.summary()

    # 그래프 그리기
    from matplotlib import pyplot as plt

    x = range(len(total_rewards))
    y = total_rewards

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b', zorder=2)  # 선과 점 그리기
    # plt.scatter(x, y, c='r', marker='o', label='scatter')  # 꼭짓점 표시
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.title('Coordinate List Graph')
    plt.grid(True)
    plt.legend()
    plt.show()

    env.close()

    # print total training time
    string = ""
    string += "=============================================================================================\n"
    end_time = datetime.now().replace(microsecond=0)
    string += f"Started training at (GMT) : {start_time}                                                    \n"
    string += f"Finished training at (GMT) : {end_time}                                                     \n"
    string += f"Total training time  : {end_time - start_time}                                              \n"
    string += "=============================================================================================\n"
    log_manager.print(string, "info")


if __name__ == '__main__':
    test()

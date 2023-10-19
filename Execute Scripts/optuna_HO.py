from datetime import datetime

import optuna
import joblib

from Data.DataManager import DataManager
from Log.LogManager import LogManager
from Units.MixedInputMultipleOutput.Base.Agent import PPO
from Units.MixedInputMultipleOutput.Base.Env import Env


def train():
    pass


def test():
    pass

LOAD_PREVIOUS_DATA = False
significant_id = None

def objective_function(trial):
    ####### initialize environment hyperparameters ######
    env_name = "Pathfinding_MIMO"
    global significant_id

    max_ep_len = 1000  # max timesteps in one episode
    max_training_timesteps = int(1e6)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = trial.suggest_int("update_timestep", 32, 5000, 1)# max_ep_len * 4  # update policy every n timesteps
    K_epochs = trial.suggest_int("epoch", 3, 30, 1)  # update policy for K epochs in one PPO update

    eps_clip = trial.suggest_categorical("eps_clip", [0.1, 0.2, 0.3])  # clip parameter for PPO
    gamma = trial.suggest_float("gamma", 0.8, 0.9997)  # discount factor

    lr_actor = trial.suggest_float("lr_actor", 5e-6, 3e-3, log=True)  # learning rate for actor network
    lr_critic = trial.suggest_float("lr_critic", 5e-6, 3e-3, log=True)  # learning rate for critic network
    #####################################################

    ############### Simulation Variables ################

    time_step = 0
    i_episode = 0

    #####################################################

    env = Env()
    state_dim = env.observation_space
    action_dim = env.action_space

    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    data_manager = DataManager()
    log_manager = None
    if LOAD_PREVIOUS_DATA:
        data_manager.init(env_name, is_loading=True, significant_id="20230908132525106346")
        last_time_step, last_episode = data_manager.load(ppo_agent)
        time_step = last_time_step
        i_episode = last_episode

        log_manager = LogManager(data_manager, data_manager.logs_file)
        data_manager.set_log_manager(log_manager)

        update_timestep, K_epochs, eps_clip, gamma, lr_actor, lr_critic = data_manager.load_hyperparameters()
        max_ep_len, max_training_timesteps, print_freq, log_freq, save_model_freq, state_dim, action_dim = data_manager.load_options()
    else:
        significant_id = data_manager.init(env_name)
        data_manager.initial_save()

        log_manager = LogManager(data_manager, data_manager.logs_file)
        data_manager.set_log_manager(log_manager)

        data_manager.save_hyperparameters(update_timestep, K_epochs, eps_clip, gamma, lr_actor, lr_critic)
        data_manager.save_options(max_ep_len, max_training_timesteps, print_freq, log_freq, save_model_freq,
                                  state_dim, action_dim)

    log_manager.print("Finish initializing!", "info")
    init_type = "Load" if LOAD_PREVIOUS_DATA else "Create"
    log_manager.print(f"Init type : {init_type}, Significant ID : {significant_id}, Env : {env_name}", "info")

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    log_manager.print(f"Started training at (GMT) : {start_time}", "info")

    # printing and logging variables
    print_avg_reward = 0
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    # training loop
    while time_step <= max_training_timesteps:

        state = env.reset()
        current_ep_reward = 0

        log_manager.start_episode(i_episode, time_step)
        log_manager.debug("reset env")
        log_manager.debug(f"map : {env.map}")
        log_manager.debug(f"trace_map : {env.trace_map}")
        log_manager.debug(f"target_pos : {env.target_pos}")

        for t in range(1, max_ep_len + 1):
            log_manager.debug(f"episode : {i_episode}, time_step : {time_step}")

            # select action with policy
            old_state = state
            action = ppo_agent.select_action(state)

            log_manager.debug(f"last_agent_pos : {env.agent_pos}, last_agent_angle : {env.agent_angle}")
            log_manager.debug(f"action: {action}")

            state, reward, done, _ = env.step(action)

            log_manager.debug(f"curr_agent_pos : {env.agent_pos}, curr_agent_angle : {env.agent_angle}")
            log_manager.debug(f"reward: {reward}, done: {done}")

            # log_manager.print(f"old state: {old_state}", "debug")
            # log_manager.print(f"new state: {state}", "debug")

            if t == max_ep_len:
                reward = -1
                done = True

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                # log_manager.print(f"{i_episode}, {time_step}, {log_avg_reward}", "info")

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                # print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                log_manager.print(
                    f"Episode : {i_episode} \t\t Timestep : {time_step} \t\t Average Reward : {print_avg_reward}",
                    "info")

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                data_manager.checkout(i_episode, time_step, ppo_agent,
                                      elapsed_time=datetime.now().replace(microsecond=0) - start_time)

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

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

    return -print_avg_reward


sampler = optuna.samplers.TPESampler()
study = optuna.create_study(sampler=sampler, direction="maximize")

study.optimize(objective_function, n_trials=20)
print(study.best_params)
joblib.dump(study, f"/Optuna/HO/MIMO/{significant_id}/mnist_optuna.pkl")

study = joblib.load(f"/Optuna/HO/MIMO/{significant_id}/mnist_optuna.pkl")
df = study.trials_dataframe().drop(['state', 'datetime_start', 'datetime_complete', 'system_attrs'], axis=1)
df.head(3)

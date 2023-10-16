from Units.MixedInputMultipleOutput.Env import Env
from Units.MixedInputMultipleOutput.Agent import PPO


LOAD_PREVIOUS_DATA = False

################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

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

    env = Env()

    # state space dimension
    state_dim = env.observation_space.shape

    # action space dimension
    action_dim = env.action_space.shape

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

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

            # break; if the episode is over
            if done:
                break

        i_episode += 1

    env.close()


if __name__ == '__main__':
    train()

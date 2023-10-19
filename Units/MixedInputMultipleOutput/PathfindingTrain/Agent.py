import functorch.dim
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from tqdm import tqdm

################################## set device ##################################

print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    torch.set_num_threads(torch.get_num_threads())
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

class RolloutBuffer:
    def __init__(self):
        self.move_actions = []
        self.rotate_actions = []
        self.move_states = []
        self.rotate_states = []
        self.move_logprobs = []
        self.rotate_logprobs = []
        self.rewards = []
        self.move_state_values = []
        self.rotate_state_values = []
        self.is_terminals = []

    def clear(self):
        del self.move_actions[:]
        del self.rotate_actions[:]
        del self.move_states[:]
        del self.rotate_states[:]
        del self.move_logprobs[:]
        del self.rotate_logprobs[:]
        del self.rewards[:]
        del self.move_state_values[:]
        del self.rotate_state_values[:]
        del self.is_terminals[:]

    def extend(self, buffer):
        self.move_actions.extend(buffer.actions)
        self.rotate_actions.extend(buffer.actions)
        self.move_states.extend(buffer.states)
        self.rotate_states.extend(buffer.states)
        self.move_logprobs.extend(buffer.logprobs)
        self.rotate_logprobs.extend(buffer.logprobs)
        self.rewards.extend(buffer.rewards)
        self.move_state_values.extend(buffer.state_values)
        self.rotate_state_values.extend(buffer.state_values)
        self.is_terminals.extend(buffer.is_terminals)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.float_layer = nn.Linear(state_dim[0], 32)

        self.map_layer1 = nn.Conv2d(state_dim[1][0], 8, kernel_size=3, stride=1, padding=1)
        self.map_layer2 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.fc_map = nn.Linear(4356, 32)

        self.fc_combined = nn.Linear(64, 64)

        self.fc_move_or_not = nn.Linear(64, action_dim[0])
        self.fc_rotate = nn.Linear(64, action_dim[1])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x1 = torch.relu(self.float_layer(state[0]))

        x2 = torch.relu(self.map_layer1(state[1]))
        x2 = torch.relu(self.map_layer2(x2))
        x2 = x2.view(x2.size(0), -1)
        # prepared: 16384, real: 69696
        # print(x2.shape)
        x2 = torch.relu(self.fc_map(x2))

        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(0)
        x = torch.cat((x1, x2), dim=1)
        x = torch.relu(self.fc_combined(x))

        out_move_or_not_logits = self.fc_move_or_not(x)
        out_move_or_not = self.softmax(out_move_or_not_logits)

        out_rotate_logits = self.fc_rotate(x)
        out_rotate_or_not = self.softmax(out_rotate_logits)

        # return out_move_or_not, out_rotate
        return out_move_or_not, out_rotate_or_not

class Critic(nn.Module):

    def __init__(self, state_dim):
        super(Critic, self).__init__()

        self.float_layer = nn.Linear(state_dim[0], 32)

        self.map_layer1 = nn.Conv2d(state_dim[1][0], 8, kernel_size=3, stride=1, padding=1)
        self.map_layer2 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.fc_map = nn.Linear(4356, 32)

        self.fc_combined = nn.Linear(64, 64)

        self.fc_move_or_not = nn.Linear(64, 1)
        self.fc_rotate = nn.Linear(64, 1)

    def forward(self, state):
        x1 = torch.relu(self.float_layer(state[0]))

        x2 = torch.relu(self.map_layer1(state[1]))
        x2 = torch.relu(self.map_layer2(x2))
        x2 = x2.view(x2.size(0), -1)
        x2 = torch.relu(self.fc_map(x2))

        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(0)
        x = torch.cat((x1, x2), dim=1)
        x = torch.relu(self.fc_combined(x))

        out_move_or_not = self.fc_move_or_not(x)

        out_rotate_or_not = self.fc_rotate(x)

        return out_move_or_not, out_rotate_or_not

class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        move_action_probs, rotate_action_probs = self.actor(state)

        move_dist = Categorical(move_action_probs)
        move_action = move_dist.sample()
        move_action_logprob = move_dist.log_prob(move_action)

        rotate_dist = Categorical(rotate_action_probs)
        rotate_action = rotate_dist.sample()
        rotate_action_logprob = rotate_dist.log_prob(rotate_action)

        move_state_val, rotate_state_val = self.critic(state)

        return [move_action.detach(), rotate_action.detach()], \
            [move_action_logprob.detach(), rotate_action_logprob.detach()], \
            [move_state_val.detach(), rotate_state_val.detach()]

    def evaluate(self, state, action):
        move_action_probs, rotate_action_probs = self.actor(state)

        move_dist = Categorical(move_action_probs)
        move_action_logprobs = move_dist.log_prob(action[0])
        move_dist_entropy = move_dist.entropy()

        rotate_dist = Categorical(rotate_action_probs)
        rotate_action_logprobs = rotate_dist.log_prob(action[1])
        rotate_dist_entropy = rotate_dist.entropy()

        move_state_values, rotate_state_values = self.critic(state)

        return [move_action_logprobs, rotate_action_logprobs], \
            [move_state_values, rotate_state_values], \
            [move_dist_entropy, rotate_dist_entropy]

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.state_dim = state_dim

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        # print(state)
        with torch.no_grad():
            state[0] = torch.FloatTensor(state[0]).to(device)
            state[1] = torch.FloatTensor(state[1]).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.move_states.append(state[0])
        self.buffer.move_actions.append(action[0])
        self.buffer.move_logprobs.append(action_logprob[0])
        self.buffer.move_state_values.append(state_val[0])

        self.buffer.rotate_states.append(state[1])
        self.buffer.rotate_actions.append(action[1])
        self.buffer.rotate_logprobs.append(action_logprob[1])
        self.buffer.rotate_state_values.append(state_val[1])

        return [action[0].item(), action[1].item()]

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_move_states = torch.squeeze(torch.stack(self.buffer.move_states, dim=0)).detach().to(device)
        old_move_actions = torch.squeeze(torch.stack(self.buffer.move_actions, dim=0)).detach().to(device)
        old_move_logprobs = torch.squeeze(torch.stack(self.buffer.move_logprobs, dim=0)).detach().to(device)
        old_move_state_values = torch.squeeze(torch.stack(self.buffer.move_state_values, dim=0)).detach().to(device)

        old_rotate_states = torch.squeeze(torch.stack(self.buffer.rotate_states, dim=0)).detach().to(device)
        old_rotate_actions = torch.squeeze(torch.stack(self.buffer.rotate_actions, dim=0)).detach().to(device)
        old_rotate_logprobs = torch.squeeze(torch.stack(self.buffer.rotate_logprobs, dim=0)).detach().to(device)
        old_rotate_state_values = torch.squeeze(torch.stack(self.buffer.rotate_state_values, dim=0)).detach().to(device)

        # calculate advantages
        move_advantages = rewards.detach() - old_move_state_values.detach()
        rotate_advantages = rewards.detach() - old_rotate_state_values.detach()

        # Optimize policy for K epochs
        for _ in tqdm(range(self.K_epochs)):
            # Evaluating old actions and values

            logprobs, state_values, dist_entropy = self.policy.evaluate([old_move_states, old_rotate_states], [old_move_actions, old_rotate_actions])

            # match state_values tensor dimensions with rewards tensor
            state_values[0] = torch.squeeze(state_values[0])
            state_values[1] = torch.squeeze(state_values[1])

            # Finding the ratio (pi_theta / pi_theta__old)
            move_ratios = torch.exp(logprobs[0] - old_move_logprobs.detach())
            rotate_ratios = torch.exp(logprobs[1] - old_rotate_logprobs.detach())

            # Finding Surrogate Loss
            move_surr1 = move_ratios * move_advantages
            move_surr2 = torch.clamp(move_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * move_advantages

            rotate_surr1 = rotate_ratios * rotate_advantages
            rotate_surr2 = torch.clamp(rotate_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * rotate_advantages

            # final loss of clipped objective PPO
            loss = -torch.min(move_surr1, move_surr2) + 0.5 * self.MseLoss(state_values[0], rewards) - 0.01 * dist_entropy[0]
            loss += -torch.min(rotate_surr1, rotate_surr2) + 0.5 * self.MseLoss(state_values[1], rewards) - 0.01 * dist_entropy[1]

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


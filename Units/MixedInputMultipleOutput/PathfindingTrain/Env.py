import copy
import math
import random
import numpy as np

from Utils import PathFinder, CurveMaker
from Utils.ATF import angled_cos, angled_sin


class PathFindingTrainEnv:
    def __init__(self, save_directory_path="../Game Data/map/empty_map.txt"):
        self.map, self.width, self.height = load_map(save_directory_path)

        self.move_speed = 0.1
        self.rotate_angle = 9
        self.agent_init_pos = [16, 16]
        self.agent_init_angle = 0

        self.agent_pos = self.agent_init_pos
        self.agent_angle = self.agent_init_angle
        self.target_pos = [16, 16]

        self.observation_space = (6, (4, self.width, self.height))
        self.action_space = (2, 3)

        self.least_propagation_value = 0.1
        self.trace_propagation_decay_value = 0.966
        self.trace_map = None
        self.path_data = None
        self.path_train_data = None

    def reset(self):
        self.initialize_agent_pos()
        self.randomize_target_pos()

        self.make_trace_map()

        # [ right, up, left, down ]
        return self.get_state(self.agent_pos, self.target_pos)

    def make_trace_map(self):
        def propagation(pos, value):
            if value < 0.1:
                return

            for action in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                new_pos = copy.deepcopy(pos)
                new_pos[0] += action[0]
                new_pos[1] += action[1]

                if not self.is_unable_pos(new_pos) and self.trace_map[new_pos[1]][new_pos[0]] < value:
                    self.trace_map[new_pos[1]][new_pos[0]] = value
                    propagation(new_pos, value * self.trace_propagation_decay_value)

        self.trace_map = [[0 for _ in range(self.width)] for _ in range(self.height)]
        propagation(self.target_pos, 1)
        # print(np.array(self.map))
        # print(np.array(self.trace_map))

    def make_path_train_data(self):
        points = PathFinder.find_path(self.map, self.agent_pos, self.target_pos)
        reshaped_points = CurveMaker.reshape_points(points, self.move_speed)
        reconstructed_data = CurveMaker.reconstruct_points(reshaped_points, self.agent_init_angle, height=self.height)
        path_data = self.pack_data(reconstructed_data)
        train_data = self.build_train_data(path_data)

        self.path_data = path_data
        self.path_train_data = train_data

    def pack_data(self, original_data):
        reconstructed_data = []
        for data in original_data:
            reward, done = self.get_reward(data[0], data[1])
            data[4] = reward
            data[5] = done
            reconstructed_data.append(data)
            if done:
                break
        return reconstructed_data

    def build_train_data(self, original_data):
        train_data = []
        for data in original_data:
            curr_point, next_point, angle, accumulated_angle, reward, done = data

            curr_state = self.get_state(curr_point, self.target_pos)
            # next_state = self.get_state(next_point, self.target_pos)
            # state = [curr_state, next_state]
            state = curr_state

            move_action = 1             # always move forward
            rotate_action = -1
            if angle < 0:
                rotate_action = 0
            elif angle > 0:
                rotate_action = 2
            elif angle == 0:
                rotate_action = 1
            action = [move_action, rotate_action]

            train_data.append((state, action, reward, done))
        return train_data

    def initialize_agent_pos(self):
        self.agent_pos = self.agent_init_pos
        self.agent_angle = self.agent_init_angle

    def randomize_target_pos(self):
        self.target_pos = [random.randint(0, self.width), random.randint(0, self.height)]
        while self.target_pos == self.agent_pos or self.is_unable_pos(self.target_pos):
            self.target_pos = [random.randint(0, self.width), random.randint(0, self.height)]

    def step(self, action):
        state = None
        reward = 0
        done = False

        action_vector = [-self.rotate_angle, 0, self.rotate_angle]
        after_agent_pos = self.agent_pos.copy()
        after_agent_angle = self.agent_angle + action_vector[action[1]]
        radians = math.radians(after_agent_angle)
        direction = [
            math.cos(radians) * action[0],
            math.sin(radians) * action[0]
        ]
        after_agent_pos[0] += self.move_speed * direction[0]
        after_agent_pos[1] += self.move_speed * direction[1]

        reward, done = self.get_reward(self.agent_pos, after_agent_pos)

        if not self.is_unable_pos(after_agent_pos):
            self.agent_pos = after_agent_pos
            self.agent_angle = after_agent_angle

        state = self.get_state(self.agent_pos, self.target_pos)

        # return state, reward, done, _
        return state, reward, done, None

    def get_state(self, agent_pos, target_pos):
        channel = []
        channel.append(copy.deepcopy(self.map))

        agent_pos_onehot = [[0 for x in range(self.width)] for y in range(self.height)]
        agent_pos_onehot[int(agent_pos[1])][int(agent_pos[0])] = 1

        target_pos_onehot = [[0 for x in range(self.width)] for y in range(self.height)]
        target_pos_onehot[target_pos[1]][target_pos[0]] = 1

        channel.append(agent_pos_onehot)
        channel.append(target_pos_onehot)
        # channel.append([])
        channel.append(self.trace_map)

        map_state = np.reshape(channel, (1, len(channel), self.width, self.height))

        dx_float = (target_pos[0] - agent_pos[0]) / self.width
        dy_float = (target_pos[1] - agent_pos[1]) / self.height
        px_float = (agent_pos[0] - int(agent_pos[0])) # / self.width
        py_float = (agent_pos[1] - int(agent_pos[1])) # / self.height
        cos = math.cos(math.radians(self.agent_angle))
        sin = math.sin(math.radians(self.agent_angle))

        float_state = [dx_float, dy_float, px_float, py_float, cos, sin]
        float_state = np.reshape(float_state, (6,))

        state = [float_state, map_state]

        # print(state)
        return state

    def get_reward(self, current_pos, new_pos):
        done = False

        old_distance = get_distance(current_pos, self.target_pos)
        new_distance = get_distance(new_pos, self.target_pos)
        reward = old_distance - new_distance
        # reward = -0.1
        if self.target_pos[0] == int(new_pos[0]) and self.target_pos[1] == int(new_pos[1]):
            reward = 1
            done = True

        return reward, done

    def render(self):
        print("=" * self.width)
        render_map = copy.deepcopy(self.map)
        render_map[int(self.agent_pos[1])][int(self.agent_pos[0])] = 2
        render_map[self.target_pos[1]][self.target_pos[0]] = 3
        for y in range(self.height-1):
            line = ""
            for x in range(self.width):
                line += str(render_map[y][x])
            print(line)

    def close(self):
        pass

    def is_unable_pos(self, pos):
        if not (0 <= pos[0] < self.width and 0 <= pos[1] < self.height) or self.map[int(pos[1])][int(pos[0])] > 0.9:
            return True
        return False

def load_map(save_directory_path="../Game Data/map/empty_map.txt"):
    cols = []
    file_path = save_directory_path
    with open(file_path, 'r') as file:
        for line in file:
            rows = []
            for x in line:
                if '\n' in x or '\r' in x:
                    continue
                rows.append(float(x))
            cols.append(rows)

    if len(cols) <= 0 and len(cols[0]) <= 0:
        print("error occur at loading map. . .")

    return cols, len(cols[0]), len(cols)

def get_distance(a, b):
    return math.sqrt((b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1]))

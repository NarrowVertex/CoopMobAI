import copy
import math
import random
import numpy as np

from Utils.ATF import angled_cos, angled_sin


class PathFindingTrainEnv:
    def __init__(self, save_directory_path="../Game Data/map/empty_map.txt"):
        self.map, self.width, self.height = load_map(save_directory_path)

        self.agent_pos = [16, 16]
        self.agent_angle = 0
        self.target_pos = [16, 16]

        self.observation_space = (6, (4, self.width, self.height))
        self.action_space = (2, 3)

        self.least_propagation_value = 0.1
        self.trace_propagation_decay_value = 0.966
        self.trace_map = None

    def reset(self):
        self.initialize_agent_pos()
        self.randomize_target_pos()

        self.make_trace_map()

        # [ right, up, left, down ]
        return self.get_state()

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

    def initialize_agent_pos(self):
        self.agent_pos = [16, 16]

    def randomize_target_pos(self):
        self.target_pos = [random.randint(0, self.width), random.randint(0, self.height)]
        while self.target_pos == self.agent_pos or self.is_unable_pos(self.target_pos):
            self.target_pos = [random.randint(0, self.width), random.randint(0, self.height)]

    def step(self, action):
        state = None
        reward = 0
        done = False

        move_speed = 0.1

        action_vector = [-9, 0, 9]
        after_agent_pos = self.agent_pos.copy()
        after_agent_angle = self.agent_angle + action_vector[action[1]]
        radians = math.radians(after_agent_angle)
        direction = [
            math.cos(radians) * action[0],
            math.sin(radians) * action[0]
        ]
        after_agent_pos[0] += move_speed * direction[0]
        after_agent_pos[1] += move_speed * direction[1]

        old_distance = get_distance(self.agent_pos, self.target_pos)
        new_distance = get_distance(after_agent_pos, self.target_pos)
        reward = old_distance - new_distance
        # reward = -0.1
        if self.target_pos[0] == int(after_agent_pos[0]) and self.target_pos[1] == int(after_agent_pos[1]):
            reward = 1
            done = True

        if not self.is_unable_pos(after_agent_pos):
            self.agent_pos = after_agent_pos
            self.agent_angle = after_agent_angle

        state = self.get_state()

        # return state, reward, done, _
        return state, reward, done, None

    def get_state(self):
        channel = []
        channel.append(copy.deepcopy(self.map))

        agent_pos_onehot = [[0 for x in range(self.width)] for y in range(self.height)]
        agent_pos_onehot[int(self.agent_pos[1])][int(self.agent_pos[0])] = 1

        target_pos_onehot = [[0 for x in range(self.width)] for y in range(self.height)]
        target_pos_onehot[self.target_pos[1]][self.target_pos[0]] = 1

        channel.append(agent_pos_onehot)
        channel.append(target_pos_onehot)
        # channel.append([])
        channel.append(self.trace_map)

        map_state = np.reshape(channel, (1, len(channel), self.width, self.height))

        dx_float = (self.target_pos[0] - self.agent_pos[0]) / self.width
        dy_float = (self.target_pos[1] - self.agent_pos[1]) / self.height
        px_float = (self.agent_pos[0] - int(self.agent_pos[0])) # / self.width
        py_float = (self.agent_pos[1] - int(self.agent_pos[1])) # / self.height
        cos = math.cos(math.radians(self.agent_angle))
        sin = math.sin(math.radians(self.agent_angle))

        float_state = [dx_float, dy_float, px_float, py_float, cos, sin]
        float_state = np.reshape(float_state, (6,))

        state = [float_state, map_state]

        # print(state)
        return state

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

import copy
import math
from math import cos, sin, radians, atan2, degrees
import matplotlib.pyplot as plt

origin_list = [(10, -9), (11, -8), (10, -7), (9, -6), (8, -5), (7, -4), (6, -3), (5, -2), (4, -1), (3, -1), (2, -1), (1, -1)]

angle = 9
length = 0.1

def check_curve_length(unit_angle, interval):
    unit_angle = radians(unit_angle)
    width = 0
    height = 0
    for i in range(9):
        width += cos((i+1) * unit_angle)
        height += sin((i+1) * unit_angle)
    width *= interval
    height *= interval
    # print(width, height)

def reshape_points(points, interval):
    # 새로운 좌표 리스트 초기화
    new_points = []

    # 주어진 좌표 리스트 순회
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        # 시작점 추가
        new_points.append((x1, y1))

        # 시작점과 끝점 사이의 점 추가 (0.1 간격)
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        num_intervals = int(distance / interval)

        if num_intervals > 0:
            dx = (x2 - x1) / num_intervals
            dy = (y2 - y1) / num_intervals
            for j in range(1, num_intervals):
                new_x = x1 + j * dx
                new_y = y1 + j * dy
                new_points.append((new_x, new_y))

    # 마지막 점 추가
    new_points.append(points[-1])

    return new_points

def check_curve(points):
    selected_points = []

    # 좌표 리스트 순회
    for i in range(1, len(points) - 1):
        x_prev, y_prev = points[i - 1]
        x_current, y_current = points[i]
        x_next, y_next = points[i + 1]

        # 현재 좌표와 앞 좌표, 현재 좌표와 뒤 좌표를 이용하여 각도 계산
        angle1 = degrees(atan2(y_current - y_prev, x_current - x_prev))
        angle2 = degrees(atan2(y_next - y_current, x_next - x_current))

        diff = abs(angle2 - angle1)
        # 각도가 0도 이상인 경우 현재 좌표를 결과 리스트에 추가
        if diff > 0:
            base_angle = angle1
            is_positive = True if angle2 > angle1 else False
            selected_points.append(((x_current, y_current), (diff, base_angle, is_positive)))

    return selected_points

def create_curve(points, selected_points, unit_angle, interval):
    curved_points = copy.deepcopy(points)
    selected_indexes = []
    for i in range(len(points)):
        point = points[i]
        for selected_point in selected_points:
            # print(point[0], selected_point[0][0], point[1], selected_point[0][1], point[0] == selected_point[0][0], point[1] == selected_point[0][1])
            if point[0] == selected_point[0][0] and point[1] == selected_point[0][1]:
                selected_indexes.append((i, selected_point[1]))

    for selected_index in selected_indexes:
        diff, base_angle, is_positive = selected_index[1]
        curve_point_number = int(diff / unit_angle)
        # print(f"Detected point is {points[selected_index[0]]} with index {selected_index[0]}")
        # print(f"Need {curve_point_number} points for curve degree {selected_index[1]} with unit angle {unit_angle}")

        start_index = int(selected_index[0] - (curve_point_number - 1) / 2)
        # reshaped_points = []
        last_reshaped_point = points[start_index] # interval * (cos(radians(unit_angle)), sin(radians(unit_angle)))
        sign = (1 if is_positive else -1)
        for i in range(1, curve_point_number+1):
            point = [
                interval * cos(radians(base_angle + sign * unit_angle * i)),
                interval * sin(radians(base_angle + sign * unit_angle * i))
            ]
            point[0] += last_reshaped_point[0]
            point[1] += last_reshaped_point[1]
            # reshaped_points.append(point)
            curved_points[start_index + i] = point
            last_reshaped_point = point

    return curved_points


import time
start_time = time.time()
print(f"Start time : {start_time}")

check_curve_length(angle, length)
reshaped_list = reshape_points(origin_list, length)
selected_list = check_curve(origin_list)
curved_list = create_curve(reshaped_list, selected_list, angle, length)

end_time = time.time()
print(f"End tiem : {end_time}")

elapsed_time = end_time - start_time
print(f"Spent time : {elapsed_time}s")


selected_points = [x[0] for x in selected_list]

# 그래프 그리기
origin_x, origin_y = zip(*origin_list)
reshaped_x, reshaped_y = zip(*reshaped_list)
selected_x, selected_y = zip(*selected_points)
curved_x, curved_y = zip(*curved_list)

fig, axis = plt.subplots(2, 2, figsize=(10, 8))
((ax1, ax2), (ax3, ax4)) = axis

ax1.plot(origin_x, origin_y, marker='o', linestyle='-', color='b', label='origin', zorder=1)  # 선과 점 그리기
ax1.scatter(selected_x, selected_y, c='r', marker='o', label='selected', zorder=2)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Original')
ax1.grid(True)
ax1.legend()

ax2.plot(reshaped_x, reshaped_y, marker='o', linestyle='-', color='r', label='reshaped')  # 선과 점 그리기
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Reshaped')
ax2.grid(True)
ax2.legend()

ax3.plot(curved_x, curved_y, marker='o', linestyle='-', color='r', label='curved')  # 선과 점 그리기
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title('Curved')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()

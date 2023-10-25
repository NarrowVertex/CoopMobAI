import copy
import math
from math import cos, sin, radians, atan2, degrees

def reshape_points(points, interval):
    if len(points) == 0:
        print("points is empty!!!")

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

def reconstruct_points(points, init_angle, height=0):
    data = []
    last_angle = init_angle
    accumulated_angle = init_angle
    for i in range(len(points) - 1):
        """
        curr_point = [points[i][0], points[i][1]]
        next_point = [points[i + 1][0], points[i + 1][1]]
        """
        original_curr_point = points[i]
        original_next_point = points[i + 1]

        # convert to calculate angles
        curr_point = None
        next_point = None
        if height != 0:
            curr_point = [points[i][0], (height-1) - points[i][1]]
            next_point = [points[i + 1][0], (height-1) - points[i + 1][1]]
        else:
            curr_point = [points[i][0], points[i][1]]
            next_point = [points[i + 1][0], points[i + 1][1]]

        angle1 = last_angle
        angle2 = round(get_angle(curr_point, next_point))
        angle = int(angle2 - angle1)
        accumulated_angle += angle

        data.append([original_curr_point, original_next_point, angle, accumulated_angle, 0, False])
        last_angle = angle2
        # print(f"{original_curr_point}  {original_next_point}  {angle1} {angle2} {angle}")
    return data

def get_angle(pos1, pos2):
    return degrees(- atan2(pos2[1] - pos1[1], pos2[0] - pos1[0]))

def check_curve(points):
    selected_points = []

    # 좌표 리스트 순회
    for i in range(1, len(points) - 1):
        x_prev, y_prev = points[i - 1]
        x_current, y_current = points[i]
        x_next, y_next = points[i + 1]

        # 현재 좌표와 앞 좌표, 현재 좌표와 뒤 좌표를 이용하여 각도 계산
        angle1 = get_angle((x_prev, y_prev), (x_current, y_current))
        angle2 = get_angle((x_current, y_current), (x_next, y_next))

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
        curve_point_number = int(diff / unit_angle) - 1
        # print(f"Detected point is {points[selected_index[0]]} with index {selected_index[0]}")
        # print(f"Need {curve_point_number} points for curve degree {selected_index[1]} with unit angle {unit_angle}")

        # selected_index[0]와 curved_point_number / 2의 위치 비교 해서 start_index 잘 조정
        start_index = int(selected_index[0] - (curve_point_number) / 2) + 1
        del curved_points[start_index:start_index + curve_point_number]
        # reshaped_points = []
        last_reshaped_point = points[start_index - 1]# interval * (cos(radians(unit_angle)), sin(radians(unit_angle)))
        sign = (1 if is_positive else -1)
        for i in range(curve_point_number):
            point = [
                interval * cos(radians(base_angle + sign * unit_angle * (i + 1))),
                interval * sin(radians(base_angle + sign * unit_angle * (i + 1)))
            ]
            point[0] += last_reshaped_point[0]
            point[1] += last_reshaped_point[1]
            # reshaped_points.append(point)
            curved_points.insert(start_index + i, point)
            last_reshaped_point = point

    return curved_points

def make_curve(original_list, unit_angle, interval):
    reshaped_list = reshape_points(original_list, interval)
    selected_list = check_curve(original_list)
    curved_list = create_curve(reshaped_list, selected_list, unit_angle, interval)
    return curved_list

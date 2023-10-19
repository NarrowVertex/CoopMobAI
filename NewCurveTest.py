import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def divide_line(points, interval):
    divided_points = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        divided_points.append((x1, y1))

        # Calculate the number of segments to divide between (x1, y1) and (x2, y2)
        num_segments = int((x2 - x1) / interval)

        # Calculate the step size for x and y
        x_step = (x2 - x1) / num_segments
        y_step = (y2 - y1) / num_segments

        # Generate the divided points between (x1, y1) and (x2, y2)
        for j in range(1, num_segments):
            new_x = x1 + j * x_step
            new_y = y1 + j * y_step
            divided_points.append((new_x, new_y))

    # Add the last point from the original list
    divided_points.append(points[-1])

    return divided_points

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between line segments formed by points p1-p2 and p2-p3.
    """
    angle_rad = math.atan2(p3[1] - p2[1], p3[0] - p2[0]) - math.atan2(p1[1] - p2[1], p1[0] - p2[0])
    angle_deg = math.degrees(angle_rad)
    return angle_deg

def find_points_with_positive_angle(points):
    result = []

    for i in range(1, len(points) - 1):
        angle = calculate_angle(points[i - 1], points[i], points[i + 1])
        if angle > 0:
            result.append(points[i])

    return result

def smooth_points_with_cubic_spline(points, num_points_on_curve):
    # Extract x and y coordinates from the original points
    x = [point[0] for point in points]
    y = [point[1] for point in points]

    # Create a parameter t based on the cumulative distance between points
    t = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    t = np.insert(t, 0, 0)  # Add t=0 at the beginning
    t /= t[-1]  # Normalize t to [0, 1]

    # Create a Cubic Spline curve for x and y coordinates
    cs_x = CubicSpline(t, x, bc_type='natural')
    cs_y = CubicSpline(t, y, bc_type='natural')

    # Evaluate the curve at evenly spaced points
    t_new = np.linspace(0, 1, num_points_on_curve)
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)

    # Create a new list of points from the smoothed x and y coordinates
    smoothed_points = [(x, y) for x, y in zip(x_new, y_new)]

    return smoothed_points


# Example usage with interval 0.1
points = [(0, 0), (1, 0), (2, -1), (3, -1), (4, -1), (5, -2), (6, -3), (7, -4), (8, -5), (9, -6), (10, -7), (11, -8), (10, -9)]
interval = 0.1  # Updated interval
divided_points = divide_line(points, interval)
print(divided_points)

selected_points = find_points_with_positive_angle(points)
print(selected_points)

# Example usage with 20 points on the smoothed curve
num_points_on_curve = 1000
smoothed_points = smooth_points_with_cubic_spline(divided_points, num_points_on_curve)

# Plot the original and smoothed curves for visualization
x_original = [point[0] for point in points]
y_original = [point[1] for point in points]

x_smoothed = [point[0] for point in smoothed_points]
y_smoothed = [point[1] for point in smoothed_points]

plt.figure(figsize=(8, 4))
plt.plot(x_original, y_original, 'o-', label='Original Points')
plt.plot(x_smoothed, y_smoothed, 'r-', label='Smoothed Curve')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Points and Smoothed Curve')
plt.grid(True)
plt.show()

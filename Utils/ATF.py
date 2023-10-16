from math import pi

def func1(x):
    return 4 / pi * x

def func2(x):
    return 1

def func3(x):
    return -4 / pi * x + 4

def func4(x):
    if 0 <= x <= pi / 4:
        return func1(x)
    elif pi / 4 < x <= 3 * pi / 4:
        return func2(x)
    elif 3 * pi / 4 < x <= pi:
        return func3(x)
    else:
        return 0

def func5(x):
    return func4(x) - func4(x - pi)

def angled_sin(x):
    mod_x = x % (2*pi)
    return func5(mod_x)

def angled_cos(x):
    mod_x = (x + pi / 2) % (2 * pi)
    return func5(mod_x)

"""
import math
import matplotlib.pyplot as plt
import numpy as np


# Generate data
x = np.arange(0, 100.01, 0.01) # 400 points between -10 and 10
y_linear = []
y_quadratic = []
for i in x:
    y_linear.append(math.cos(i))
    y_quadratic.append(angled_cos(i))

y_linear = np.array(y_linear)
y_quadratic = np.array(y_quadratic)

# Create the figure and axes
fig, ax = plt.subplots()

# Plot data
ax.plot(x, y_linear, label='Linear: y = x', color='blue')
ax.plot(x, y_quadratic, label='Quadratic: y = x^2', color='red')

# Set title and labels
ax.set_title('Linear vs Quadratic Functions')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()  # display the legend

# Display the plot
plt.grid(True)
plt.show()
"""

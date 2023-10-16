import re
import matplotlib.pyplot as plt

log_file = "../Saves/Pathfinding_MIMO/20230908132525106346/logs.csv"

logs = []
with open(log_file, 'r') as file:
    for line in file:
        logs.append(line.strip())
    file.close()

# Define a regular expression pattern to match the reward value
pattern = r"reward: ([\-0-9\.eE]+)"

# Iterate through the log entries and extract reward values
reward_values = []
for log_entry in logs:
    match = re.search(pattern, log_entry)
    if match:
        reward_values.append(float(match.group(1)))

# Create x-axis values (assuming rewards are sequential)
x_values = range(1, len(reward_values) + 1)

# Create a line plot
plt.plot(x_values, reward_values, marker='o', linestyle='-', color='b', label='Rewards')

# Set labels for the axes and a title
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Progression')

# Show a legend (if you have multiple lines)
plt.legend()

# Display the plot
plt.show()

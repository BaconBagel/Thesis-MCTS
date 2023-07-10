import numpy as np
import pandas as pd
import time
import xgboost as xgb
from xgboost import DMatrix
import math
import numpy as np
import random
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import jit
from matplotlib.collections import LineCollection
from collections import Counter
from matplotlib import rc
import statsmodels.api as sm
import networkx as nx
rc('animation', html='jshtml')

# Define the state space and action space
state_space = 1000 * 100 * 100
action_space = 5
position_history = []
reward_history = []
edges = {}


# Define the number of simulations
num_simulations = 200000

c_values = np.load("c_values_train.npy")


random_rewards = np.random.rand(1000, 10000)  # can be ued for testing
rewards = c_values
sum_of_max_values = np.sum(np.amax(c_values, axis=1))
print("Theoretical maximum of a drone travelling a c", sum_of_max_values)
# Calculate the sum of each column
column_sums = np.sum(c_values, axis=0)

# Find the index of the column with the highest sum
max_sum_index = np.argmax(column_sums)

# Sort the column sums in descending order and find the second highest index
sorted_column_sums = np.argsort(column_sums)[::-1]
second_max_sum_index = sorted_column_sums[1]
print(sorted_column_sums)
# Print the result
print("Index with highest sum:", max_sum_index)
print("Second highest index:", second_max_sum_index)
# Define the exploration constant


c = 0.1
path = []

def preprocess_state_action(state, action):
    t, x, y = state

    # Encode the state features
    state_features = [t, x, y]  # Assuming the state features are integers

    # Encode the action feature
    action_feature = [0] * 4  # Assuming there are 4 possible actions
    action_feature[action] = 1

    # Concatenate the state and action features
    features = state_features + action_feature

    # Convert the features to a NumPy array
    features_array = np.array(features)

    # Reshape the array to match the expected input shape of the XGBoost model
    features_array = features_array.reshape(1, -1)

    return features_array


# Define the transition function

def transition(state, action):
    # Unpack the state
    t, x, y = state

    # Compute the next position of the character based on the action
    if action == 0:  # up
        x = max(x - 1, 0)
    elif action == 1:  # down
        x = min(x + 1, 99)
    elif action == 2:  # left
        y = max(y - 1, 0)
    elif action == 3:  # right
        y = min(y + 1, 99)


    if t == 999:
        next_t = 0
    else:
        next_t = min(t + 1, 999)

    next_state = (next_t, x, y)

    # Compute the reward
    rand_mult = random.random()
    reward_base = rewards[next_t, x * 100 + y] + (0*rand_mult*rewards[next_t, x * 100 + y])

    # Compute neighbour rewards

    reward_horizon = 0
    speed = 5

    total_reward = 0

    if reward_horizon > 0:
        for dx in range(-reward_horizon, reward_horizon + 1):
            for dy in range(-reward_horizon, reward_horizon + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip the center cell

                # Check if neighbour cell is within grid
                if 0 <= x + dx < 100 and 0 <= y + dy < 100:
                    random_number = 1 + (random.random() * 0)  # Random value between 0 and 1

                    # Calculate reward for this neighbour cell
                    reward = random_number * max(
                        rewards[next_t, (x + dx) * 100 + (y + dy)] - (
                                    ((0.4 * math.sqrt(dx ** 2 + dy ** 2)) / speed) * 3600), 0)

                    total_reward += reward
    reward = reward_base + total_reward
    return next_state, reward



# Create the figure and axis
fig, ax = plt.subplots()

# Create the heatmap plot for the initial frame
heatmap = ax.imshow(static_rewards[0], cmap='hot', interpolation='nearest')
plt.colorbar(heatmap)
num_timesteps = 1000
window_size = 100

# Update function for each animation frame
def update(frame):
    start_frame = frame - window_size + 1
    avg_rewards = np.mean(static_rewards[start_frame:frame + 1], axis=0)

    # Update the heatmap data for the current frame
    heatmap.set_array(avg_rewards)
    ax.set_title(f'Timestep: {frame}')
    return heatmap


# Create the animation
anim = animation.FuncAnimation(fig, update, frames=num_timesteps - window_size + 1, interval=30, blit=False)

# Export the animation as an MP4 video file
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=128000)
anim.save('heatmap_animation4.mp4', writer=writer)

# Show the animation
plt.show()

"""
static_rewards = []
for x in range(100):
    for y in range(100):
        sum_rwd = 0
        for o in range(1000):
            nxt = (o, x, y)
            nxt, rwd = transition(nxt, 10)
            sum_rwd += rwd
        static_rewards.append([sum_rwd / 1000])
        print(x,y,sum_rwd)
    print("max",max(static_rewards),len(static_rewards))
"""

nxt = (0, 37, 52)
sum_rwd = 0
reward_sequences = []
for o in range(400,900):
    nxt = (o, 37, 52)
    nxt, rwd = transition(nxt, 10)
    sum_rwd += rwd
print(sum_rwd/500)
"""
x = list(range(len(reward_sequences)))

plt.scatter(x, reward_sequences)
# Set the y-axis scale to logarithmic
plt.yscale('log')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Scatter Plot of the Sequence')

# Display the plot
plt.show()
"""
print("Training reward", sum_rwd/1000)
# Define a function to compute a unique index for each state
def state_index(state):
    t, x, y = state
    return t * 100 * 100 + x * 100 + y


# Define the MCTS algorithm

def mcts(state, models=False):
    # Initialize the search tree
    path_limit = 10000
    walk_limit = 1
    N = np.zeros((state_space, action_space))
    Q = np.zeros((state_space, action_space))
    max_reward = 0
    trailing_period = 50
    paths = []
    reward_paths = []
    max_reward_no_discount = 0
    max_reward_per_iter = 100
    last_update_time = time.time()
    last_update_time2 = time.time()
    total_time = time.time()
    status = state
    sum_of_totals = 0
    count_totals = 0
    path = []
    reward_base = 0
    # Run the simulations
    for _ in range(num_simulations):
        if _ % 5 == 0:
            current_time2 = time.time()
            print("Simulation #:", _)
            print("Time taken:", last_update_time2 - current_time2)
            last_update_time2 = time.time()
            print("path length:", len(path))
            print("last base reward:", reward_base/(len(path)+1))
        # Select a path through the tree
        path = []
        path_rewards = []
        current_state = status
        trajectory_positions = []
        init_reward = 0
        reward_base = 0
        while len(path) < path_limit:
            # Compute a unique index for the current state
            index = state_index(current_state)
            if models == True:
                model = ensemble_model.estimators_[index]

                # Create an empty list to store the predicted rewards for each action
                action_rewards = []

                # Iterate over each action and make a reward prediction
                for act in range(5):
                    feature = np.array([[state[0], state[1], state[2], act]])
                    reward_prediction = model.predict(feature)
                    action_rewards.append(reward_prediction[0])
                UCT = action_rewards[index]*(Q[index] / N[index]) + c * np.sqrt(np.log(N[index].sum()) / (N[index] + 1e-9))

            # Compute the UCT values
            else:
                UCT = (Q[index]/N[index]) + c * np.sqrt(np.log(N[index].sum()) / (N[index] + 1e-9))
            # Select the action with the maximum UCT value
            action = np.argmax(UCT)
            # Add the state-action pair to the path
            path.append((current_state, action))
            path_rewards.append([current_state[0],current_state[1],current_state[2],action,init_reward])


            # Check if we have reached a leaf node
            if N[index, action] == 0:
                break

            # Transition to the next state
            current_state, init_reward = transition(current_state, action)
            reward_base += init_reward
            """
            for i in range(len(path)):
                state, action = path[i]
                next_state, rew = transition(state, action)
                reward_base += rew
                if state_index(state) not in edges:
                    edges[state_index(state)] = []
                edges[state_index(state)].append((action, state_index(next_state)))"""

        # Simulate a trajectory from the leaf node
        total_reward = reward_base
        total_reward_no_discount = reward_base
        action_max = 0
        gamma = 0.97
        discount_factor = 1
        walk_length = len(path)
        while True:
            action_max += 1
            walk_length += 1
            # Select a random action
            action = 10

            # Transition to the next state
            current_state, reward = transition(current_state, action)

            # Update the total reward
            total_reward_no_discount += reward
            total_reward += discount_factor * reward
            if total_reward > max_reward:
                max_reward = total_reward
                max_reward_per_iter = total_reward / walk_length
            if total_reward_no_discount / walk_length > max_reward_no_discount and walk_length > 20:
                max_reward_no_discount = total_reward_no_discount / walk_length
                current_time = time.time()
                print('Time elapsed since last update:', current_time-last_update_time)
                print("New maximum reward:", max_reward_no_discount)
                print("Simulation number", _)
                print("Path Length", len(path))
                last_update_time = time.time()

            # trajectory_positions.append((current_state[1], current_state[2]))
            # Update the discount factor
            discount_factor *= gamma

            # Check if we have reached a terminal state
            if action_max > (walk_limit - len(path)):
                sum_of_totals += (total_reward/walk_length)
                count_totals += 1
                break
        # Append path to paths for testing
        if _ % 10 == 5:
            paths.append(path)
            reward_paths.append(path_rewards)
        # Update the search tree
        if len(path) < path_limit:
            for current_state, action in reversed(path):
                index = state_index(current_state)
                N[index, action] += 1
                Q[index, action] += total_reward / walk_length
        else:
            recency = 0
            for current_state, action in reversed(path):
                recency += 1
                index = state_index(current_state)
                N[index, action] += 1/(recency*recency)

        # position_history.append(trajectory_positions)
        reward_history.append(reward_base/len(path))

    # Return the optimal action
    #return np.argmax(N[state_index])
    return paths, reward_paths


base_tree = mcts((400,37,52))
paths = base_tree[0]
path_rewards = base_tree[1]



# Now you can use the ensemble_model for predictions


c_values = np.load("c_values_test.npy")
rewards = c_values

nxt = (0, 37, 52)
sum_rwd = 0
reward_sequences = []


for o in range(1000):
    nxt, rwd = transition(nxt, 10)
    sum_rwd += rwd

print("static testing reward", sum_rwd)
for path in paths:
    reward_test = 0
    reward_sequence = []
    print(len(path))
    for i in range(len(path)):
        state, action = path[i]
        next_state, rew = transition(state, action)
        reward_sequence.append([state[0], state[1], state[2], action, rew])
        reward_test += rew
    reward_sequences.append(reward_sequence)

    print("testing reward dynamic", reward_test/len(path))


# Create an empty list to store the predicted rewards
predicted_rewards = []
num_runs = 0
# Initialize a list to hold the individual XGBoost models
models = []
num_runs = 0
# Train an XGBoost model for each path in 'path_rewards'
for path in reward_sequences:
    num_runs += 1
    print(num_runs)
    # Extract features and rewards from the path
    features = np.array(path)[:, :-1]
    rewards = np.array(path)[:, -1]
    # Create an XGBoost model and fit it to the data
    model = XGBRegressor(objective='reg:squarederror', eta=0.01, max_depth=5)
    model.fit(features, rewards)

    # Add the trained model to the list
    models.append(model)

# Create an ensemble model by bagging the individual models
ensemble_model = BaggingRegressor(base_estimator=None, n_estimators=len(models), random_state=42)
ensemble_model.estimators_ = models


# Iterate over the last 200 paths in 'paths' for testing
# Iterate over the last 200 paths in 'paths' for testing
for path in paths:
    num_runs += 1
    print(num_runs)
    # Create an empty list to store the predicted rewards for the path
    path_predictions = []

    # Iterate over each state-action pair in the path
    for index, (state, action) in enumerate(path):
        # Get the corresponding XGBoost model from the ensemble
        model = ensemble_model.estimators_[index]

        # Create an empty list to store the predicted rewards for each action
        action_rewards = []

        # Iterate over each action and make a reward prediction
        for act in range(5):
            feature = np.array([[state[0], state[1], state[2], act]])
            reward_prediction = model.predict(feature)
            action_rewards.append(reward_prediction[0])

        # Append the list of predicted rewards for each action to 'path_predictions'
        print(action_rewards)
        path_predictions.append(action_rewards)

    # Append 'path_predictions' to 'predicted_rewards'
    predicted_rewards.append(path_predictions)

# Print the estimated rewards for each action given a state

def print_list_if_elements_differ(lst):
    if len(set(lst)) > 1:
        return True
    else:
        return False


for path in predicted_rewards:
    sum_rewards = 0
    for action_rewards in path:
        print(action_rewards)
        sum_rewards += max(action_rewards)
    print(sum_rewards/len(path))  # Empty line to separate paths

mcts((0,37,52),models=ensemble_model.estimators_)



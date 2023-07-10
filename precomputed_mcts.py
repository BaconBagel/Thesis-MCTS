import numpy as np
import pandas as pd
import time
import xgboost as xgb
from xgboost import DMatrix
import math
import numpy as np
from joblib import Parallel, delayed
import random
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import f
from scipy.stats import ttest_ind
import matplotlib as mpl
from matplotlib import animation
import cartopy.crs as ccrs
import contextily as ctx
from sklearn.cluster import KMeans
from numba import jit
from matplotlib.collections import LineCollection
from collections import Counter
from matplotlib import rc
import matplotlib.colors as mpl_colors
import networkx as nx
rc('animation', html='jshtml')

# Define the state space and action space
state_space = 100 * 100 * 100
action_space = 5
position_history = []
reward_history = []
edges = {}


# Define the number of simulations
num_simulations = 30000000
c = 200

path = []

"""
c_values = np.load("c_values_train_massive_days2.npy")
rewards = c_values

# Define the transition function

# Define the transition function
reward_array = np.zeros((500, 100, 100, 100))
n_days = 500
reward_horizon = 5
speed = 10

# Create indices arrays
x_indices = np.arange(100)
y_indices = np.arange(100)

# Create indices grid
x_grid, y_grid = np.meshgrid(x_indices, y_indices)

# Use vectorized operations
for n in range(n_days):
    for t in range(100):
        reward_base = rewards[n, t, x_grid * 100 + y_grid]

        if reward_horizon > 0:
            for dx in range(-reward_horizon, reward_horizon + 1):
                for dy in range(-reward_horizon, reward_horizon + 1):
                    if dx == 0 and dy == 0:
                        continue  # Skip the center cell

                    x_plus_dx = x_grid + dx
                    y_plus_dy = y_grid + dy

                    valid_indices = (0 <= x_plus_dx) & (x_plus_dx < 100) & (0 <= y_plus_dy) & (y_plus_dy < 100)

                    reward = rewards[n, t, x_plus_dx[valid_indices] * 100 + y_plus_dy[valid_indices]] - (
                            ((0.4 * np.sqrt(dx ** 2 + dy ** 2)) / speed) * 3600
                    )
                    reward[reward < 0] = 0
                    reward_array[n, t, x_grid[valid_indices], y_grid[valid_indices]] += reward

    print(n)

np.save('reward_array.npy', reward_array)
"""

reward_array = np.load("reward_array.npy")
# Set numbers above 1000 to 1000
reward_array = np.where(reward_array > 1000, 1000, reward_array)
reward_array = (reward_array - np.min(reward_array)) / (np.max(reward_array) - np.min(reward_array))
print(np.mean(reward_array))
p_zeros = np.count_nonzero(reward_array == 0) / reward_array.size
print("zeroes", p_zeros)


def transition(state, action, day):
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


    if t == 99:
        next_t = 1
    else:
        next_t = min(t + 1, 99)

    next_state = (next_t, x, y)
    reward = reward_array[day, next_t, x, y]  # Lookup reward in the pre-computed array
    return next_state, reward


# Define a function to compute a unique index for each state
def state_index(state):
    t, x, y = state
    return t * 100 * 100 + x * 100 + y
"""
reward_adjustment = np.zeros((100, 100, 100))
static_rewards = []
for x in range(0,100):
    for y in range(0,100):
        for o in range(0,100):
            sum_rwd = 0
            for i in range(499):
                nxt = (o, x, y)
                nxt, rwd = transition(nxt, 10, i)
                sum_rwd += rwd
            reward_adjustment[o, x, y] +=  sum_rwd/500
        static_rewards.append([sum_rwd / (100*500)])
        print(x,y,sum_rwd)
        print("max",max(static_rewards),len(static_rewards))

arr = reward_adjustment
window_size = 3  # Number of entries to consider for gradient calculation

gradients = np.zeros_like(arr)  # Initialize an array for storing gradients

for i in range(arr.shape[0]):
    start_idx = i + 1
    end_idx = start_idx + window_size

    if end_idx <= arr.shape[0]:
        diff = np.diff(arr[start_idx:end_idx], axis=0)
        gradients[i + 1:end_idx] = diff / arr[start_idx - 1:end_idx - 1]
    else:
        remaining = end_idx - arr.shape[0]
        diff = np.diff(arr[start_idx:], axis=0)
        gradients[i + 1:] = diff / arr[start_idx - 1:-1]

        cycled_values = arr[:remaining]
        cycled_diff = arr[start_idx] - cycled_values
        cycled_gradients = cycled_diff / cycled_values
        gradients[:remaining] = cycled_gradients

np.save("gradients.npy", gradients)
"""


nxt = (0, 37, 52)
sum_rwd = 0


reward_sequences = []
reward_sequences2 = []
reward_sequences3 = []
# Create arrays to store the data
x_data_combined = []
y_data_combined = []


# Iterate over all possible coordinates and compute the rewards
for x in range(100):
    for y in range(100):
        reward_sequences = []
        for t in range(100):
            sum_rwd = 0
            for i in range(499):
                nxt = (t, x, y)
                nxt, rwd = transition(nxt, 10, i)
                sum_rwd += rwd
            reward_sequences.append(sum_rwd / 499)

        x_data_combined.extend([(t, x, y) for t in range(100)])
        y_data_combined.extend(reward_sequences)

# Convert the data to numpy arrays
x_data_combined = np.array(x_data_combined)
y_data_combined = np.array(y_data_combined)

# Compute regression for the combined data
y_fit_combined, residuals_combined, mse_combined = compute_regression(x_data_combined, y_data_combined)

# Print the MSE reduction
mse_reduction = mse_combined / np.mean(residuals_combined ** 2)
print(f"MSE reduction: {mse_reduction}")
for o in range(0, 99):
    sum_rwd = 0
    for i in range(499):
        nxt = (o, 40, 54)
        nxt, rwd = transition(nxt, 10, i)
        sum_rwd += rwd
    reward_sequences.append(sum_rwd/(499))
    print(sum_rwd / (100 * 571))

for o in range(0, 99):
    sum_rwd = 0
    for i in range(499):
        nxt = (o, 36, 54)
        nxt, rwd = transition(nxt, 10, i)
        sum_rwd += rwd
    reward_sequences2.append(sum_rwd/(499))
    print(sum_rwd / (100 * 571))

for o in range(0, 99):
    sum_rwd = 0
    for i in range(499):
        nxt = (o, 28, 44)
        nxt, rwd = transition(nxt, 10, i)
        sum_rwd += rwd
    reward_sequences3.append(sum_rwd/(499))
    print(sum_rwd / (100 * 571))

print("count",reward_sequences.count(0))
print("count",reward_sequences2.count(0))
print("count",reward_sequences3.count(0))

def sine_func(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset


def compute_regression(x_data, y_data):
    popt, _ = curve_fit(sine_func, x_data, y_data, maxfev=50000, absolute_sigma=False)
    y_fit = sine_func(x_data, *popt)
    residuals = y_data - y_fit
    ssr = np.sum(residuals**2)
    return y_fit, residuals, ssr


# Compute sine regression for reward_sequences
x_data = np.arange(len(reward_sequences))
y_fit, residuals, ssr1 = compute_regression(x_data, reward_sequences)
mse_sine1 = np.mean(residuals**2)

# Compute sine regression for reward_sequences2
x_data2 = np.arange(len(reward_sequences2))
y_fit2, residuals2, ssr2 = compute_regression(x_data2, reward_sequences2)
mse_sine2 = np.mean(residuals2**2)

# Compute sine regression for reward_sequences3
x_data3 = np.arange(len(reward_sequences3))
y_fit3, residuals3, ssr3 = compute_regression(x_data3, reward_sequences3)
mse_sine3 = np.mean(residuals3**2)

# Compute linear regression for reward_sequences
coefficients1 = np.polyfit(x_data, reward_sequences, 1)
linear_regression1 = np.poly1d(coefficients1)
y_linear_fit1 = linear_regression1(x_data)
mse_linear1 = np.mean((reward_sequences - y_linear_fit1)**2)

# Compute linear regression for reward_sequences2
coefficients2 = np.polyfit(x_data2, reward_sequences2, 1)
linear_regression2 = np.poly1d(coefficients2)
y_linear_fit2 = linear_regression2(x_data2)
mse_linear2 = np.mean((reward_sequences2 - y_linear_fit2)**2)

# Compute linear regression for reward_sequences3
coefficients3 = np.polyfit(x_data3, reward_sequences3, 1)
linear_regression3 = np.poly1d(coefficients3)
y_linear_fit3 = linear_regression3(x_data3)
mse_linear3 = np.mean((reward_sequences3 - y_linear_fit3)**2)

# F-test and p-value calculation
df1 = len(reward_sequences) - 5
df2 = len(reward_sequences2) - 5
df3 = len(reward_sequences3) - 5

f_stat12 = (ssr1 / df1) / (ssr2 / df2)
f_stat13 = (ssr1 / df1) / (ssr3 / df3)
f_stat23 = (ssr2 / df2) / (ssr3 / df3)

p_value12 = f.sf(f_stat12, df1, df2)
p_value13 = f.sf(f_stat13, df1, df3)
p_value23 = f.sf(f_stat23, df2, df3)

print(f"p-value 1 vs 2: {p_value12}")
print(f"p-value 1 vs 3: {p_value13}")
print(f"p-value 2 vs 3: {p_value23}")

# t-test calculation
t_stat, p_value = ttest_ind(reward_sequences, reward_sequences2)
t_stat2, p_value2 = ttest_ind(reward_sequences, reward_sequences3)
t_stat3, p_value3 = ttest_ind(reward_sequences2, reward_sequences3)

# Print t-test results
print(f"t-statistic: {t_stat, t_stat2, t_stat3}")
print(f"p-value: {p_value, p_value2, p_value3}")

# Print MSE for sine regression
print(f"MSE Sine Regression 1: {mse_sine1}")
print(f"MSE Sine Regression 2: {mse_sine2}")
print(f"MSE Sine Regression 3: {mse_sine3}")

# Print MSE for linear regression
print(f"MSE Linear Regression 1: {mse_linear1}")
print(f"MSE Linear Regression 2: {mse_linear2}")
print(f"MSE Linear Regression 3: {mse_linear3}")

# Plotting
plt.scatter(x_data, reward_sequences, label='Reduction Potential at 40, 54', s=4, c="black")
plt.plot(x_data, y_fit, '--', label='Sine Regression 40, 54', linewidth=3.0)
plt.plot(x_data, y_linear_fit1, '--', label='Linear Regression 40, 54', linewidth=3.0)

plt.scatter(x_data2, reward_sequences2, label='Reduction Potential at 56, 40', s=4)
plt.plot(x_data2, y_fit2, '--', label='Sine Regression 56, 40', linewidth=3.0)
plt.plot(x_data2, y_linear_fit2, '--', label='Linear Regression 56, 40', linewidth=3.0)

plt.scatter(x_data3, reward_sequences3, label='Reward Sequence 28, 44', s=4)
plt.plot(x_data3, y_fit3, '--', label='Sine Regression 28, 44', linewidth=3.0)
plt.plot(x_data3, y_linear_fit3, '--', label='Linear Regression 28, 44', linewidth=3.0)

plt.xlabel('Timestep (ToD/100)')
plt.ylabel('Normalised Reduction Potential (seconds per timestep)')
plt.legend()
plt.show()




def mcts(state, models=False):
    # Initialize the search tree
    path_limit = 10000
    walk_limit = 5
    N = np.zeros((state_space, action_space))
    Q = np.zeros((state_space, action_space))
    max_reward = 0
    trailing_period = 50
    paths = []
    initialise = 1
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
        if _ % 500 == 0:
            current_time2 = time.time()
            print("Simulation #:", _)
            print("Time taken:", last_update_time2 - current_time2)
            last_update_time2 = time.time()
            print("path length:", len(path))
            print("last base reward:", reward_base/(len(path)+1))
            q_sum = 0
            q_count = 0
            q_state = status
            q_index = state_index(q_state)
            sum_red = 0
            for b in range(100):
                q_count += 1
                max_index = np.argmax(N[q_index])  # Find the index with the highest N value at q_index
                action = max_index # Assign the action as the max_index

                if N[q_index, action] == 0:
                    break

                q_sum += Q[q_index, action] / N[q_index, action]
                for x in range(500):
                    tem, red = transition(q_state, action, x)
                    sum_red += red
                q_state, tem = transition(q_state, action, x)
                q_index = state_index(q_state)
                reward_history.append(sum_red / (500 * q_count))
            print("last Q", q_sum/q_count, sum_red / (500 * q_count), q_count)
            if q_count > 99 and sum_red / (500 * q_count) > 3.8:
                break

        # Select a path through the tree
        path = []
        path_rewards = []
        current_state = status
        current_state = list(current_state)
        current_state[0] = random.randint(0,99)
        current_state = tuple(current_state )
        trajectory_positions = []
        init_reward = 0
        reward_base = 0
        while len(path) < path_limit:
            day = random.randint(0, 499)
            # Compute a unique index for the current state
            index = state_index(current_state)
            # Compute the UCT values
            UCT = (Q[index]/N[index]) + c * np.sqrt(np.log(N[index].sum()) / (N[index] + 1e-9))
            # Select the action with the maximum UCT value
            if len(path) < 99 and initialise == 1:
                action = 4
            elif len(path) < 99 and initialise == 0:
                action = np.argmax(UCT)
            else:
                # Get the sorted indices of UCT values in descending order
                sorted_indices = np.argsort(UCT)[::-1]
                # Find the first action that has not been used in the current path
                for ind in sorted_indices:
                    if (current_state, ind) not in path:
                        action = ind
                        break
                    else:
                        # If all actions have been used, choose a random action
                        action = np.random.choice(action_space)
            if len(path) > 99:
                initialise = 0
            # Add the state-action pair to the path
            path.append((current_state, action))
            path_rewards.append([current_state[0],current_state[1],current_state[2],action,init_reward])


            # Check if we have reached a leaf node
            if N[index, action] == 0:
                break

            # Transition to the next state
            current_state, init_reward = transition(current_state, action, day)
            reward_base += init_reward

        # Simulate a trajectory from the leaf node
        total_reward = reward_base
        total_reward_no_discount = reward_base
        action_max = 0
        gamma = 1
        discount_factor = 1
        walk_length = len(path)
        while True:
            action_max += 1
            walk_length += 1
            # Select a random action
            action = random.randint(0,20)

            # Transition to the next state
            for day in range(500):
                current_state, reward = transition(current_state, action, day)
            reward = reward / 500

            # Update the total reward
            total_reward_no_discount += reward
            total_reward += discount_factor * reward
            if total_reward > max_reward:
                max_reward = total_reward
                max_reward_per_iter = total_reward / walk_length
            if total_reward_no_discount / walk_length > max_reward_no_discount:
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
            if action_max > walk_limit:
                sum_of_totals += (total_reward/walk_length)
                count_totals += 1
                break
        # Append path to paths for testing
        if _ % 10 == 5:
            paths.append(path)
            reward_paths.append(path_rewards)
        # Update the search tree
        for current_state, action in reversed(path):
            index = state_index(current_state)
            N[index, action] += 1
            Q[index, action] += (total_reward) / walk_length


    #return np.argmax(N[state_index])
    return Q, N


base_tree = mcts((40,36,54))
Q = base_tree[0]
N = base_tree[1]

q_sum = 0
q_count = 0
q_state = (40,36,54)
q_index = state_index(q_state)
sum_red = 0
reward_history = []
location_history = []

for b in range(110):
    q_count += 1
    max_index = np.argmax(Q[q_index])  # Find the index with the highest N value at q_index
    action = max_index  # Assign the action as the max_index

    if N[q_index, action] == 0:
        break

    q_sum += Q[q_index, action] / N[q_index, action]
    for x in range(500):
        tem, red = transition(q_state, action, x)
        sum_red += red
    q_state, tem = transition(q_state, action, x)
    print(q_state, action)
    location_history.append(q_state)
    q_index = state_index(q_state)
    reward_history.append(sum_red/(500*q_count))

print(sum(reward_history)/len(reward_history),len(reward_history))

time.sleep(5000)
# Define the grid size
grid_size = 100

# Initialize the rewards array
static_rewards = np.zeros((100, grid_size, grid_size))


# Compute rewards for each grid cell
for x in range(grid_size):
    for y in range(grid_size):
        for o in range(100):
            sum_rwd = 0
            for i in range(500):
                nxt = (o, x, y)
                nxt, rwd = transition(nxt, 10, i)
                sum_rwd += rwd
            static_rewards[o][x][y] = sum_rwd / 500
            print(x, y, sum_rwd)

# Create the figure and axis with a cartopy projection
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Define the extent of the map (New York)
lon_min, lon_max = -74.17128737, -73.70831903
lat_min, lat_max = 40.56145833, 40.91090125
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Add a background map using contextily
ctx.add_basemap(ax, zoom=12, crs=ccrs.PlateCarree(), source=ctx.providers.Stamen.Toner)

cmap = plt.cm.get_cmap('hot')  # Get the colormap
alpha = np.linspace(0, 1, 256)  # Create an array of alpha values

# Create a new colormap by concatenating the original colormap with alpha values
new_cmap = mpl.colors.ListedColormap(cmap(np.arange(cmap.N)))
new_cmap.colors[:, -1] = alpha

heatmap = ax.imshow(static_rewards[0], cmap=new_cmap, origin='lower', extent=[lon_min, lon_max, lat_min, lat_max])
plt.colorbar(heatmap)
num_timesteps = 100
window_size = 1


def plot_clusters(ax, rewards, n_clusters=3):
    X, Y = np.meshgrid(np.linspace(lon_min, lon_max, grid_size), np.linspace(lat_min, lat_max, grid_size))
    if np.isnan(rewards).all():
        Z = np.zeros_like(rewards.flatten())
    else:
        Z = np.nan_to_num(rewards.flatten(), nan=np.nanmean(rewards))
    XY = np.vstack((X.flatten(), Y.flatten())).T

    kmeans = KMeans(n_clusters=n_clusters).fit(XY, sample_weight=Z)
    for i in range(n_clusters):
        cluster_points = XY[kmeans.labels_ == i]

        if len(cluster_points) > 0:
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                ax.plot(cluster_points[simplex, 0], cluster_points[simplex, 1], 'k-', transform=ccrs.PlateCarree())

            cross_x, cross_y = kmeans.cluster_centers_[i]
            ax.scatter(cross_x, cross_y, marker='x', color='red', transform=ccrs.PlateCarree(), zorder=20)

    if len(Z) == 0:
        ax.text(lon_min, lat_min, 'Empty rewards array', transform=ccrs.PlateCarree())




# Update function for each animation frame
def update(frame):
    start_frame = frame - window_size + 1
    avg_rewards = np.mean(static_rewards[start_frame:frame], axis=0)

    # Update the heatmap data for the current frame
    heatmap.set_array(avg_rewards)
    ax.set_title(f'Timestep: {frame}')
    # plot_clusters(ax, avg_rewards)
    # Plot all the saved locations up to the current frame
    if frame < len(location_history):
        loc = location_history[frame]
        lon = loc[1] / grid_size * (lon_max - lon_min) + lon_min
        lat = loc[2] / grid_size * (lat_max - lat_min) + lat_min
        x, y = ccrs.PlateCarree().transform_point(lon, lat, ccrs.PlateCarree())

        scatter = ax.scatter(x, y, color='blue', s=20, transform=ccrs.PlateCarree(), zorder=10)

        return heatmap, scatter  # Return both the heatmap and scatter plot

    return heatmap,

# Create the animation
anim = animation.FuncAnimation(fig, update, frames=num_timesteps - window_size + 1, interval=200, blit=True)

# Export the animation as an MP4 video file
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=128000)
anim.save('heatmap_animation_map4.mp4', writer=writer)

# Show the animation
plt.show()
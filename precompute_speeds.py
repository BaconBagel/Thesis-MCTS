import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def scale_column(column_name, df, min_value, max_value):
    # Define the column to be scaled
    column_to_scale = column_name

    # Perform Min-Max scaling to normalize the column between 0 and 1
    df[column_to_scale] = (df[column_to_scale] - min_value) / (max_value - min_value)

    # Scale the column to be between 0 and 100
    df[column_to_scale] = df[column_to_scale] * 100

    # Convert the column to integer data type
    df[column_to_scale] = df[column_to_scale].astype(int)

    return df


# Load the rewards CSV file into a pandas dataframe
rewards_df = pd.read_csv("mcts_2021_huge.csv", skiprows=lambda x: x % 2 != 1)

# Multiply the 'time' column by 1000 and convert it to an integer
rewards_df['time'] = rewards_df['time'] * 100
rewards_df['time'] = rewards_df['time'].astype(int)

# Calculate the minimum and maximum latitude and longitude values
min_latitude = rewards_df['latitude'].min()
max_latitude = rewards_df['latitude'].max()
min_longitude = rewards_df['longitude'].min()
max_longitude = rewards_df['longitude'].max()

rewards_df = scale_column("latitude", rewards_df, min_latitude, max_latitude)
rewards_df = scale_column("longitude", rewards_df, min_longitude, max_longitude)

rewards_df['location_index'] = (rewards_df["latitude"] * 100) + rewards_df["longitude"]

columns_to_use = ['time', 'location_index', 'response_time']

# Extract the date component from the "date" column
rewards_df['date'] = pd.to_datetime(rewards_df['date'])
rewards_df['date'] = rewards_df['date'].dt.date

# Add a new column for the year and month
rewards_df['year_month'] = pd.to_datetime(rewards_df['date']).dt.to_period('M')

# Specify the desired year and month range
start_year_month = pd.Period('2020-01', 'M')
end_year_month = pd.Period('2020-06', 'M')

selected_rewards_df = rewards_df.loc[(rewards_df['year_month'] >= start_year_month) & (rewards_df['year_month'] <= end_year_month)]

# Determine the number of unique days in the selected range
n_days = selected_rewards_df['date'].nunique()

# Create a lookup dictionary for response times by (a, b) pair and date
lookup_dict = {}
for date, group in selected_rewards_df.groupby('date'):
    pivot_table = group.pivot_table(index=['time', 'location_index'], values='response_time', aggfunc=np.sum)
    index_values = set(pivot_table.index)
    lookup_dict[date] = {(a, b): pivot_table.loc[(a, b), 'response_time'] for a, b in index_values}
    print(date)

# Load the traffic CSV file into a pandas dataframe
traffic_df = pd.read_csv("DOT_Traffic_Speeds_NBE.csv", dtype={'LINK_POINTS': str})

# Extract the date and time components from the "DATA_AS_OF" column
traffic_df['date'] = pd.to_datetime(traffic_df['DATA_AS_OF']).dt.date
traffic_df['time'] = pd.to_datetime(traffic_df['DATA_AS_OF']).dt.time

# Extract the latitude and longitude from the link points
traffic_df['latitude'] = traffic_df['LINK_POINTS'].apply(lambda x: np.mean([float(pair.split(',')[0]) for pair in x.split()]))
traffic_df['longitude'] = traffic_df['LINK_POINTS'].apply(lambda x: np.mean([float(pair.split(',')[1]) for pair in x.split()]))

# Scale latitude and longitude based on rewards data
traffic_df = scale_column("latitude", traffic_df, min_latitude, max_latitude)
traffic_df = scale_column("longitude", traffic_df, min_longitude, max_longitude)

# Create a k-d tree from rewards data for nearest neighbor lookup
rewards_tree = cKDTree(rewards_df[['latitude', 'longitude']].values)

# Convert the link points coordinates to location indices
traffic_df['location_index'] = rewards_tree.query(traffic_df[['latitude', 'longitude']].values)[1]

# Calculate the average speed for each location index, date, and time
traffic_avg_speeds = traffic_df.groupby(['date', 'time', 'location_index'])['SPEED'].mean().reset_index()

# Create a lookup dictionary for traffic speeds by (a, b) pair, date, and time
traffic_lookup_dict = {}
for date, time, group in traffic_avg_speeds.groupby(['date', 'time']):
    pivot_table = group.pivot_table(index='location_index', values='SPEED', aggfunc=np.mean)
    index_values = set(pivot_table.index)
    traffic_lookup_dict[(date, time)] = {(a, b): pivot_table.loc[(a, b), 'SPEED'] for a, b in index_values}
    print(date, time)

# Specify the batch length
batch_length = 1
n_batches = n_days // batch_length

# Create a list to store the mean values for each batch
mean_values = []
speeds_array = []
# Calculate the mean and speed values for each batch
for i, date in enumerate(lookup_dict.keys()):
    print(date)
    if i % batch_length == 0:
        max_a = max(a for a, _ in lookup_dict[date])
        max_b = max(b for _, b in lookup_dict[date])
        current_batch = np.zeros((max_a + 1, max_b + 1))
        current_speeds = np.zeros((max_a + 1, max_b + 1))
    for a, b in lookup_dict[date]:
        current_batch[a, b] += lookup_dict[date][(a, b)]
        if (date, time) in traffic_lookup_dict and (a, b) in traffic_lookup_dict[(date, time)]:
            current_speeds[a, b] += traffic_lookup_dict[(date, time)][(a, b)]

    if i % batch_length == (batch_length - 1):
        current_batch /= batch_length
        current_speeds /= batch_length
        mean_values.append(current_batch)
        speeds_array.append(current_speeds)

# Convert the lists of mean and speed values to numpy arrays
mean_values = np.array(mean_values)
speeds_array = np.array(speeds_array)

print(mean_values)
print(mean_values.shape)
np.save('rewards_array.npy', mean_values)
np.save('speeds_array.npy', speeds_array)
print("done")

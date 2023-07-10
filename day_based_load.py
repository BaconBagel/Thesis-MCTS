import pandas as pd
import numpy as np

def scale_column(column_name, df):
    # Define the column to be scaled
    column_to_scale = column_name

    # Perform Min-Max scaling to normalize the column between 0 and 1
    min_value = df[column_to_scale].min()
    max_value = df[column_to_scale].max()
    df[column_to_scale] = (df[column_to_scale] - min_value) / (max_value - min_value)

    # Scale the column to be between 0 and 100
    df[column_to_scale] = df[column_to_scale] * 100

    # Convert the column to integer data type
    df[column_to_scale] = df[column_to_scale].astype(int)

    # Print the updated dataframe
    return df


# Load the CSV file into a pandas dataframe
df = pd.read_csv("mcts_2021_huge.csv", skiprows=lambda x: x % 2 != 1)

# Multiply the 'time' column by 1000 and convert it to an integer
df['time'] = df['time'] * 100
df['time'] = df['time'].astype(int)

df = scale_column("latitude", df)
df = scale_column("longitude", df)

df['location_index'] = (df["latitude"] * 100) + df["longitude"]

columns_to_use = ['time', 'location_index', 'response_time']

# Extract the date component from the "date" column
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.date

# Add a new column for the year and month
df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')

# Specify the desired year and month range
start_year_month = pd.Period('2020-01', 'M')
end_year_month = pd.Period('2020-12', 'M')

df = df.loc[(df['year_month'] >= start_year_month) & (df['year_month'] <= end_year_month)]
# Determine the number of unique days
n_days = df['date'].nunique()

# Create a 3D numpy array with the specified dimensions
a_values = np.arange(1, 101)
b_values = np.arange(1, 10001)
c_values = np.zeros((n_days, len(a_values), len(b_values)))

# Create a lookup dictionary for response times by (a, b) pair and date
lookup_dict = {}
for date, group in df.groupby('date'):
    pivot_table = group.pivot_table(index=['time', 'location_index'], values='response_time', aggfunc=np.sum)
    index_values = set(pivot_table.index)
    print(date)
    lookup_dict[date] = {(a, b): pivot_table.loc[(a, b), 'response_time'] for a, b in index_values}

# Populate the array for each day
for i, date in enumerate(lookup_dict.keys()):
    print(date)
    for j, a in enumerate(a_values):
        for k, b in enumerate(b_values):
            if (a, b) in lookup_dict[date]:
                c_values[i, j, k] = lookup_dict[date][(a, b)]
print(c_values)
print(c_values.shape)
np.save('c_values_train_2020.npy', c_values)
print("done")

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
end_year_month = pd.Period('2020-06', 'M')

selected_df = df.loc[(df['year_month'] >= start_year_month) & (df['year_month'] <= end_year_month)]

# Determine the number of unique days in the selected range
n_days = selected_df['date'].nunique()

# Create a lookup dictionary for response times by (a, b) pair and date
lookup_dict = {}
for date, group in selected_df.groupby('date'):
    pivot_table = group.pivot_table(index=['time', 'location_index'], values='response_time', aggfunc=np.sum)
    index_values = set(pivot_table.index)
    lookup_dict[date] = {(a, b): pivot_table.loc[(a, b), 'response_time'] for a, b in index_values}
    print(date)

# Specify the batch length
batch_length = 1
n_batches = n_days // batch_length

# Create a list to store the mean values for each batch
mean_values = []

# Calculate the mean for each batch
for i, date in enumerate(lookup_dict.keys()):
    print(date)
    if i % batch_length == 0:
        max_a = max(a for a, _ in lookup_dict[date])
        max_b = max(b for _, b in lookup_dict[date])
        current_batch = np.zeros((max_a + 1, max_b + 1))
    for a, b in lookup_dict[date]:
        current_batch[a, b] += lookup_dict[date][(a, b)]

    if i % batch_length == (batch_length - 1):
        current_batch /= batch_length
        mean_values.append(current_batch)

# Convert the list of mean values to a numpy array
mean_values = np.array(mean_values)

print(mean_values)
print(mean_values.shape)
np.save('mean_values_train.npy', mean_values)
print("done")

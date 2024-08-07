import pandas as pd
import numpy as np
from datetime import datetime
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


pm25_ground = pd.read_csv('./SampleData/pm25_ground.txt')
pm25_missing = pd.read_csv('./SampleData/pm25_missing.txt')

# Ensure datetime is in datetime format
pm25_ground['datetime'] = pd.to_datetime(pm25_ground['datetime'])
pm25_missing['datetime'] = pd.to_datetime(pm25_missing['datetime'])

# Get the number of entries
num_entries = pm25_ground.shape[0]
print(num_entries)

#pm25_ground = pm25_ground.dropna(thresh=len(pm25_ground.columns)-2)
pm25_ground = pm25_ground.dropna(thresh=len(pm25_ground.columns)-2)

print(pm25_ground.shape[0])
num_entries = pm25_ground.shape[0]


total_values = pm25_ground.size

# Get the number of missing values
num_missing_values = pm25_ground.isnull().sum().sum()

percentage_missing = (num_missing_values / total_values) * 100

print(f"Number of entries: {num_entries}")
print(f"Number of missing values: {num_missing_values}")
print(f"Percentage of missing values: {percentage_missing}")

# Get the number of entries
num_entries = pm25_missing.shape[0]

total_values = pm25_missing.size

# Get the number of missing values
num_missing_values = pm25_missing.isnull().sum().sum()

percentage_missing = (num_missing_values / total_values) * 100

print(f"Number of entries: {num_entries}")
print(f"Number of missing values: {num_missing_values}")
print(f"Percentage of missing values: {percentage_missing}")

# # Create a copy of the missing data to maintain original nulls
# combined_data = pm25_missing.copy()
#
# # Add ground truth columns to the combined data
# for col in pm25_missing.columns[1:]:
#     combined_data[f'{col}_ground'] = pm25_ground[col]
#
# # Ensure datetime column is in datetime format in combined_data
# combined_data['datetime'] = pd.to_datetime(combined_data['datetime'])

# Split the data into training and testing sets
def split_train_test(data):
    data['month'] = data['datetime'].dt.month
    test_data = data[data['month'].isin([3, 6, 9, 12])]
    train_data = data[~data['month'].isin([3, 6, 9, 12])]
    return train_data.drop(columns=['month']), test_data.drop(columns=['month'])

train_data_, test_data_ = split_train_test(pm25_ground)#combined_data)

# Get the number of entries
num_entries = test_data_.shape[0]

total_values = test_data_.size

# Get the number of missing values
num_missing_values = test_data_.isnull().sum().sum()

percentage_missing = (num_missing_values / total_values) * 100

print(f"Number of entries: {num_entries}")
print(f"Number of missing values: {num_missing_values}")
print(f"Percentage of missing values: {percentage_missing}")

# Introduce additional missing values randomly (10% of the data) while keeping ground truth intact
def introduce_missing_values(data, missing_percentage=0.23):
    modified_data = data.copy()
    total_values = modified_data.size - len(modified_data['datetime'])  # Exclude datetime column
    num_missing = int(total_values * missing_percentage)

    # Get the current missing mask
    current_missing_mask = modified_data.isna().values[:, 1:]

    # Flatten the data except for the datetime column
    data_values = modified_data.iloc[:, 1:].values.flatten()

    # Get indices that are not currently missing
    valid_indices = np.where(~current_missing_mask.flatten())[0]

    # Select random indices from valid_indices to introduce new missing values
    missing_indices = np.random.choice(valid_indices, num_missing, replace=False)

    # Introduce NaNs
    data_values[missing_indices] = np.nan

    # Reshape and put back into DataFrame
    modified_data.iloc[:, 1:] = data_values.reshape(modified_data.iloc[:, 1:].shape)
    return modified_data

# Apply missing values to train and test data
train_data = introduce_missing_values(train_data_)
test_data = introduce_missing_values(test_data_)

# Add ground truth columns to the data
for col in train_data_.columns[1:]:
    train_data[f'{col}_ground'] = train_data_[col]
    test_data[f'{col}_ground'] = test_data_[col]

# Ensure datetime column is in datetime format in the modified data
train_data['datetime'] = pd.to_datetime(train_data['datetime'])
test_data['datetime'] = pd.to_datetime(test_data['datetime'])

# Get the number of entries
num_entries = train_data.shape[0]

total_values = train_data.size

# Get the number of missing values
num_missing_values = train_data.isnull().sum().sum()

percentage_missing = (num_missing_values / total_values) * 100

print(f"Number of entries: {num_entries}")
print(f"Number of missing values: {num_missing_values}")
print(f"Percentage of missing values: {percentage_missing}")

# Get the number of entries
num_entries = test_data.shape[0]

total_values = test_data.size

# Get the number of missing values
num_missing_values = test_data.isnull().sum().sum()

percentage_missing = (num_missing_values / total_values) * 100

print(f"Number of entries: {num_entries}")
print(f"Number of missing values: {num_missing_values}")
print(f"Percentage of missing values: {percentage_missing}")

# Generate random time series samples of 36 consecutive steps for the training set
def random_select_time_series(data, sequence_length=36, num_samples=1000):
    time_series_data = []
    ground_truth_data = []
    max_start_index = len(data) - sequence_length
    random_indices = np.random.randint(0, max_start_index + 1, num_samples)
    for start in random_indices:
        end = start + sequence_length
        time_series_data.append(data.iloc[start:end, 1:37].values)
        ground_truth_data.append(data.iloc[start:end, 37:].values)
    return np.array(time_series_data), np.array(ground_truth_data)

def sequential_select_time_series2(data, sequence_length=36):
    time_series_data = []
    ground_truth_data = []
    for start in range(0, len(data) - sequence_length + 1, 2):
        end = start + sequence_length
        time_series_data.append(data.iloc[start:end, 1:37].values)
        ground_truth_data.append(data.iloc[start:end, 37:].values)
    return np.array(time_series_data), np.array(ground_truth_data)

# Create non-random time series samples of 36 consecutive steps for the test set
def sequential_select_time_series(data, sequence_length=36):
    time_series_data = []
    ground_truth_data = []
    for start in range(0, len(data) - sequence_length + 1):
        end = start + sequence_length
        time_series_data.append(data.iloc[start:end, 1:37].values)
        ground_truth_data.append(data.iloc[start:end, 37:].values)
    return np.array(time_series_data), np.array(ground_truth_data)

# Define number of samples you want to select
num_samples = 4500

# Generate the time series from the training data
train_series, train_ground_truth = random_select_time_series(train_data, num_samples=num_samples)

# Generate the time series from the test data
test_series, test_ground_truth = sequential_select_time_series(test_data)

# Display the shapes of the resulting datasets
print(train_series.shape, train_ground_truth.shape, test_series.shape, test_ground_truth.shape,
 train_data.shape, test_data.shape)

# Filter out invalid time series
def filter_valid_time_series(series_data, ground_truth_data):
    valid_indices = []
    for i in range(series_data.shape[0]):
        series = series_data[i]
        if not np.isnan(series).all(axis=1).any() and not np.isnan(series).all(axis=0).any():
            valid_indices.append(i)
    return series_data[valid_indices], ground_truth_data[valid_indices]


# Filter training and test data
train_series_filtered, train_ground_truth_filtered = filter_valid_time_series(train_series, train_ground_truth)
test_series_filtered, test_ground_truth_filtered = filter_valid_time_series(test_series, test_ground_truth)

# Display the shapes of the resulting datasets
print(train_series_filtered.shape, train_ground_truth_filtered.shape, test_series_filtered.shape, test_ground_truth_filtered.shape)


# Normalize features

# Normalize features ignoring NaNs
def normalize_data(train_data, test_data, ground_truth_train, ground_truth_test):
    mean = np.nanmean(train_data, axis=0)
    std = np.nanstd(train_data, axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    ground_truth_train = (ground_truth_train - mean) / std
    ground_truth_test = (ground_truth_test - mean) / std
    return train_data, test_data, ground_truth_train, ground_truth_test, mean, std


train_series_normalized, test_series_normalized, train_ground_truth_normalized, test_ground_truth_normalized, mean, std = normalize_data(train_series_filtered, test_series_filtered, train_ground_truth_filtered, test_ground_truth_filtered)


# Function to generate masks and deltas
def generate_masks_and_deltas(values):
    masks = ~np.isnan(values)
    deltas = np.zeros_like(values)
    deltas[0] = 1
    for t in range(1, values.shape[0]):
        deltas[t] = 1 + (1 - masks[t]) * deltas[t - 1]
    return masks, deltas


# Function to process a time series record
def process_record(values, ground_truth):
    masks, deltas = generate_masks_and_deltas(values)
    eval_masks = masks ^ ~np.isnan(ground_truth)

    forwards = pd.DataFrame(values).ffill().fillna(0.0).to_numpy()
    record = {
        'values': np.nan_to_num(values).tolist(),
        'masks': masks.astype('int32').tolist(),
        'evals': np.nan_to_num(ground_truth).tolist(),
        'eval_masks': eval_masks.astype('int32').tolist(),
        'forwards': forwards.tolist(),
        'deltas': deltas.tolist()
    }
    return record


# Process all training records
train_records = []
for i in range(train_series_normalized.shape[0]):
    values = train_series_normalized[i]
    ground_truth = train_ground_truth_normalized[i]
    record = {
        'forward': process_record(values, ground_truth),
        'backward': process_record(values[::-1], ground_truth[::-1]),
        'is_train': 1  # Indicates that this record is from the training set

    }
    train_records.append(record)

# Process all test records
test_records = []
for i in range(test_series_normalized.shape[0]):
    values = test_series_normalized[i]
    ground_truth = test_ground_truth_normalized[i]
    record = {
        'forward': process_record(values, ground_truth),
        'backward': process_record(values[::-1], ground_truth[::-1]),
        'is_train': 0  # Indicates that this record is from the test set

    }
    test_records.append(record)

# Save to JSON files
with open('train.json', 'w') as f:
    json.dump(train_records, f, cls=NpEncoder)

with open('test.json', 'w') as f:
    json.dump(test_records, f, cls=NpEncoder)
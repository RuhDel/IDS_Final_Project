import pandas as pd
import os

# Path to your dataset folder
dataset_path = "."

# Load and concatenate all CSV files
dataframes = []
for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(dataset_path, file))
        dataframes.append(df)

# Combine all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

# Check for missing values
print("Missing Values:\n", combined_df.isnull().sum())

# Drop or handle missing values
combined_df = combined_df.dropna()  # You can also choose to fill missing values

# Preview the data
print("Data Preview:\n", combined_df.head())

# Save the combined data for later use
combined_df.to_csv("combined_dataset.csv", index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

data_path = "combined_dataset.csv"
with tqdm(total=1, desc="Loading Data") as pbar:
    data = pd.read_csv(data_path)
    pbar.update(1)

with tqdm(total=1, desc="Splitting Data") as pbar:
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['Label'])
    pbar.update(1)

with tqdm(total=2, desc="Saving Datasets") as pbar:
    train_data.to_csv("train_dataset.csv", index=False)
    pbar.update(1)
    test_data.to_csv("test_dataset.csv", index=False)
    pbar.update(1)

print("Data split into training and testing sets successfully.")
print(f"Training set has {train_data.shape[0]} samples.")
print(f"Testing set has {test_data.shape[0]} samples.")

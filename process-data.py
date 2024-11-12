import pandas as pd
import os
from tqdm import tqdm

dataset_path = "."
columns_to_keep = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", 
                   "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max", 
                   "Bwd Packet Length Max", "Flow IAT Mean", "Fwd IAT Mean", "Bwd IAT Mean", 
                   "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "FIN Flag Count",
                   "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", 
                   "CWE Flag Count", "ECE Flag Count", "Label"]

all_data = []

csv_files = [file for file in os.listdir(dataset_path) if file.endswith(".csv")]
with tqdm(total=len(csv_files), desc="Combining CSV files") as pbar:
    for file in csv_files:
        pbar.set_postfix({"Current File": file})

        df = pd.read_csv(os.path.join(dataset_path, file))
        df.columns = df.columns.str.strip()

        existing_columns = [col for col in columns_to_keep if col in df.columns]
        df = df[existing_columns]

        all_data.append(df)
        pbar.update(1)

combined_df = pd.concat(all_data, ignore_index=True)

combined_df = combined_df.drop_duplicates()

if "Label" in combined_df.columns:
    benign_count = combined_df[combined_df["Label"] == "BENIGN"].shape[0]
    non_benign_count = combined_df[combined_df["Label"] != "BENIGN"].shape[0]
    keep_benign_count = int(non_benign_count * 0.75)

    benign_df = combined_df[combined_df["Label"] == "BENIGN"].sample(n=keep_benign_count, random_state=42)
    non_benign_df = combined_df[combined_df["Label"] != "BENIGN"]
    combined_df = pd.concat([benign_df, non_benign_df]).reset_index(drop=True)

print(f"Final combined data has {combined_df.shape[0]} rows after cleaning.")

combined_df.to_csv("combined_dataset.csv", index=False)
print("Data combined and cleaned successfully.")

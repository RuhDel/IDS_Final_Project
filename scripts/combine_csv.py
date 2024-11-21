import pandas as pd
import os

def combine_csv_files(input_dir, output_file):
    
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]
    combined_df = pd.DataFrame()

    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        print(f"Reading {file_path}...")
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        print(f"Combined {file} into the main DataFrame.")

    # Optionally, reset index
    combined_df.reset_index(drop=True, inplace=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"All CSV files combined into {output_file}.")

if __name__ == "__main__":
    input_directory = './data/'        # Directory containing the 8 CSV files
    output_csv = './data/combined.csv' # Output combined CSV file
    combine_csv_files(input_directory, output_csv)

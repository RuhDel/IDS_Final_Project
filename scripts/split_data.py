import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_csv, train_csv, test_csv, test_size=0.2, random_state=42):
    
    # Load the preprocessed data
    print(f"Loading preprocessed data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Define the label column based on your dataset
    label_column = 'label'

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in the dataset.")

    X = df.drop(columns=[label_column])
    y = df[label_column]

    print(f"Dataset shape before split: {df.shape}")
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")

    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Combine features and labels for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Save to CSV
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Training set saved to {train_csv} with shape {train_df.shape}.")
    print(f"Testing set saved to {test_csv} with shape {test_df.shape}.")

if __name__ == "__main__":
    input_preprocessed_csv = './data/preprocessed.csv'
    output_train_csv = './data/train.csv'
    output_test_csv = './data/test.csv'
    split_data(input_preprocessed_csv, output_train_csv, output_test_csv)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pickle

def preprocess_data(input_csv, output_csv):
    
    # Load the data
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Standardize column names: lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip()
    print("Column names standardized to lowercase and stripped of whitespace.")

    # Display initial information
    print("Initial DataFrame shape:", df.shape)
    print("Missing values per column:\n", df.isnull().sum())

    # Replace infinite values with NaN
    print("Replacing infinite values with NaN...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Replacement complete.")

    # Identify numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    print(f"Numerical columns: {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    # Cap extremely large values to a defined threshold
    threshold = 1e6
    print(f"Capping numerical values at {threshold} to handle extremely large values...")
    df[numerical_cols] = df[numerical_cols].applymap(lambda x: threshold if x > threshold else x)
    print("Capping complete.")

    # Impute numerical features with median
    print("Imputing missing values in numerical columns with median...")
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    print("Numerical imputation complete.")

    # Impute categorical features with most frequent value
    if len(categorical_cols) > 0:
        print("Imputing missing values in categorical columns with most frequent value...")
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        print("Categorical imputation complete.")
    else:
        print("No categorical columns to impute.")

    # Encoding Categorical Variables
    if len(categorical_cols) > 0:
        print("Encoding categorical variables using Label Encoding...")
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"Encoded categorical column: {col}")
        print("Categorical encoding complete.")
    else:
        print("No categorical columns to encode.")

    # Mapping Labels to Binary
    if 'label' in df.columns:
        print("Mapping labels to binary: 'Normal' as 0 and others as 1.")
        
        # Check unique labels before mapping
        unique_labels = df['label'].unique()
        print(f"Unique labels before mapping: {unique_labels}")

        # Find the integer value corresponding to 'Normal'
        normal_label = None
        if 'normal' in df['label'].astype(str).str.lower().unique():
            df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'normal' else 1)
        else:
            df['label'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
        print("Label mapping complete.")
    else:
        print("Label column 'label' not found.")
        raise ValueError("Label column 'label' not found in the dataset.")

    # Normalization of Numerical Features
    print("Normalizing numerical features using StandardScaler...")
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print("Normalization complete.")

    # Save the preprocessed data
    df.to_csv(output_csv, index=False)
    print(f"Preprocessed data saved to {output_csv}.")

    # Save label encoders and scaler for future use
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Preprocessing artifacts saved.")

if __name__ == "__main__":
    input_combined_csv = './data/combined.csv'
    output_preprocessed_csv = './data/preprocessed.csv'
    preprocess_data(input_combined_csv, output_preprocessed_csv)

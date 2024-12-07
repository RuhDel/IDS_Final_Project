import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import json

def load_data(train_csv, label_column='label'):
    df = pd.read_csv(train_csv)
    X_train = df.drop(columns=[label_column])
    y_train = df[label_column]
    return X_train, y_train

def build_dnn_model(input_dim, output_dim=1, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='sigmoid'))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):

    # Calculate class weights to handle class imbalance
    class_weights_values = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=y_train.unique(),
        y=y_train
    )
    class_weights_dict = dict(zip(y_train.unique(), class_weights_values))
    print(f"Class weights: {class_weights_dict}")

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        class_weight=class_weights_dict,
        verbose=1
    )
    return history

def save_model(model, model_path):
    model.save(model_path)
    print(f"Model saved to {model_path}.")

def save_training_history(history, history_file):
    history_dict = history.history
    with open(history_file, 'w') as f:
        json.dump(history_dict, f)
    print(f"Training history saved to {history_file}.")

if __name__ == "__main__":
    # Paths
    train_csv = './data/train.csv'
    model_save_path = './models/dnn_model.h5'
    history_save_path = './models/training_history.json'

    # Load data
    X_train, y_train = load_data(train_csv, label_column='label')
    input_dim = X_train.shape[1]
    print(f"Input dimension: {input_dim}")

    # Build model
    model = build_dnn_model(input_dim=input_dim)

    # Train model
    history = train_model(
        model,
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2
    )

    # Save the trained model
    save_model(model, model_save_path)

    # Save training history
    save_training_history(history, history_save_path)

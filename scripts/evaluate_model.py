import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def load_data(test_csv, label_column='label'):
    
    df = pd.read_csv(test_csv)
    X_test = df.drop(columns=[label_column])
    y_test = df[label_column]
    return X_test, y_test

def evaluate_model_performance(model, X_test, y_test):
    
    # Predict probabilities
    y_pred_prob = model.predict(X_test).ravel()

    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    cr = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(cr)

    # ROC-AUC
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"\nROC-AUC Score: {auc:.4f}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    plt.show()
    print("Confusion matrix plot saved as 'confusion_matrix.png'.")

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('models/roc_curve.png')
    plt.show()
    print("ROC curve plot saved as 'roc_curve.png'.")

def save_evaluation_report(cr, report_file):
    """
    Save the classification report to a text file.

    Parameters:
    - cr (str): Classification report as a string.
    - report_file (str): Path to save the report.
    """
    with open(report_file, 'w') as f:
        f.write(cr)
    print(f"Classification report saved to {report_file}.")

if __name__ == "__main__":
    # Paths
    test_csv = './data/test.csv'
    model_path = './models/dnn_model.h5'
    report_save_path = './models/classification_report.txt'

    # Load test data
    X_test, y_test = load_data(test_csv, label_column='label')
    print(f"Test data loaded: {X_test.shape[0]} samples.")

    # Load the trained model
    model = load_model(model_path)
    print("Trained model loaded successfully.")

    # Evaluate model performance
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'])
    auc = roc_auc_score(y_test, y_pred_prob)

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    print(f"\nROC-AUC Score: {auc:.4f}")

    # Save classification report
    save_evaluation_report(cr, report_save_path)

    # Plot Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig('./models/confusion_matrix.png')
    plt.show()
    print("Confusion matrix plot saved as 'confusion_matrix.png'.")

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('./models/roc_curve.png')
    plt.show()
    print("ROC curve plot saved as 'roc_curve.png'.")

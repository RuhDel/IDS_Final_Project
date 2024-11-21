# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from PIL import Image
import json
import base64
import os

# Set page configuration
st.set_page_config(
    page_title="IDS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.title("Intrusion Detection System Performance Dashboard")
st.markdown("""
This dashboard provides an overview of the performance metrics of our Intrusion Detection System. 
It visualizes key findings, including classification metrics, confusion matrix, ROC curve, and training history.
""")

# Function to load classification report
@st.cache_data
def load_classification_report(report_path):
    if not os.path.exists(report_path):
        st.error(f"Classification report not found at {report_path}. Please ensure the file exists.")
        return pd.DataFrame()  # Return empty DataFrame if file not found

    with open(report_path, 'r') as f:
        report = f.read()
    # Convert the report to a pandas DataFrame
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-4]:  # Adjust indices based on the report structure
        if line.strip() == '':
            continue
        parts = line.split()
        if len(parts) < 5:
            continue  # Skip lines that don't have enough parts
        class_name = parts[0]
        try:
            precision, recall, f1, support = float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])
            report_data.append({
                'Class': class_name,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Support': support
            })
        except ValueError:
            continue  # Skip lines with invalid numerical values
    return pd.DataFrame(report_data)

# Function to load training history
@st.cache_data
def load_training_history(history_path):
    if not os.path.exists(history_path):
        st.error(f"Training history not found at {history_path}. Please ensure the file exists.")
        return {}  # Return empty dict if file not found

    with open(history_path, 'r') as f:
        history = json.load(f)
    return history

# Function to create download link
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Load data
classification_report_df = load_classification_report('../data/classification_report.txt')
training_history = load_training_history('../models/training_history.json')

# Extract overall metrics
# Ensure that classification_report_df is not empty
if not classification_report_df.empty:
    accuracy = 0.98  # As per your report
    roc_auc = 0.9989  # As per your report

    # KPIs
    st.markdown("### Key Performance Indicators")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric(
        label="Accuracy",
        value=f"{accuracy * 100:.2f}%",
        delta="",
    )

    attack_precision = classification_report_df.loc[classification_report_df['Class'] == 'Attack', 'Precision'].values
    attack_recall = classification_report_df.loc[classification_report_df['Class'] == 'Attack', 'Recall'].values

    # Handle cases where 'Attack' class might not be present
    if len(attack_precision) > 0 and len(attack_recall) > 0:
        kpi2.metric(
            label="Precision (Attack)",
            value=f"{attack_precision[0] * 100:.2f}%",
            delta="",
        )

        kpi3.metric(
            label="Recall (Attack)",
            value=f"{attack_recall[0] * 100:.2f}%",
            delta="",
        )
    else:
        kpi2.metric(
            label="Precision (Attack)",
            value="N/A",
            delta="",
        )

        kpi3.metric(
            label="Recall (Attack)",
            value="N/A",
            delta="",
        )

    kpi4.metric(
        label="ROC-AUC Score",
        value=f"{roc_auc:.4f}",
        delta="",
    )

    # Confusion Matrix
    st.markdown("### Confusion Matrix")

    # Define confusion matrix values
    cm = np.array([[441961, 12659],
                   [545, 110984]])

    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'], ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Classification Report
    st.markdown("### Classification Report")

    st.dataframe(classification_report_df)

    # ROC Curve
    st.markdown("### ROC Curve")

    fig_roc, ax_roc = plt.subplots(figsize=(6,4))
    # Simulate y_test and y_pred_prob based on confusion matrix for demonstration
    # Replace this with actual y_test and y_pred_prob if available
    y_true = [0]*cm[0][0] + [1]*cm[1][0] + [0]*cm[0][1] + [1]*cm[1][1]
    y_pred_prob = [0.1]*cm[0][0] + [0.2]*cm[1][0] + [0.8]*cm[0][1] + [0.9]*cm[1][1]
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1], [0,1], 'k--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(fig_roc)

    # Training History
    st.markdown("### Training History")

    if training_history:
        # Plot Loss
        st.markdown("#### Loss Over Epochs")
        fig_loss, ax_loss = plt.subplots(figsize=(6,4))
        if 'loss' in training_history and 'val_loss' in training_history:
            ax_loss.plot(training_history['loss'], label='Training Loss')
            ax_loss.plot(training_history['val_loss'], label='Validation Loss')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_title('Loss Over Epochs')
            ax_loss.legend()
            st.pyplot(fig_loss)
        else:
            st.warning("Loss data not found in training history.")

        # Plot Accuracy
        st.markdown("#### Accuracy Over Epochs")
        fig_acc, ax_acc = plt.subplots(figsize=(6,4))
        if 'accuracy' in training_history and 'val_accuracy' in training_history:
            ax_acc.plot(training_history['accuracy'], label='Training Accuracy')
            ax_acc.plot(training_history['val_accuracy'], label='Validation Accuracy')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.set_title('Accuracy Over Epochs')
            ax_acc.legend()
            st.pyplot(fig_acc)
        else:
            st.warning("Accuracy data not found in training history.")
    else:
        st.warning("Training history is empty or not available.")

    # Detailed Metrics with Interactive Selection
    st.markdown("### Detailed Metrics")

    option = st.selectbox(
        'Select Metric to View:',
        ('Precision', 'Recall', 'F1-Score')
    )

    if not classification_report_df.empty:
        if option == 'Precision':
            fig, ax = plt.subplots()
            sns.barplot(x='Class', y='Precision', data=classification_report_df, ax=ax)
            ax.set_ylim(0,1)
            ax.set_title('Precision by Class')
            st.pyplot(fig)
        elif option == 'Recall':
            fig, ax = plt.subplots()
            sns.barplot(x='Class', y='Recall', data=classification_report_df, ax=ax)
            ax.set_ylim(0,1)
            ax.set_title('Recall by Class')
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            sns.barplot(x='Class', y='F1-Score', data=classification_report_df, ax=ax)
            ax.set_ylim(0,1)
            ax.set_title('F1-Score by Class')
            st.pyplot(fig)
    else:
        st.warning("Classification report data is not available to display detailed metrics.")

    # Download Classification Report
    if not classification_report_df.empty:
        st.markdown("### Download Reports")
        st.markdown(get_table_download_link(classification_report_df, 'classification_report.csv', 'Download Classification Report'), unsafe_allow_html=True)
    else:
        st.warning("Classification report data is not available to download.")

    # Footer
    st.markdown("""
    ---
    *Created with Streamlit*
    """)

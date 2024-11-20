# Intrusion Detection System (IDS) Project

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Streamlit Dashboard](#running-the-streamlit-dashboard)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [Dashboard](#dashboard)
- [Video Presentation](#video-presentation)

## Overview
Welcome to the **Intrusion Detection System (IDS) Project**! This project is developed as part of a cybersecurity course assignment with the primary goal of detecting system anomalies, intrusions, and various types of cyberattacks using advanced data processing and machine learning/deep learning techniques. The IDS simulates real-time network activities, processes the data, and presents findings through a user-friendly dashboard, providing valuable insights into network security.

### Key Features
- **Data Preprocessing:** Clean, normalize, and prepare the dataset for modeling.
- **Exploratory Data Analysis (EDA):** Analyze and visualize key features and patterns.
- **Feature Engineering:** Select and engineer features to optimize model performance.
- **Model Development:** Train and evaluate machine learning and deep learning models.
- **Real-Time Simulation:** Simulate network traffic to mimic real-world conditions.
- **Dashboard Presentation:** Visualize results using an interactive and intuitive Streamlit dashboard.
- **Video Presentation:** Summarize the project in a detailed video presentation.

## Project Structure
The IDS project follows a structured approach to ensure comprehensive detection and analysis:
IDS_Project/
- data/
  - classification_report.txt
  - combined.csv
  - preprocessed.csv
  - roc_curve.csv
  - test.csv
  - train.csv
- models/
  - dnn_model.h5
  - confusion_matrix.png
  - roc_curve.png
  - training_history.json
- dashboard/
  - app.py
- scripts/
  - preprocess.py
  - split_data.py
  - train_model.py
  - evaluate_model.py
- README.md
- requirements.txt

## Dataset
We are using the **CICIDS2017** dataset, which provides a realistic representation of network traffic, including both benign and attack data. The dataset is available on [Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset?resource=download).

### Dataset Features
- **Duration:** Length of the flow (in seconds).
- **Protocol:** Protocol used (e.g., TCP, UDP).
- **Source IP / Destination IP:** IP addresses involved in the traffic.
- **Source Port / Destination Port:** Port numbers used.
- **Packet Lengths:** Number of packets, bytes, etc.
- **Flow Features:** Statistical measures like mean, standard deviation of packet sizes, inter-arrival times.
- **Label:** Classification of the traffic (e.g., Normal, Attack types).

## Installation
To set up the project on your local machine, follow these steps:

### Prerequisites
- **Python 3.8 or higher**: Ensure Python is installed. You can download it from the [official website](https://www.python.org/downloads/).
- **Git**: For cloning the repository. Download from [here](https://git-scm.com/downloads).

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/IDS_Project.git
   cd IDS_Project
2. **Create a Virtual Environment (Recommended)**
It's best practice to use a virtual environment to manage dependencies.
   ```bash
   python -m venv venv
3. **Activate the Virtual Enviroment**
- Windows:
  ```bash
  venv\Scripts\activate
- macOS/Linux:
  ```bash
  source venv/bin/activate
4. **Install Dependencies**
- Navigate to the project root and install required packages using requirements.txt
  ```bash
  pip install --upgrade pip
  pip install -r requirements
  
## Usage
### Running the streamlit dashboard
The dashboard provides an interactive interface to visualize the IDS performance metrics.
1. Navigate to the Dashboard Directory
   ```bash
   cd dashboard
3. Run the Streamlit App
   ```bash
   streamlit run app.py
   or
   python -m streamlit run app.py
5. Access the Dashboard
   After running the command, Streamlit will provide a local URL (e.g., http://localhost:8501). Open this URL in your web browser to view the dashboard.
   
### Training the Model
If you wish to train the model yourself:
1. Navigate to the Scripts Directory
   ```bash
   cd scripts
2. Run the Evaluation Script
   ```bash
   python train_model.py
   
### Evaluating the Model
To evaluate the trained model:
1. Ensure the Model is Trained
   Make sure train_model.py has been executed and dnn_model.h5 is available in the **models/** directory.
2. Run the Evaluation Script
   ```bash
   python evaluate_model.py
  This script generates evaluation reports, including confusion matrices and ROC curves, and saves them in the **data/** and **models/** directories.

## Dashboard
The Streamlit dashboard offers a comprehensive view of the IDS's performance, featuring:
- **Key Performance Indicators (KPIs)**: Displaying Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Confusion Matrix**: Visual representation of true vs. predicted classifications.
- **ROC Curve**: Illustrating the trade-off between true positive rate and false positive rate.
- **Training History**: Graphs showing loss and accuracy over training epochs.
- **Detailed Metrics**: Interactive selection of Precision, Recall, and F1-Score by class.
- **Download Reports**: Option to download classification reports in CSV format.

## Video Presentation
We have created a 15-20 minute video presentation that summarizes our approach, methodology, model development, and dashboard results. The presentation provides an in-depth walkthrough of the project, highlighting key insights and demonstrating the dashboard in action.

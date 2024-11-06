# Intrusion Detection System (IDS) Project

## Overview
This project is an Intrusion Detection System (IDS) developed as part of a cybersecurity course assignment. The IDS is designed to detect system anomalies, intrusions, and various types of cyberattacks using advanced data processing and machine learning/deep learning techniques. The project simulates real-time network activities, processes the data, and presents findings through a user-friendly dashboard.

## Project Structure
The IDS project follows a structured approach to ensure comprehensive detection and analysis:
1. **Data Preprocessing**: Clean, normalize, and prepare the dataset for modeling.
2. **Exploratory Data Analysis (EDA)**: Analyze and visualize key features and patterns.
3. **Feature Engineering**: Select and engineer features to optimize model performance.
4. **Model Development**: Train and evaluate machine learning and deep learning models.
5. **Real-Time Simulation**: Simulate network traffic to mimic real-world conditions.
6. **Dashboard Presentation**: Visualize results using an interactive and intuitive dashboard.
7. **Video Presentation**: Summarize the project in a detailed video presentation.

## Dataset
We are using the **CICIDS2017** dataset, which provides a realistic representation of network traffic, including both benign and attack data. The dataset is available on [Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset?resource=download).

## Installation and Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourUsername/Intrusion-Detection-System.git
   cd Intrusion-Detection-System
2. **Create a Virtual Enviroment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows
3. **Install Depencencies**:
   ```bash
   pip install -r requirements.txt
4. **Download and Place the Dataset**
   - Download the CICIDS2017 dataset from [Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset?resource=download) and place it in the data folder.

## Usage
1. **Data Preprocessing**:
   - Run the `data_preprocessing.py` script to clean and prepare the data.
     ```bash
     python data_preprocessing.py
2. **Model Training**:
   - Train model using the `model_training.py` script.
     ```bash
     python model_training.py
3. **Real-Time Simulation**:
   - Simulate real-time network trafic using the `trafic_simulation.py` script.
     ```bash
     python traffic_simulation.py
4. **Dashboard**:
   - Launch the bashboard to visualize using Streamlit.
     ```bash
     streamlit run dashboard.py

## Project Components
1. **Data Preprocessing**:
   - **Scripts**: `data_preprocessing.py`
   - **Description**: Loads, cleans, and normalizes the dataset. Encodes categorical features if necessary and saves the preprocessed data for model training.
2. **Exploratory Data Analysis (EDA)**:
   - **Scripts**: `eda.py`
   - **Description**: Performs statistical analysis and visualizes feature distributions, correlations, and class imbalances.
3. **Data Preprocessing**:
   - **Scripts**: `model_training.py`
   - **Description**: Implements machine learning and deep learning models. Includes hyperparameter tuning and performance evaluation using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
4. **Data Preprocessing**:
   - **Scripts**: `traffic_simulation.py`
   - **Description**: Simulates real-time network activity to test the IDS in a dynamic environment.
5. **Data Preprocessing**:
   - **Scripts**: `dashboard.py`
   - **Description**: Interactive dashboard created with Streamlit to display intrusion detection results, metrics, and visualizations.

## Results
The IDS effectively detects various types of cyberattacks with high accuracy. Detailed results and performance metrics are available on the dashboard.

## Video Presentaion
A comprehensive video presentation summarizing our approach, methodology, and findings will be submitted through the Bongo platform.

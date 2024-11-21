# Intrusion Detection System (IDS) Project

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
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
  - roc_curve.png
  - (Downloaded Data set)
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
2. **Download Dataset**
   Download the IDS2017 Dataset from [Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset?resource=download)
3. **Create a Virtual Environment (Recommended)**
It's best practice to use a virtual environment to manage dependencies.
   ```bash
   python -m venv venv
4. **Activate the Virtual Enviroment**
- Windows:
  ```bash
  venv\Scripts\activate
- macOS/Linux:
  ```bash
  source venv/bin/activate
5. **Install Dependencies**
- Navigate to the project root and install required packages using requirements.txt
  ```bash
  pip install --upgrade pip
  pip install -r requirements
  
## Usage
### Combine the Dataset
This step consolidates all the individual CSV files into a single dataset for ease of processing and analysis.
- Run the script **combine_csv.py** to merge the files:
   ```bash
   python combine_csv.py
**Output:** A single CSV file named *combined.csv* containing data from all input files.
   
### Preprocess the Combined Dataset
Preprocessing cleans and standardizes the *combined.csv* data by handling missing values, encoding categorical variables, and scaling numeric features as necessary.
- Run the script **preprocess.py** to prepare the data for modeling:
   ```bash
   python preprocess.py
**Output:** A preprocessed dataset saved as *preprocessed.csv*.


### Split Data
This step divides the preprocessed dataset into two subsets: 80% for training and 20% for testing. This ensures that the model can be evaluated on unseen data.
- Run the script **split_data.py** to split the dataset:
   ```bash
   python split_data.py
**Output:** Two CSV files, *train.csv* and *test.csv*, containing the training and testing data, respectively.

### Train Model
Train the Intrusion Detection System model using the training data. This step utilizes deep learning algorithms to build a model capable of detecting anomalies or intrusions.
- Run the script **train_model.py** to train the model:
   ```bash
   python train_model.py
**Output:** A trained model saved as *dnn_model.h5* in the **models/** directory.
  
### Evaluating the Model
Evaluate the performance of the trained model using test data. This step generates metrics like confusion matrices, classification reports, and ROC curves to measure the model's effectiveness.
1. Ensure the model is trained:
   Confirm that **train_model.py** has been executed and the file *dnn_model.h5* exists in the **models/** directory.
2. Run the evaluation script
   ```bash
   python evaluate_model.py
**Output:** Evaluation results, including performance reports and visualizations, saved in the **data/** and **models/** directories for further analysis.

### Running the Streamlit Dashboard
The Streamlit dashboard provides an interactive user interface to visualize and analyze the IDS performance metrics, such as accuracy, precision, recall, and other evaluation results.
1. Navigate to the **dashboard/** Directory:
   ```bash
   cd dashboard
3. Run the Streamlit App
   ```bash
   streamlit run app.py
   or
   python -m streamlit run app.py
5. Access the Dashboard
   After running the command, Streamlit will provide a local URL (e.g., http://localhost:8501). Open this URL in your web browser to view the dashboard.

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

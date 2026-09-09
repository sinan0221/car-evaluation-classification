# 🚗 Car Evaluation Classification

## 📌 Project Overview

This project focuses on **Car Evaluation Classification** using Machine Learning.

The goal is to classify cars into different evaluation categories based on features such as buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety level.

Three Machine Learning classification algorithms are applied and compared to identify the best-performing model.

## 🎯 Problem Statement

To develop a Machine Learning model that can classify cars into different categories:

- Unacceptable (`unacc`)
- Acceptable (`acc`)
- Good (`good`)
- Very Good (`vgood`)

based on the characteristics of the car.

## 📊 Dataset

The project uses the **Car Evaluation Dataset**.

### Features

| Feature | Description |
|---|---|
| Buying | Buying price of the car |
| Maintenance | Maintenance price |
| Doors | Number of doors |
| Persons | Passenger capacity |
| Lug Boot | Luggage boot size |
| Safety | Safety level |

### Target

**Car Evaluation Class**

- `unacc`
- `acc`
- `good`
- `vgood`

## 🤖 Machine Learning Algorithms

The following three classification algorithms were used:

### 1. K-Nearest Neighbors (KNN)

KNN classifies a car based on the classes of its nearest data points.

### 2. Random Forest

Random Forest uses multiple decision trees and combines their predictions to improve classification performance.

### 3. Support Vector Machine (SVM)

SVM finds an optimal boundary to separate different car evaluation classes.

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Checked the dataset structure.
3. Checked for missing values.
4. Removed duplicate records.
5. Encoded categorical features using Label Encoding.
6. Split the dataset into training and testing sets.
7. Applied Standard Scaling for KNN and SVM.

## 📈 Evaluation Metrics

The models were compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 🏆 Results

After training and evaluating the three models, **Random Forest achieved the highest accuracy** among the tested models.

| Model | Performance |
|---|---|
| K-Nearest Neighbors | Compared using Accuracy, Precision, Recall and F1 Score |
| Random Forest | 🏆 Best Performing Model |
| Support Vector Machine | Compared using Accuracy, Precision, Recall and F1 Score |

### Best Model

**Random Forest Classifier**

Random Forest was selected as the best-performing algorithm based on the highest accuracy obtained during testing.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook / VS Code

## 📁 Project Structure

```text
CAR-EVALUTION/
│
├── car_evaluation.csv
├── tapp.py
├── README.md
└── .gitignore

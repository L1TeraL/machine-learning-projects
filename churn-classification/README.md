# Customer Churn Classification

A machine learning project for predicting customer churn in a telecommunications company.

## Project Overview

The goal is to predict whether a customer will leave the company based on their characteristics and service usage.

## Methods

- Data cleaning and preprocessing
- Categorical feature encoding
- Handling class imbalance with RandomOverSampler
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Stratified cross-validation
- Hyperparameter tuning
- Model evaluation using precision, recall, F1-score, and a confusion matrix

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- Matplotlib
- Seaborn

## Dataset

The dataset is stored in `data/Telco-Customer-Churn.csv`.

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the script from this directory:

```bash
python main.py
```

## Visualizations

Generated visualizations are stored in the `images/` directory.
# online-payment-fraud-detection

This project aims to build a machine learning model for detecting fraud in online payment transactions. The dataset contains millions of transactions with various features, and the goal is to classify whether a transaction is fraudulent (`isFraud` column) based on these features.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dependencies](#dependencies)
3. [Dataset Description](#dataset-description)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training](#model-training)
6. [Model Evaluation](#model-evaluation)
7. [Results](#results)
8. [Conclusion](#conclusion)

## Project Overview

This project leverages machine learning algorithms to detect fraudulent transactions in online payment systems. The dataset consists of transaction records where the task is to classify whether the transaction is fraudulent (target column: `isFraud`). The following algorithms were used for classification:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

## Dependencies

To run this project, you need to install the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

You can install the required dependencies by running:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

## Dataset

The dataset used in this project is available for download from [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/) (please replace the URL with the correct one).

Alternatively, you can download the dataset manually and place it in the project folder.

### Example:
```python
data = pd.read_csv("path/to/PS_20174392719_1491204439457_log.csv")


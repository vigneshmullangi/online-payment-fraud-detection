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

## bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost


## Model Evaluation

After training the models, we evaluated them using the **ROC-AUC score** to measure their performance. Below are the results for each model:

### Logistic Regression
- **Training Accuracy**: 88.89%
- **Validation Accuracy**: 88.81%

### XGBoost Classifier
- **Training Accuracy**: 99.99%
- **Validation Accuracy**: 99.90%

### Random Forest Classifier
- **Training Accuracy**: 99.99%
- **Validation Accuracy**: 95.32%

These evaluation metrics demonstrate that **XGBoost** performed the best, achieving nearly perfect training and validation accuracy. However, the **Random Forest** model also showed strong results, particularly with a solid validation accuracy of 95.32%. **Logistic Regression**, while slightly lower in performance, still achieved a respectable accuracy of around 88% on both training and validation data.

We further visualized the performance of the best model (XGBoost) using a **confusion matrix**, which helped to assess the true positive and false positive rates.


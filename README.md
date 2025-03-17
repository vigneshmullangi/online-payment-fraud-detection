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
--pip install numpy pandas matplotlib seaborn scikit-learn xgboost


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

## Results

The models were evaluated using various performance metrics, with a particular focus on the **ROC-AUC score**. The results from the models show promising performance, especially with the **XGBoost** model, which provided the highest accuracy.

- **XGBoost** achieved excellent validation accuracy (99.90%) and training accuracy (99.99%), making it the most effective model for this task.
- **Random Forest** showed a strong validation accuracy of 95.32%, which also demonstrates its suitability for this type of classification task.
- **Logistic Regression**, although not as high-performing as the others, still achieved an accuracy of 88.81% on validation data, which is reasonable given the nature of the dataset and its features.

These results suggest that the models can effectively identify fraudulent transactions. The dataset's imbalance (with far fewer fraudulent transactions than non-fraudulent ones) did not significantly hinder model performance, but further work can be done to address this imbalance (e.g., through resampling techniques).

## Conclusion

This project successfully demonstrates the application of machine learning models for detecting fraud in online payment systems. By leveraging algorithms such as **Logistic Regression**, **Random Forest**, and **XGBoost**, we were able to build models that can classify transactions as fraudulent or non-fraudulent with high accuracy.

### Key Takeaways:
- **XGBoost** performed the best, achieving nearly perfect accuracy on both training and validation data.
- **Random Forest** also showed promising results, especially in terms of generalization to unseen data.
- **Logistic Regression**, while less effective, still offered a valuable baseline.

## Dataset

You can download the dataset by clicking the button below:

<a href="https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset" target="_blank">

</a>




# Breast Cancer Prediction using Machine Learning Classification

This project focuses on predicting whether a breast tumor is **Benign** or **Malignant** using multiple machine learning classification algorithms. 
Various models were trained, evaluated, and compared to identify the best-performing classifier.

## Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier
- Gaussian Naive Bayes
- Stacking Classifier (Ensemble Learning)

## Project Workflow

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Encoding
- Feature Scaling
- Model Training
- Hyperparameter Tuning using GridSearchCV
- Ensemble Learning using Staking Classifier
- Model Evaluation and Comparison

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

## Evaluation Metrics

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

## Results

- Compared the performance of seven classification algorithms.
- Support Vector Machine (SVM) achieved the highest test accuracy of **98.25%**.
- Hyperparameter tuning was performed using **GridSearchCV** to optimize model performance.
- Stacking Ensemble Learning was implemented using multiple base learners and different meta-learners. 
  The ensemble produced performance similar to the best individual classifier, 
  indicating that the base models generated highly similar predictions on this dataset.
